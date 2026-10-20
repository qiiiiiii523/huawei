"""Optional paired continuation: SAME real40s blocks/exposure, original full-wave FM loss."""
import argparse
from pathlib import Path
import math
import numpy as np
import torch

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet
from baselines.B5.losses import flow_loss
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import write, sha, state_sha
from baselines.B5_final.profile import BASE_SHA
from . import PROTOCOL, FIELDS, MODES
from .common import (load_study, read_record, code_sha, candidate_payload_matches)
from .evaluate import Evaluator


def schedule(seed, update, records, count):
    """Stateless schedule shared by10s and40s; no validation records enter here."""
    if min(update, records, count) < 1:
        raise ValueError("Positive update/record count required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, update]))
    return [(int(rng.integers(records)), int(rng.integers(3)) * 20000) for _ in range(count)]


def training_randomness(seed, update, slot):
    # Separate CPU RNG keeps data/time/path noise invariant to model forward count/dropout.
    sequence = np.random.SeedSequence([seed, update, slot, 4010])
    key = int(sequence.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))
    gen = torch.Generator(device="cpu").manual_seed(key)
    noise = torch.randn((1, 11, 20000), generator=gen)
    time = torch.rand((1,), generator=gen)
    return noise, time


def microbatches(data, start, seconds, scales, noise, time, device):
    if seconds not in (10, 40) or start not in (0, 20000, 40000):
        raise ValueError("Use10s or40s slices of a real nonoverlapping40s block")
    width = seconds * 500
    scale = np.asarray(scales, np.float32)
    for offset in range(0, 20000, width):
        span = slice(start + offset, start + offset + width)
        batch = {key: torch.from_numpy(np.asarray(data[key])[None].copy()).to(device) for key in FIELDS}
        batch.update(anchor=torch.from_numpy((data["anchor_uV"][:, span] / scale[0])[None].copy()).to(device),
                     target=torch.from_numpy((data["target_uV"][:, span] / scale[:, None])[None].copy()).to(device),
                     quality_mask=torch.from_numpy(data["mask"][None].copy()).to(device))
        yield batch, noise[..., offset:offset + width].to(device), time.to(device)


def fresh_evaluation_path(output, update):
    path = output / "validation" / f"update_{update:06d}"
    attempt = 0
    while path.exists():
        attempt += 1
        path = output / "validation" / f"update_{update:06d}_attempt{attempt}"
    return path


def run(args):
    root, output = Path(args.run).resolve(), Path(args.output).resolve()
    study = load_study(root)
    settings = study["experiment"]["training"]
    if not args.execute_training:
        raise ValueError("Training is opt-in: add --execute-training on the training server")
    if args.window_seconds not in (10, 40) or args.batch_size < 1:
        raise ValueError("Use10s/40s and a positive inference batch size")
    if output.exists() and not args.resume:
        raise FileExistsError("Use a new group directory or explicit --resume")
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("Resume requires this group's last.pt")
    device = device_from_name(args.device)
    seed_all(settings["seed"], True)
    bundle = load_bundle(study["base_checkpoint"])
    model = B5UNet(ModelConfig.from_dict(study["model_config"])).to(device)
    model.load_state_dict(bundle["fm"]["ema"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best, update = {mode: None for mode in MODES}, 0
    if args.resume:
        old = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        candidate_payload_matches(old, study, root / "study.json")
        if old["window_seconds"] != args.window_seconds:
            raise ValueError("Cannot resume another window length")
        model.load_state_dict(old["model"], strict=True)
        optimizer.load_state_dict(old["optimizer"])
        ema = {key: value.to(device) for key, value in old["ema"].items()}
        best, update = old["best"], old["update"]
        restore_rng(old["rng"])
    output.mkdir(parents=True, exist_ok=True)
    write(output / "completed.json", {"completed": False, "window_seconds": args.window_seconds})
    write(output / "training_contract.json", {"protocol": PROTOCOL, "study_sha256": sha(root / "study.json"),
        "window_seconds": args.window_seconds, "training": settings, "fm_loss": study["fm_loss"],
        "seconds_per_update": settings["records_per_update"] * 40,
        "same_train_record_and40s_block_schedule": True, "path_noise_and_time_paired": True,
        "condition_dropout_draws_identical": False, "context": "disabled", "slow_optimizer_updates": 0,
        "initial_fm_state_sha256": study["initial_fm_state_sha256"], "new_optimizer": not args.resume})
    evaluator = Evaluator(root, study, device, args.batch_size)
    scale_tensor = torch.tensor(study["scale_uV"], device=device)

    def payload():
        return {"protocol": PROTOCOL, "window_seconds": args.window_seconds,
            "base_sha256": BASE_SHA, "study_sha256": sha(root / "study.json"), "code_sha256": code_sha(),
            "model_config": study["model_config"], "scale_uV": study["scale_uV"], "fm_loss": study["fm_loss"],
            "experiment": study["experiment"], "update": update, "best": best,
            "model": model.state_dict(), "ema": ema, "state_sha256": state_sha(ema),
            "optimizer": optimizer.state_dict(), "rng": pack_rng(),
            "initial_fm_state_sha256": study["initial_fm_state_sha256"],
            "training_waveform_seconds": update * settings["records_per_update"] * 40}

    def validate():
        nonlocal best
        rng = pack_rng()
        evaluation_model = B5UNet(ModelConfig.from_dict(study["model_config"])).to(device)
        evaluation_model.load_state_dict(ema, strict=True)
        try:
            report = evaluator.evaluate(evaluation_model.eval(), args.window_seconds,
                                        fresh_evaluation_path(output, update), "trained", update)
        finally:
            del evaluation_model
            restore_rng(rng)
        improved = []
        for mode in MODES:
            value = report["summary"][mode]["mean_r"]
            if not math.isfinite(value):
                raise ValueError("Nonfinite full-record selection metric")
            if best[mode] is None or value > best[mode]["mean_r"]:
                best[mode] = {"mean_r": value, "update": update, "summary": report["summary"][mode]}
                improved.append(mode)
        state = payload()
        for mode in improved:
            atomic_save(output / ("best_" + mode + ".pt"), state)
        atomic_save(output / "last.pt", state)
        return report

    if not args.resume:
        validate()  # Includes unchanged initialization in BOTH candidate sets.
    for update in range(update + 1, settings["updates"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        selection = schedule(settings["seed"], update, len(study["train"]), settings["records_per_update"])
        divisor = settings["records_per_update"] * (40 // args.window_seconds)
        total, records = {}, []
        for slot, (index, start) in enumerate(selection):
            entry = study["train"][index]
            data = read_record(root, entry)
            records.append({"target_record_id": entry["target_record_id"], "start_sample": start})
            noise, time = training_randomness(settings["seed"], update, slot)
            for batch, z, t in microbatches(data, start, args.window_seconds, study["scale_uV"], noise, time, device):
                loss, parts = flow_loss(model, batch, study["fm_loss"], scale_tensor, noise=z, time=t)
                (loss / divisor).backward()
                for key, value in {"loss": loss, **parts}.items():
                    total[key] = total.get(key, 0.) + float(value.detach()) / divisor
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for key, value in model.state_dict().items():
                ema[key].mul_(settings["ema_decay"]).add_(value, alpha=1 - settings["ema_decay"])
        row = {"update": update, "window_seconds": args.window_seconds, "train": total,
               "records": records, "seconds_seen": update * settings["records_per_update"] * 40}
        if update % settings["validate_every"] == 0 or update == settings["updates"]:
            row["validation"] = validate()["summary"]
        import json
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(row, flush=True)
    write(output / "completed.json", {"completed": True, "window_seconds": args.window_seconds,
          "updates": update, "best": best, "context": "disabled", "slow_optimizer_updates": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--window-seconds", required=True, type=int, choices=(10, 40))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size, not training exposure")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
