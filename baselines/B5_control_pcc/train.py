"""A/B differ only by TRAIN FM cache; B/C differ only by multiscale supervision."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best import SAMPLING
from baselines.B5_best.common import write, sha, state_sha
from . import PROTOCOL, GROUPS
from .common import load_study, load_cache, load_checkpoint, slow_model, Records, multiscale_for_group
from .core import generate, device_batch, model_inputs
from .evaluate import evaluate
from .losses import objective


def append_metrics(output, epoch, score):
    row = {"epoch": epoch, **score}
    path = output / "validation_metrics.csv"
    existed = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not existed:
            writer.writeheader()
        writer.writerow(row)


def run(study_path, cache_root, output, group, device_name, resume=None):
    study_path, cache_root, output = map(lambda p: Path(p).resolve(), (study_path, cache_root, output))
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use empty output; put train log beside it")
    study, bundle = load_study(study_path)
    manifest = load_cache(cache_root, study_path)
    cache_sha = sha(cache_root)
    cfg = study["training"]
    device = device_from_name(device_name)
    seed_all(cfg["seed"], True)
    model = slow_model(bundle, device, trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_source = manifest["train_sources"][group]
    loader = DataLoader(Records(train_source["root"], train_source["entries"]), batch_size=cfg["batch_records"],
                        shuffle=True, generator=generator, num_workers=0)
    scales = torch.tensor(bundle["fm"]["scales"]["d12"], device=device)
    combo_weight = study["combo_weight"]
    multiscale = multiscale_for_group(study, group)
    start, updates, best_r, best_epoch = 1, 0, None, None
    fm_digest = bundle["fm_state_sha256"]
    if resume:
        checkpoint, _, _, _ = load_checkpoint(resume, study_path, device)
        if (checkpoint["group"] != group or checkpoint["output_dir"] != str(output) or
                checkpoint["cache_sha256"] != cache_sha):
            raise ValueError("Resume must preserve group, cache and output directory")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["loader_rng"].cpu())
        start, updates = checkpoint["epoch"] + 1, checkpoint["updates"]
        best_r, best_epoch = checkpoint["best_r"], checkpoint["best_epoch"]
        restore_rng(checkpoint["rng"])
        if (output / "best.pt").exists():
            previous = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            if previous["study_sha256"] != sha(study_path) or previous["group"] != group:
                raise ValueError("Existing best belongs to another experiment")
            if previous["best_r"] > best_r:
                best_r, best_epoch = previous["best_r"], previous["best_epoch"]
    else:
        output.mkdir(parents=True, exist_ok=True)
        write(output / "initialization.json", {"group": group, "study_sha256": sha(study_path),
              "cache_sha256": cache_sha, "initial_slow_state_sha256": state_sha(model.state_dict()),
              "fixed_FM_state_sha256": fm_digest, "optimizer_initialization": "fresh_AdamW_same_all_arms",
              "training": cfg, "old_F_weight": study["residual_weight"], "mean_weight": 0.,
              "combo_PCC_weight": combo_weight, "sampling": dict(SAMPLING), "beta": study["beta"],
              'multiscale': multiscale,
              "selection": "final_hybrid_main_correlation", "epoch_zero_is_candidate": True,
              "score_protocol": study["score_protocol"], "train_FM_cache_root": train_source["root"],
              "validation_FM_same_all_arms": "control100", "require_reliable_input": True,
              "FM_training_started": False})
        seed_all(cfg["seed"], True)

    def payload(epoch, score):
        return {"protocol": PROTOCOL, "group": group, "study_sha256": sha(study_path),
                "cache_sha256": cache_sha, "output_dir": str(output), "epoch": epoch, "updates": updates,
                "best_r": best_r, "best_epoch": best_epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "loader_rng": generator.get_state(), "rng": pack_rng(),
                "sampling_config": dict(SAMPLING), "validation_score": score,
                "fixed_FM_state_sha256": fm_digest, "combo_PCC_weight": combo_weight,
                'multiscale': multiscale,
                "score_protocol": study["score_protocol"], "torch_version": str(torch.__version__)}

    def validation_at(epoch):
        rng = pack_rng()
        try:
            folder = output / "validation" / f"epoch_{epoch:04d}"
            if folder.exists():
                import time
                folder = folder.with_name(folder.name + "_attempt_" + str(time.time_ns()))
            print({"phase": "cached_FM_hybrid_validation", "epoch": epoch, "group": group}, flush=True)
            return evaluate(model, bundle, study, cache_root, manifest, device, folder, group, epoch, initial=(epoch == 0))
        finally:
            restore_rng(rng)

    if not resume:
        score = validation_at(0)
        best_r, best_epoch = score["main_correlation"], 0
        append_metrics(output, 0, score)
        checkpoint = payload(0, score)
        atomic_save(output / "best.pt", checkpoint)
        atomic_save(output / "last.pt", checkpoint)
        print({"epoch": 0, "initial_hybrid_r": best_r}, flush=True)
    for epoch in range(start, cfg["epochs"] + 1):
        model.train()
        totals, count = {}, 0
        for batch in loader:
            inputs = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = generate(model, model_inputs(inputs))
            loss, parts = objective(prediction, inputs, scales, study["residual_weight"], combo_weight,
                                    study["residual_width_native"], beta=study["beta"], multiscale=multiscale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            n = len(prediction)
            for key, value in {"loss": loss, **parts}.items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * n
            count += n
        record = {"epoch": epoch, "updates": updates, "train": {k: v / count for k, v in totals.items()}}
        score, improved = None, False
        if epoch % cfg["validate_every"] == 0 or epoch == cfg["epochs"]:
            score = validation_at(epoch)
            append_metrics(output, epoch, score)
            improved = score["main_correlation"] > best_r
            if improved:
                best_r, best_epoch = score["main_correlation"], epoch
            record["validation"] = score
        record.update(best_r=best_r, best_epoch=best_epoch, selected_best=improved)
        # FM weights are never on the optimizer/graph; this also checks the shared payload is untouched.
        if state_sha(bundle["fm"]["ema"]) != fm_digest:
            raise ValueError("Frozen FM state unexpectedly changed")
        checkpoint = payload(epoch, score)
        if improved:
            atomic_save(output / "best.pt", checkpoint)
        atomic_save(output / "last.pt", checkpoint)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
        print(json.dumps(record, allow_nan=False, ensure_ascii=False), flush=True)
    write(output / "completed.json", {"completed": True, "group": group, "epochs": cfg["epochs"],
          "best_r": best_r, "best_epoch": best_epoch, 'multiscale_weight': multiscale['weight'],
          "FM_training_started": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume")
    p.add_argument("--execute-training", action="store_true")
    args = p.parse_args()
    if not args.execute_training:
        p.error("Training was not started; pass --execute-training on the server")
    run(args.study, args.cache, args.output, args.group, args.device, args.resume)


if __name__ == "__main__":
    main()
