"""Train ONLY the new correction, retaining epoch0 and exact epoch-boundary resume."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import seed_all, device_from_name
from baselines.B5_best.common import write, sha, state_sha
from baselines.B5_best.bundle import load_bundle
from baselines.B5_final.profile import BASE_SHA, BODY_SHA
from . import PROTOCOL, GROUPS
from .common import load_study, load_correction, Records
from .model import FullSlowCNN, inputs
from .losses import objective
from .evaluate import Evaluator


def run(args):
    root, output = Path(args.run).resolve(), Path(args.output).resolve()
    study = load_study(root)
    cfg = study["training"]
    device = device_from_name(args.device)
    seed_all(cfg["seed"], True)
    model = FullSlowCNN(args.group, **study["model"]).to(device)
    model.initialize_from_s50(load_bundle(study["base_checkpoint"])["slow"]["model"])
    if state_sha(model.state_dict()) != study["initial_weights_sha256"]:
        raise ValueError("Initializer differs from prepared matched arms")
    # Freeze the established S50 backbone: this screening isolates the effect
    # of the two newly added residual blocks and their receptive field.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("extra."))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(Records(root, study["train"]), batch_size=cfg["batch_records"], shuffle=True,
                        generator=generator, num_workers=0)
    scales = torch.tensor(study["scale_uV"], device=device, dtype=torch.float32)
    start, updates, best_epoch, best_r, best_row = 1, 0, 0, None, None
    if args.resume:
        if Path(args.resume).resolve() != output / "last.pt":
            raise ValueError("Exact resume requires this output directory's last.pt")
        _, cp = load_correction(args.resume, study, root / "study.json", device)
        if cp["group"] != args.group or cp["output_dir"] != str(output):
            raise ValueError("Resume must keep the arm and output directory")
        _, selected = load_correction(output / "best.pt", study, root / "study.json", device)
        if selected["group"] != args.group:
            raise ValueError("Best checkpoint belongs to another arm")
        if selected["epoch"] > cp["epoch"]:
            raise ValueError("Best checkpoint is newer than the resume checkpoint")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        generator.set_state(cp["loader_rng"].cpu())
        start, updates = cp["epoch"] + 1, cp["updates"]
        best_r, best_epoch, best_row = selected["best_r"], selected["best_epoch"], selected["validation"]
        restore_rng(cp["rng"])
    else:
        output.mkdir(parents=True, exist_ok=False)
        write(output / "initialization.json", {"protocol": PROTOCOL, "group": args.group,
            "training": cfg, "sampling": study["sampling"], "initial_weights_sha256": study["initial_weights_sha256"],
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "epoch_zero_is_candidate": True, "FM_optimizer_updates": 0, "P_optimizer_updates": 0,
            "body_optimizer_updates": 0, "selection": "full120_with_fixed_context_mean_r"})
    # Loading a frozen body model constructs temporary randomly initialized
    # layers. Do not let that consume training RNG, especially after resume.
    construction_rng = pack_rng()
    try:
        evaluator = Evaluator(root, study, device)
    finally:
        restore_rng(construction_rng)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def payload(epoch, validation):
        return {"protocol": PROTOCOL, "group": args.group, "study_sha256": sha(root / "study.json"),
            "base_sha256": BASE_SHA, "body_sha256": BODY_SHA, "output_dir": str(output),
            "epoch": epoch, "updates": updates, "best_r": best_r, "best_epoch": best_epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "loader_rng": generator.get_state(), "rng": pack_rng(), "validation": validation,
            "training_seed": cfg["seed"], "torch_version": str(torch.__version__)}
    def validate(epoch):
        rng = pack_rng()
        folder = output / "validation" / f"epoch_{epoch:04d}"
        if folder.exists():
            folder = folder.with_name(folder.name + "_retry_" + str(time.time_ns()))
        try:
            return evaluator.evaluate(model, folder, args.group, epoch, verify_zero=False)
        finally:
            restore_rng(rng)
    try:
        if not args.resume:
            best_row = validate(0)
            best_r = best_row["mean_r"]
            atomic_save(output / "best.pt", payload(0, best_row))
            atomic_save(output / "last.pt", payload(0, best_row))
        for epoch in range(start, cfg["epochs"] + 1):
            model.train()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begun = time.perf_counter()
            totals, count = {}, 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                delta = model(**inputs(batch))
                loss, parts = objective(delta, batch, scales)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                n = len(delta)
                for k, v in {"loss": loss, **parts}.items():
                    totals[k] = totals.get(k, 0.) + float(v.detach()) * n
                count += n
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - begun
            validation, improved = None, False
            if epoch % cfg["validate_every"] == 0 or epoch == cfg["epochs"]:
                validation = validate(epoch)
                improved = validation["mean_r"] > best_r
                if improved:
                    best_r, best_epoch, best_row = validation["mean_r"], epoch, validation
            record = {"epoch": epoch, "updates": updates, "train": {k: v / count for k, v in totals.items()},
                "validation": validation, "best_r": best_r, "best_epoch": best_epoch,
                "selected_best": improved, "train_seconds": elapsed,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "FM_optimizer_updates": 0, "P_optimizer_updates": 0, "body_optimizer_updates": 0}
            if improved:
                atomic_save(output / "best.pt", payload(epoch, validation))
            atomic_save(output / "last.pt", payload(epoch, validation))
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(record, flush=True)
        write(output / "completed.json", {"protocol": PROTOCOL, "completed": True, "group": args.group,
            "epochs": cfg["epochs"], "updates": updates, "best_epoch": best_epoch,
            "best_r": best_r, "best": best_row, "FM_optimizer_updates": 0, "P_optimizer_updates": 0})
    finally:
        evaluator.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Prepared shared directory containing study.json")
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", help="Same output's last.pt; resumes the next complete epoch")
    p.add_argument("--execute-training", action="store_true")
    args = p.parse_args()
    if not args.execute_training:
        p.error("No training started; add --execute-training on the server")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    run(args)


if __name__ == "__main__":
    main()
