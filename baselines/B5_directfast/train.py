"""Matched central-window update budgets for F1/D1; exact epoch-boundary resume."""
import argparse
import json
from pathlib import Path
import time
import torch

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.losses import flow_loss
from baselines.B5.runtime import seed_all, device_from_name
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_final.profile import BASE_SHA, BODY_SHA
from . import PROTOCOL, GROUPS
from .common import load_study, load_candidate, code_sha
from .data import TrainingWindows
from .model import build_model
from .losses import direct_objective
from .evaluate import Evaluator


def train(args):
    root, output = Path(args.run).resolve(), Path(args.output).resolve()
    study = load_study(root)
    cfg = study["experiment"]["training"]
    f0 = read(root.parent / "F0" / "evaluation.json")
    if (f0.get("completed") is not True or f0.get("group") != "F0"
            or f0.get("population") != study["population"]
            or abs(f0["summary"]["both"]["mean_r"] - study["reference"]["mean_r"]) > 1e-7):
        raise ValueError("First evaluate this shared cache's F0 and reproduce Final")
    device = device_from_name(args.device)
    seed_all(cfg["seed"], True)
    bundle = load_bundle(study["base_checkpoint"])
    initial = bundle["fm"]["ema"]
    if sha(study["base_checkpoint"]) != BASE_SHA or state_sha(initial) != study["initial_weights_sha256"]:
        raise ValueError("Initializer changed")
    model = build_model(args.group, study["model_config"], initial, device)
    del bundle, initial
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    dataset = TrainingWindows(root, study["train"], study["scale_uV"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    scales = torch.tensor(study["scale_uV"], device=device, dtype=torch.float32)
    start, updates, best_score, best_epoch, best_report = 1, 0, None, 0, None
    if args.resume:
        if Path(args.resume).resolve() != output / "last.pt" or (output / "completed.json").exists():
            raise ValueError("Resume only an incomplete run from its own last.pt")
        _, checkpoint = load_candidate(args.resume, device, study, root / "study.json")
        _, selected = load_candidate(output / "best.pt", device, study, root / "study.json")
        if (checkpoint["group"] != args.group or selected["group"] != args.group
                or checkpoint["output_dir"] != str(output) or selected["epoch"] > checkpoint["epoch"]):
            raise ValueError("Resume arm/output/best lineage mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["order_rng"].cpu())
        start, updates = checkpoint["epoch"] + 1, checkpoint["updates"]
        best_score, best_epoch, best_report = selected["best_score"], selected["best_epoch"], selected["validation"]
        restore_rng(checkpoint["rng"])
    else:
        output.mkdir(parents=True, exist_ok=False)
        write(output / "initialization.json", {"protocol": PROTOCOL, "group": args.group,
            "training": cfg, "initial_weights_sha256": study["initial_weights_sha256"],
            "source": "selected_final_fm_ema", "new_optimizer": True,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "P_optimizer_updates": 0, "body_optimizer_updates": 0,
            "selection_route": cfg["selection_route"], "epoch_zero_is_candidate": True,
            "epoch_zero_expected_to_equal_Final": args.group == "F1",
            "F1_original_FM_objective": study["fm_loss"] if args.group == "F1" else None,
            "D1_loss": study["experiment"]["direct_loss"] if args.group == "D1" else None,
            "budget_match": "central windows and optimizer updates; not FLOPs or wall time",
            "F1_I_head": "auxiliary training head; deployed I remains frozen original head"})
    rng = pack_rng()
    try:
        evaluator = Evaluator(root, study, device, args.batch_size)
    finally:
        restore_rng(rng)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def payload(epoch, report):
        return {"protocol": PROTOCOL, "code_sha256": code_sha(), "group": args.group,
            "study_sha256": sha(root / "study.json"), "base_sha256": BASE_SHA, "body_sha256": BODY_SHA,
            "model_config": study["model_config"], "scale_uV": study["scale_uV"],
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": pack_rng(),
            "order_rng": generator.get_state(), "output_dir": str(output), "epoch": epoch,
            "updates": updates, "best_score": best_score, "best_epoch": best_epoch,
            "validation": report, "selection_route": cfg["selection_route"],
            "training_seed": cfg["seed"], "torch_version": str(torch.__version__)}

    def validate(epoch):
        rng = pack_rng()
        folder = output / "validation" / f"epoch_{epoch:04d}"
        if folder.exists():
            folder = folder.with_name(folder.name + "_retry_" + str(time.time_ns()))
        try:
            return evaluator.evaluate(model, folder, args.group, epoch)
        finally:
            restore_rng(rng)

    try:
        if not args.resume:
            best_report = validate(0)
            if args.group == "F1":
                for route in ("baseline", "d6_only", "both"):
                    if abs(best_report["summary"][route]["mean_r"] - f0["summary"][route]["mean_r"]) > 1e-6:
                        raise ValueError("F1 epoch0 failed to reproduce original FM; inspect keys/scales/assembly")
            best_score = best_report["selected_score"]
            atomic_save(output / "best.pt", payload(0, best_report))
            atomic_save(output / "last.pt", payload(0, best_report))
        for epoch in range(start, cfg["epochs"] + 1):
            model.train()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begun = time.perf_counter()
            order = torch.randperm(len(dataset), generator=generator).tolist()
            totals = {}
            accumulation = cfg["gradient_accumulation"]
            for chunk in range(0, len(order), accumulation):
                indices = order[chunk:chunk + accumulation]
                optimizer.zero_grad(set_to_none=True)
                for index in indices:
                    batch = dataset[index]
                    batch["inputs"] = {k: v.to(device) for k, v in batch["inputs"].items()}
                    for key in ("target", "fast_target", "slow", "mask"):
                        batch[key] = batch[key].to(device)
                    if args.group == "D1":
                        loss, parts = direct_objective(model, batch, study["experiment"]["direct_loss"])
                    else:
                        c = batch["centre"]
                        fm_batch = {k: v[c:c + 1] for k, v in batch["inputs"].items()}
                        fm_batch.update(target=batch["target"], quality_mask=batch["mask"])
                        loss, parts = flow_loss(model, fm_batch, study["fm_loss"], scales)
                    (loss / len(indices)).backward()
                    for key, value in {"loss": loss, **parts}.items():
                        totals[key] = totals.get(key, 0.) + float(value.detach())
                torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - begun
            report, improved = None, False
            if epoch % cfg["validate_every"] == 0 or epoch == cfg["epochs"]:
                report = validate(epoch)
                improved = report["selected_score"] > best_score
                if improved:
                    best_score, best_epoch, best_report = report["selected_score"], epoch, report
            if improved:
                atomic_save(output / "best.pt", payload(epoch, report))
            atomic_save(output / "last.pt", payload(epoch, report))
            row = {"epoch": epoch, "updates": updates, "central_windows": len(order),
                "train": {k: v / len(order) for k, v in totals.items()}, "validation": report,
                "best_score": best_score, "best_epoch": best_epoch, "selected_best": improved,
                "train_seconds": elapsed, "P_optimizer_updates": 0, "body_optimizer_updates": 0,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(row, flush=True)
        write(output / "completed.json", {"protocol": PROTOCOL, "completed": True, "group": args.group,
            "epochs": cfg["epochs"], "updates": updates, "best_epoch": best_epoch,
            "best_score": best_score, "best": best_report, "selection_route": cfg["selection_route"]})
    finally:
        evaluator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--group", required=True, choices=GROUPS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference windows per batch")
    parser.add_argument("--resume")
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    if not args.execute_training:
        parser.error("No training started; add --execute-training on the server")
    if args.batch_size < 1:
        parser.error("Positive inference batch size required")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    train(args)


if __name__ == "__main__":
    main()
