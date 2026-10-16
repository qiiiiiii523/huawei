"""Paired continuation: original FM loss, frozen S50-F, final hybrid-r selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.losses import flow_loss
from baselines.B5.runtime import collate, seed_all, seed_worker, to_device, device_from_name
from baselines.B5_best import SCORE_PROTOCOL
from baselines.B5_best.common import write, sha
from . import PROTOCOL, ARCHITECTURE_ID
from .common import load_study, datasets_for, load_experiment_checkpoint
from .evaluation import ExperimentalPredictor, evaluate
from .model import initialized_model, KERNELS


def append_metrics(output, epoch, score):
    row = {"epoch": epoch, **score}
    path = output / "validation_metrics.csv"
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run(study_path, group, output, device_name, resume=None):
    study_path, output = Path(study_path).resolve(), Path(output).resolve()
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory; keep tee logs beside it")
    if resume and not output.is_dir():
        raise FileNotFoundError("Resume requires the original output directory")
    device = device_from_name(device_name)
    study, bundle, preprocessor = load_study(study_path)
    train, validation = datasets_for(study, bundle, preprocessor)
    config, training = study["data_config"], study["data_config"]["training"]
    seed_all(int(training["seed"]), bool(training["deterministic"]))
    model = initialized_model(bundle, group, device)
    # Each model initializes old keys from the SAME original EMA; both optimize all FM weights.
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                 weight_decay=float(training["weight_decay"]))
    amp = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    generator = torch.Generator().manual_seed(int(training["seed"]))
    loader = DataLoader(train, batch_size=int(training["batch_size"]), shuffle=True,
                        generator=generator, num_workers=int(training["workers"]),
                        collate_fn=collate, worker_init_fn=seed_worker,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    evaluation_model = initialized_model(bundle, group, device).eval()
    engine = ExperimentalPredictor(bundle, device, evaluation_model)
    scales = torch.as_tensor(bundle["fm"]["scales"]["d12"], device=device)
    start_epoch, updates, best_r, best_epoch = 1, 0, None, None
    if resume:
        checkpoint, _, _, _, _ = load_experiment_checkpoint(resume, study_path, device,
                                                            verify_data=False)
        if checkpoint["group"] != group or checkpoint["output_dir"] != str(output):
            raise ValueError("Resume must preserve group and output directory")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["grad_scaler"])
        ema = {k: v.to(device).clone() for k, v in checkpoint["ema"].items()}
        start_epoch, updates = checkpoint["epoch"] + 1, checkpoint["updates"]
        best_r, best_epoch = checkpoint["best_r"], checkpoint["best_epoch"]
        generator.set_state(checkpoint["loader_rng"].cpu())
        restore_rng(checkpoint["rng"])
        # An interruption may occur after best.pt was saved but before last.pt.
        # Preserve that better candidate when replaying the incomplete epoch.
        saved_best = output / "best.pt"
        if saved_best.exists():
            previous = torch.load(saved_best, map_location="cpu", weights_only=True)
            if previous.get("study_sha256") != sha(study_path) or previous.get("group") != group:
                raise ValueError("Existing best.pt belongs to another study/group")
            if previous["best_r"] > best_r:
                best_r, best_epoch = previous["best_r"], previous["best_epoch"]
    else:
        output.mkdir(parents=True, exist_ok=True)
        write(output / "initialization.json", {"group": group, "study_sha256": sha(study_path),
              "source_provenance": bundle["provenance"], "training": training,
              "loss": config["loss"], "sampling": study["sampling"],
              "beta": 1.0, "all_FM_parameters_trainable": True, "S50_F_frozen": True,
              "selection": "final_hybrid_main_correlation", "epoch_zero_is_candidate": True})
        # Discard random-number consumption from constructing the extra branch/eval models.
        # Dropout/noise/t and shuffled training order therefore start identically in both arms.
        seed_all(int(training["seed"]), bool(training["deterministic"]))

    def payload(epoch):
        return {"protocol": PROTOCOL, "architecture_id": ARCHITECTURE_ID if group == "v13" else "B5-U-v1",
                "score_protocol": SCORE_PROTOCOL, "study_sha256": sha(study_path),
                "study_path": str(study_path), "output_dir": str(output), "group": group,
                "epoch": epoch, "updates": updates, "best_r": best_r, "best_epoch": best_epoch,
                "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
                "grad_scaler": scaler.state_dict(), "rng": pack_rng(),
                "loader_rng": generator.get_state(), "sampling_config": study["sampling"],
                "loss_config": config["loss"], "training_config": training, "kernels": list(KERNELS),
                "torch_version": str(torch.__version__)}

    def validation_at(epoch):
        print(json.dumps({"phase": "hybrid_validation", "epoch": epoch,
                          "sampling": study["sampling"], "group": group}), flush=True)
        rng = pack_rng()
        try:
            evaluation_model.load_state_dict(ema, strict=True)
            folder = output / "validation" / f"epoch_{epoch:04d}"
            if folder.exists():
                import time
                folder = folder.with_name(folder.name + "_attempt_" + str(time.time_ns()))
            return evaluate(evaluation_model, bundle, validation, device, folder, group, epoch,
                            engine=engine, verify_initial=(epoch == 0))
        finally:
            restore_rng(rng)

    if not resume:
        score = validation_at(0)
        best_r, best_epoch = score["main_correlation"], 0
        append_metrics(output, 0, score)
        checkpoint = payload(0)
        atomic_save(output / "best.pt", checkpoint)
        atomic_save(output / "last.pt", checkpoint)
        print(json.dumps({"epoch": 0, "updates": 0, "initial_hybrid_r": best_r}), flush=True)
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = {}, 0
        for index, batch in enumerate(loader):
            inputs = to_device(batch, device)
            accumulation = int(training["gradient_accumulation"])
            divisor = min(accumulation, len(loader) - (index // accumulation) * accumulation)
            with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                loss, parts = flow_loss(model, inputs, config["loss"], scales)
            scaler.scale(loss / divisor).backward()
            n = len(inputs["anchor"])
            for key, value in {"loss": loss, **parts}.items():
                total[key] = total.get(key, 0.) + float(value.detach()) * n
            count += n
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]),
                                               error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema[k].mul_(float(training["ema_decay"])).add_(
                            v, alpha=1 - float(training["ema_decay"]))
        record = {"epoch": epoch, "updates": updates, "train": {k: v / count for k, v in total.items()}}
        improved = False
        if epoch % int(training["validate_every"]) == 0 or epoch == int(training["epochs"]):
            score = validation_at(epoch)
            append_metrics(output, epoch, score)
            improved = score["main_correlation"] > best_r
            if improved:
                best_r, best_epoch = score["main_correlation"], epoch
            record["validation"] = score
        record.update(best_r=best_r, best_epoch=best_epoch, selected_best=improved)
        checkpoint = payload(epoch)
        if improved:
            atomic_save(output / "best.pt", checkpoint)
        atomic_save(output / "last.pt", checkpoint)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
    write(output / "completed.json", {"completed": True, "group": group, "epochs": training["epochs"],
          "best_r": best_r, "best_epoch": best_epoch, "study_sha256": sha(study_path)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--group", choices=("control", "v13"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume")
    p.add_argument("--execute-training", action="store_true")
    args = p.parse_args()
    if not args.execute_training:
        p.error("Training not started: pass --execute-training on the training server")
    run(args.study, args.group, args.output, args.device, args.resume)


if __name__ == "__main__":
    main()
