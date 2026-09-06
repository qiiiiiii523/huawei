"""B0 route B: strict pretraining followed by independent L0/L1 fine-tuning.

Place this file in huawei/scripts beside train_b0_route.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ecg12gen.dataset import ECGDataConfig
from ecg12gen.b0_route_data import B0RouteDataset, load_frozen_preprocessor
from ecg12gen.b0_route_loss import B0RouteLoss
from ecg12gen.models.b0_linear import B0Linear
from ecg12gen.evaluate import evaluate_predictions
from train_b0_route import make_loader, predict_validation, save_evaluation


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", choices=["task1", "task2"], required=True)
    parser.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--weak-loss", choices=["L0", "L1"])
    parser.add_argument("--init-run", help="Completed B pretraining output directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.stage == "pretrain" and (args.weak_loss or args.init_run):
        parser.error("Pretraining does not accept --weak-loss or --init-run")
    if args.stage == "finetune" and (not args.weak_loss or not args.init_run):
        parser.error("Fine-tuning requires --weak-loss and --init-run")

    seed, batch_size = 42, 16
    epochs = 1 if args.smoke_test else 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    init_run = None
    checkpoint = None
    parent_summary = None
    scales_path = ROOT / "outputs/b0/preprocessing_scales.npz"
    if args.stage == "finetune":
        init_run = (ROOT / args.init_run).resolve()
        parent_config = read_json(init_run / "config_snapshot.json")
        parent_summary = read_json(init_run / "training_summary.json")
        for key, expected in (
            ("route", "B"), ("stage", "pretrain"), ("task_id", args.task_id)
        ):
            if parent_config.get(key) != expected:
                raise ValueError(f"Pretraining {key} must be {expected!r}")
        if parent_summary.get("completed") is not True:
            raise ValueError("Pretraining run is not marked completed")
        if not args.smoke_test and (
            parent_config.get("smoke_test") is not False
            or parent_config.get("epochs") != 100
            or parent_summary.get("epochs_completed") != 100
        ):
            raise ValueError("Full fine-tuning requires a completed 100-epoch pretrain")
        checkpoint = init_run / "best_model.pt"
        if sha256(checkpoint) != parent_summary["best_model_sha256"]:
            raise ValueError("Pretraining checkpoint was changed after completion")
        scales_path = init_run / "preprocessing_scales.npz"
        if sha256(scales_path) != parent_summary["scales_sha256"]:
            raise ValueError("Pretraining scales were changed after completion")

    config = ECGDataConfig.from_yaml(ROOT / "configs/common.yaml")
    preprocessor = load_frozen_preprocessor(config, scales_path)
    scale = torch.tensor(
        preprocessor.scale_uV_by_source["d12"], dtype=torch.float32
    )
    strict_train = B0RouteDataset(
        config, preprocessor, args.task_id, "train", "strict"
    )
    train_data = strict_train if args.stage == "pretrain" else B0RouteDataset(
        config, preprocessor, args.task_id, "train", "weak"
    )
    validation = B0RouteDataset(
        config, preprocessor, args.task_id, "validation", "weak"
    )
    train_loader = make_loader(train_data, batch_size, True, seed)
    reference_loader = make_loader(strict_train, batch_size, True, seed + 1)
    validation_loader = make_loader(validation, batch_size, False, seed + 2)

    channels = 1 if args.task_id == "task1" else 6
    model = B0Linear(channels)
    if checkpoint is not None:
        model.load_state_dict(torch.load(
            checkpoint, map_location="cpu", weights_only=True
        ), strict=True)
    # Each fine-tuning arm starts with fresh optimizer state.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = B0RouteLoss(ROOT / "configs/losses.yaml")

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "preprocessing_scales.npz", **preprocessor.scale_uV_by_source
    )
    run_config = {
        **vars(args), "route": "B", "seed": seed, "batch_size": batch_size,
        "epochs": epochs, "optimizer": "AdamW", "learning_rate": 0.001,
        "weight_decay": 0.0001, "optimizer_reset_at_finetune": True,
        "data_variant": "A", "device": "cpu", "baseline_uV": 0,
        "physiology_space": "mV", "scheduler": None, "early_stopping": False,
        "checkpoint_selection": "best_raw_original_validation_r",
        "checkpoint_tie_break": "earliest_epoch_exact_equality",
        "validation_view": "weak_raw_uV",
        "train_windows": len(train_data),
        "strict_train_windows": len(strict_train),
        "validation_windows": len(validation),
        "init_run": str(init_run) if init_run else None,
        "init_checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        "init_checkpoint_epoch": parent_summary["best_epoch"] if parent_summary else None,
    }
    write_json(output / "config_snapshot.json", run_config)
    # Preserve the actual loss weights and protocol used in this run.
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for relative in (
        "configs/common.yaml", "configs/losses.yaml", "configs/preprocessing.yaml",
        "configs/training_protocol_v1.yaml", "metadata/subject_split.csv",
        "scripts/train_b0_route.py", "scripts/train_b0_route_b.py",
        "ecg12gen/models/b0_linear.py", "ecg12gen/b0_route_data.py",
        "ecg12gen/b0_route_loss.py",
    ):
        source = ROOT / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    best_r, best_epoch = -float("inf"), None
    with (output / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for epoch in range(1, epochs + 1):
            model.train()
            reference_iterator = iter(reference_loader)
            totals, n_seen = {}, 0
            for step, batch in enumerate(train_loader):
                if args.smoke_test and step >= 2:
                    break
                if not all(s == "train" for s in batch["split"]):
                    raise ValueError("Non-training sample reached optimizer")
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["inputs"])
                kwargs = dict(
                    lead_mask=batch["lead_mask"],
                    observed_input=batch["observed_input"], d12_scale_uV=scale,
                )
                if args.stage == "pretrain":
                    if (not batch["pointwise_loss_allowed"].all()
                        or not all(a == "same_window" for a in batch["alignment_mode"])):
                        raise ValueError("Pretraining requires strictly synchronous targets")
                    loss, logs = loss_fn(
                        prediction, **kwargs, profile="strict_sync",
                        alignment_mode="same_window", pointwise_loss_allowed=True,
                        target=batch["target"],
                    )
                else:
                    if (batch["pointwise_loss_allowed"].any()
                        or not all(a == "weak_subject_pair_record_start"
                                   for a in batch["alignment_mode"])):
                        raise ValueError("Fine-tuning requires weak alignment")
                    try:
                        reference = next(reference_iterator)
                    except StopIteration:
                        reference_iterator = iter(reference_loader)
                        reference = next(reference_iterator)
                    if not all(s == "train" for s in reference["split"]):
                        raise ValueError("Reference batch must be train-only")
                    loss, logs = loss_fn(
                        prediction, **kwargs, profile=args.weak_loss,
                        alignment_mode="weak_subject_pair_record_start",
                        pointwise_loss_allowed=False,
                        target=batch["target"] if args.weak_loss == "L1" else None,
                        strict_reference=reference["target"],
                    )
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Non-finite gradient")
                optimizer.step()
                count = batch["inputs"].shape[0]
                n_seen += count
                for key, value in logs.items():
                    totals[key] = totals.get(key, 0.0) + value * count

            if not n_seen:
                raise ValueError("No training batches")
            prediction, target, _, _ = predict_validation(model, validation_loader, scale)
            metrics, _ = evaluate_predictions(prediction, target, args.task_id)
            validation_r = float(metrics["twelve_lead_mean_pearson_r"])
            if not np.isfinite(validation_r):
                raise FloatingPointError("Non-finite validation r")
            row = {
                "epoch": epoch, **{k: v / n_seen for k, v in totals.items()},
                "validation_r": validation_r,
                "validation_rmse_uV": float(metrics["twelve_lead_mean_rmse_uV"]),
            }
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if validation_r > best_r:
                best_r, best_epoch = validation_r, epoch
                torch.save(model.state_dict(), output / "best_model.pt")
            print(
                f"B {args.stage} | Epoch {epoch}/{epochs} | loss={row['total']:.6f} | "
                f"val_r={validation_r:.6f} | best_r={best_r:.6f}", flush=True,
            )

    model.load_state_dict(torch.load(
        output / "best_model.pt", map_location="cpu", weights_only=True
    ))
    prediction, target, raw_input, metadata = predict_validation(model, validation_loader, scale)
    original = save_evaluation(
        output / "evaluation/original", prediction, target, args.task_id, metadata
    )
    copied = prediction.copy()
    copied[:, :channels] = raw_input
    if not np.array_equal(prediction[:, channels:], copied[:, channels:]):
        raise RuntimeError("Copy-at-eval changed missing leads")
    copy_metrics = save_evaluation(
        output / "evaluation/copy_at_eval", copied, target, args.task_id, metadata
    )
    write_json(output / "training_summary.json", {
        "completed": True, "route": "B", "stage": args.stage,
        "smoke_test": args.smoke_test, "epochs_completed": epochs,
        "best_epoch": best_epoch, "original": original, "copy_at_eval": copy_metrics,
        "best_model_sha256": sha256(output / "best_model.pt"),
        "scales_sha256": sha256(output / "preprocessing_scales.npz"),
        "init_checkpoint_sha256": run_config["init_checkpoint_sha256"],
    })
    print(f"Completed. Results: {output}")


if __name__ == "__main__":
    main()
