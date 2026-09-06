from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg12gen.dataset import ECGDataConfig
from ecg12gen.b0_route_data import (
    B0RouteDataset,
    load_frozen_preprocessor,
)
from ecg12gen.b0_route_loss import B0RouteLoss
from ecg12gen.models.b0_linear import B0Linear
from ecg12gen.evaluate import (
    evaluate_predictions,
    evaluate_centered_diagnostic,
    write_report,
    evaluate_task2_diagnostics,
    write_task2_diagnostics,
)


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


@torch.no_grad()
def predict_validation(model, loader, scale):
    model.eval()
    predictions, targets, inputs, metadata = [], [], [], []

    for batch in loader:
        prediction = model(batch["inputs"])
        prediction = prediction * scale[None, :, None]

        predictions.append(prediction.numpy())
        targets.append(batch["raw_target_uV"].numpy())
        inputs.append(batch["raw_input_uV"].numpy())

        metadata.extend(
            {
                "subject_id": subject,
                "input_type": device,
            }
            for subject, device in zip(
                batch["subject_id"],
                batch["device_type"],
            )
        )

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(inputs),
        metadata,
    )


def save_evaluation(output, prediction, target, task, metadata):
    overall, details = evaluate_predictions(
        prediction, target, task
    )
    paths = write_report(
        output, overall, details,
        title=f"B0 {task} validation: {output.name}",
    )

    centered_overall, centered_details = (
        evaluate_centered_diagnostic(prediction, target, task)
    )
    centered_dir = output / "centered_diagnostic"
    centered_paths = write_report(
        centered_dir,
        centered_overall,
        centered_details,
        title="Centered morphology diagnostic - not official",
    )

    if task == "task2":
        subject_rows, device_rows = evaluate_task2_diagnostics(
            prediction, target, metadata
        )
        write_task2_diagnostics(
            output, subject_rows, device_rows, paths[-1]
        )

        centered_prediction = prediction - np.median(
            prediction, axis=2, keepdims=True
        )
        centered_target = target - np.median(
            target, axis=2, keepdims=True
        )
        subject_rows, device_rows = evaluate_task2_diagnostics(
            centered_prediction, centered_target, metadata
        )
        write_task2_diagnostics(
            centered_dir,
            subject_rows,
            device_rows,
            centered_paths[-1],
        )

    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id", choices=["task1", "task2"], required=True
    )
    parser.add_argument("--route", choices=["A"], default="A")
    parser.add_argument(
        "--weak-loss", choices=["L0", "L1"], required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    # 本版仅支持原始数据版本A。
    seed = 42
    batch_size = 16
    epochs = 1 if args.smoke_test else 100

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    config = ECGDataConfig.from_yaml(
        ROOT / "configs/common.yaml"
    )
    scales_path = ROOT / "outputs/b0/preprocessing_scales.npz"
    preprocessor = load_frozen_preprocessor(config, scales_path)

    scale = torch.tensor(
        preprocessor.scale_uV_by_source["d12"],
        dtype=torch.float32,
    )

    # 保存本次使用的冻结尺度，便于独立复现。
    np.savez_compressed(
        output / "preprocessing_scales.npz",
        **preprocessor.scale_uV_by_source,
    )

    weak_train = B0RouteDataset(
        config, preprocessor, args.task_id, "train", "weak"
    )
    strict_train = B0RouteDataset(
        config, preprocessor, args.task_id, "train", "strict"
    )
    validation = B0RouteDataset(
        config, preprocessor, args.task_id, "validation", "weak"
    )

    # 两个独立采样器；L0/L1使用相同种子。
    train_loader = make_loader(
        weak_train, batch_size, True, seed
    )
    reference_loader = make_loader(
        strict_train, batch_size, True, seed + 1
    )
    validation_loader = make_loader(
        validation, batch_size, False, seed + 2
    )

    channels = 1 if args.task_id == "task1" else 6
    model = B0Linear(channels)
    loss_fn = B0RouteLoss(ROOT / "configs/losses.yaml")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    run_config = {
        **vars(args),
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "data_variant": "A",
        "device": "cpu",
        "baseline_uV": 0,
        "physiology_space": "mV",
        "checkpoint_selection": "best_raw_original_validation_r",
        "weak_train_windows": len(weak_train),
        "strict_reference_windows": len(strict_train),
        "validation_windows": len(validation),
    }
    (output / "config_snapshot.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_r = -float("inf")
    best_epoch = None

    with (output / "training_log.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = None

        for epoch in range(1, epochs + 1):
            model.train()
            reference_iterator = iter(reference_loader)
            totals = {}
            n_seen = 0

            for step, batch in enumerate(train_loader):
                if args.smoke_test and step >= 2:
                    break

                assert all(s == "train" for s in batch["split"])
                assert not batch["pointwise_loss_allowed"].any()
                assert all(
                    a == "weak_subject_pair_record_start"
                    for a in batch["alignment_mode"]
                )

                try:
                    reference = next(reference_iterator)
                except StopIteration:
                    reference_iterator = iter(reference_loader)
                    reference = next(reference_iterator)

                assert all(s == "train" for s in reference["split"])

                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["inputs"])

                loss, logs = loss_fn(
                    prediction,
                    profile=args.weak_loss,
                    lead_mask=batch["lead_mask"],
                    observed_input=batch["observed_input"],
                    d12_scale_uV=scale,
                    alignment_mode="weak_subject_pair_record_start",
                    pointwise_loss_allowed=False,
                    target=(
                        batch["target"]
                        if args.weak_loss == "L1"
                        else None
                    ),
                    strict_reference=reference["target"],
                )

                loss.backward()

                for parameter in model.parameters():
                    if (
                        parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                    ):
                        raise FloatingPointError("Non-finite gradient")

                optimizer.step()

                count = batch["inputs"].shape[0]
                n_seen += count
                for key, value in logs.items():
                    totals[key] = totals.get(key, 0.0) + value * count

            prediction, target, _, _ = predict_validation(
                model, validation_loader, scale
            )
            metrics, _ = evaluate_predictions(
                prediction, target, args.task_id
            )
            validation_r = float(
                metrics["twelve_lead_mean_pearson_r"]
            )
            if not np.isfinite(validation_r):
                raise FloatingPointError("Non-finite validation r")

            row = {
                "epoch": epoch,
                **{
                    key: value / n_seen
                    for key, value in totals.items()
                },
                "validation_r": validation_r,
                "validation_rmse_uV": float(
                    metrics["twelve_lead_mean_rmse_uV"]
                ),
            }

            if writer is None:
                writer = csv.DictWriter(
                    handle, fieldnames=list(row)
                )
                writer.writeheader()
            writer.writerow(row)
            handle.flush()

            # 相同指标保留更早epoch。
            if validation_r > best_r:
                best_r = validation_r
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    output / "best_model.pt",
                )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"loss={row['total']:.6f} | "
                f"val_r={validation_r:.6f} | "
                f"best_r={best_r:.6f}",
                flush=True,
            )

    # 同一个best checkpoint用于两种输出评价。
    model.load_state_dict(torch.load(
        output / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    ))
    prediction, target, raw_input, metadata = predict_validation(
        model, validation_loader, scale
    )

    original_metrics = save_evaluation(
        output / "evaluation/original",
        prediction, target, args.task_id, metadata,
    )

    prediction_copy = prediction.copy()
    prediction_copy[:, :channels] = raw_input

    copy_metrics = save_evaluation(
        output / "evaluation/copy_at_eval",
        prediction_copy, target, args.task_id, metadata,
    )

    # 明确：复制可见导联不能改变缺失导联预测。
    assert np.array_equal(
        prediction[:, channels:],
        prediction_copy[:, channels:],
    )

    summary = {
        "smoke_test": args.smoke_test,
        "best_epoch": best_epoch,
        "original": original_metrics,
        "copy_at_eval": copy_metrics,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Completed. Results: {output}")


if __name__ == "__main__":
    main()