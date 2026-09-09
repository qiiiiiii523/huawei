"""B0 Task2 P0: shared strict machine-I -> D12 pretraining.

Place in scripts/.  P0 does not consume D6 context, but reserves the same
six-channel architecture required by both Task2 P1-C3 A/B experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.evaluate import (
    evaluate_centered_diagnostic,
    evaluate_joint_anchor_predictions,
    evaluate_predictions,
    evaluate_task2_diagnostics,
    write_report,
    write_task2_diagnostics,
)
from ecg12gen.losses import strict_anchor_pretrain_loss
from ecg12gen.models.b0_joint_anchor import B0JointAnchor
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, values):
    Path(path).write_text(
        json.dumps(values, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )


def tensor(value):
    return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))


class StrictTorchDataset(Dataset):
    def __init__(self, source, preprocessor, subject_split):
        self.source = source
        self.preprocessor = preprocessor
        if any(subject_split.get(str(row["subject_id"])) != "train" for row in source.rows):
            raise ValueError("Strict index includes non-training subjects")

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        if sample.split != "train" or not np.array_equal(sample.X_ecg, sample.Y_12lead[:1]):
            raise ValueError("Strict training requires train-only same-window anchor")
        anchor = self.preprocessor.transform_window(sample.X_ecg, "ecg_machine_i").model_signal
        target = self.preprocessor.transform_d12_target(sample.Y_12lead).model_signal
        return tensor(anchor), tensor(target)


def ridge_weights(dataset, alpha):
    xx = 0.0
    xy = np.zeros(12, dtype=np.float64)
    for index in range(len(dataset)):
        anchor, target = dataset[index]
        x = anchor.numpy()[0].astype(np.float64)
        xx += np.dot(x, x)
        xy += target.numpy().astype(np.float64) @ x
    if xx <= 0 or not np.isfinite(xx) or not np.isfinite(xy).all():
        raise ValueError("Invalid train-only Ridge sufficient statistics")
    return (xy / (xx + alpha)).astype(np.float32)


def validation_arrays(source, preprocessor, subject_split):
    anchors, targets, raw_anchors, metadata = [], [], [], []
    for sample in source:
        sid = str(sample.subject_id)
        if sample.split != "validation" or subject_split.get(sid) != "validation":
            raise ValueError("Invalid Task2 validation split")
        if not np.array_equal(sample.anchor_i_ecg, sample.Y_12lead[:1]):
            raise ValueError("Invalid validation anchor identity")
        anchors.append(
            preprocessor.transform_window(sample.anchor_i_ecg, "ecg_machine_i").model_signal
        )
        targets.append(sample.Y_12lead.copy())
        raw_anchors.append(sample.anchor_i_ecg.copy())
        metadata.append({
            "subject_id": sid, "window_id": sample.window_id,
            "target_record_id": sample.target_record_id, "input_type": sample.input_type,
        })
    if not anchors:
        raise ValueError("No Task2 validation samples")
    return tensor(np.stack(anchors)), np.stack(targets), np.stack(raw_anchors), metadata


@torch.no_grad()
def predict(model, anchors, scale):
    model.eval()
    predictions = []
    for batch in anchors.split(16):
        predictions.append((model(None, batch) * scale[None, :, None]).cpu().numpy())
    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", default="outputs/b0_joint_anchor/task2/preprocessing")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or not np.isfinite(args.ridge_alpha) or args.ridge_alpha <= 0:
        parser.error("epochs and ridge-alpha must be positive")

    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    prep = (ROOT / args.preparation_dir).resolve()
    summary_path = prep / "preparation_summary.json"
    prepared = json.loads(summary_path.read_text(encoding="utf-8"))
    scale_a_path = prep / "preprocessing_scales_A.npz"
    scale_b_path = prep / "preprocessing_scales_B.npz"
    if (
        prepared.get("completed") is not True
        or prepared.get("task_id") != "task2"
        or prepared.get("fit_split") != "train"
        or prepared.get("validation_used_to_fit_scales") is not False
        or not prepared.get("variants_are_separate_experiments")
        or sha(scale_a_path) != prepared["scales"]["A_raw_window"]["sha256"]
        or sha(scale_b_path) != prepared["scales"]["B_detrend_0p2Hz_then_window"]["sha256"]
    ):
        raise ValueError("Invalid or modified Task2 preparation")

    with np.load(scale_a_path, allow_pickle=False) as handle:
        scales_a = {key: handle[key].copy() for key in handle.files}
    with np.load(scale_b_path, allow_pickle=False) as handle:
        scales_b = {key: handle[key].copy() for key in handle.files}
    for key in ("d12", "ecg_machine_i", "ecg_machine_d6"):
        if not np.array_equal(scales_a[key], scales_b[key]):
            raise ValueError(f"Shared A/B scale differs: {key}")
    if not np.array_equal(scales_a["ecg_machine_i"], scales_a["d12"][:1]):
        raise ValueError("Anchor scale differs from D12 I")

    cfg = ECGDataConfig.from_yaml(ROOT / "configs/common.yaml")
    for name, path in {
        "common.yaml": ROOT / "configs/common.yaml",
        "preprocessing.yaml": cfg.path("preprocessing_config"),
        "subject_split.csv": cfg.path("subject_split_csv"),
        "d12_strict_pretrain_index.csv": ROOT / "metadata/d12_strict_pretrain_index.csv",
        "pair_manifest_task2.csv": cfg.path("task2_pair_manifest_csv"),
        "task2_window_metadata.csv": cfg.path("task2_output") / "task2_window_metadata.csv",
        "body_scale_b_metadata.csv": cfg.path("task2_body_scale_b_metadata"),
    }.items():
        if sha(path) != prepared["source_sha256"][name]:
            raise ValueError(f"Data/preprocessing contract changed since preparation: {name}")

    preprocessor = ECGPreprocessor(
        PreprocessingConfig.from_yaml(cfg.path("preprocessing_config")), scales_a
    )
    with cfg.path("subject_split_csv").open(encoding="utf-8-sig", newline="") as handle:
        subject_split = {row["subject_id"]: row["split"] for row in csv.DictReader(handle)}
    strict_source = StrictD12PretrainDataset(cfg, "d12_i_pretrain")
    train = StrictTorchDataset(strict_source, preprocessor, subject_split)
    # A/B targets and row identities were proved equal during preparation; P0 consumes neither context.
    validation_source = JointAnchorDataset(cfg, "task2", "validation", body_scale_variant="A_raw_window")
    anchors, target, raw_anchor, metadata = validation_arrays(
        validation_source, preprocessor, subject_split
    )
    train_subjects = {str(row["subject_id"]) for row in strict_source.rows}
    if train_subjects & {row["subject_id"] for row in metadata}:
        raise ValueError("Training/validation subject overlap")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    protocol = yaml.safe_load(
        (ROOT / "configs/context_fusion_protocol.yaml").read_text(encoding="utf-8")
    )
    gate_bias = float(protocol["fusion_definitions"]["initialization"]["gate_logit_bias"])
    model = B0JointAnchor(context_channels=6, gate_logit_bias=gate_bias)
    print(f"Fitting Ridge initialization on {len(train)} strict training windows...", flush=True)
    weights = ridge_weights(train, args.ridge_alpha)
    with torch.no_grad():
        model.mapping.weight.copy_(torch.from_numpy(weights[:, None, None]))

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.001, weight_decay=0.0001,
    )
    loss_cfg = yaml.safe_load(
        (ROOT / "configs/losses.yaml").read_text(encoding="utf-8")
    )["strict_anchor_pretrain"]
    loss_weights = {
        "huber_weight": float(loss_cfg["huber_full_d12"]),
        "pcc_weight": float(loss_cfg["pcc_full_d12"]),
        "physiology_weight": float(loss_cfg["physiology"]),
        "observed_weight": float(loss_cfg["observed_anchor_consistency"]),
    }
    scale = tensor(scales_a["d12"])
    loader = DataLoader(
        train, batch_size=16, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    epochs = 1 if args.smoke_test else args.epochs

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scale_a_path, output / "preprocessing_scales_A.npz")
    shutil.copy2(scale_b_path, output / "preprocessing_scales_B.npz")
    write_json(output / "config_snapshot.json", {
        **vars(args), "stage": "P0_anchor_only", "task_id": "task2", "seed": seed,
        "epochs": epochs, "batch_size": 16, "optimizer": "AdamW",
        "learning_rate": 0.001, "weight_decay": 0.0001,
        "scheduler": None, "early_stopping": False, "baseline_uV": 0,
        "initialization": "strict_train_closed_form_ridge",
        "ridge_objective": "sum_squared_error_plus_alpha_times_squared_weights",
        "loss": "strict_anchor_pretrain_loss", "loss_weights": loss_weights,
        "context_enabled": False, "shared_for_task2_variants": ["A", "B"],
        "architecture_id": model.architecture_id,
        "architecture_config": model.architecture_config,
        "architecture_config_hash": model.architecture_config_hash,
        "checkpoint_metric": "r_raw_12_official_raw_uV_v0",
        "train_windows": len(train), "validation_windows": len(validation_source),
        "d12_scale_sha256": array_sha(scales_a["d12"]),
        "preparation_summary_sha256": sha(summary_path),
    })
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for relative in [
        "scripts/train_b0_joint_task2_p0.py", "ecg12gen/models/b0_joint_anchor.py",
        "ecg12gen/losses.py", "ecg12gen/evaluate.py", "ecg12gen/preprocessing.py",
        "ecg12gen/dataset.py", "ecg12gen/d12_pretrain.py", "configs/losses.yaml",
        "configs/context_fusion_protocol.yaml", "configs/training_protocol_v1.yaml",
    ]:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(summary_path, output / "preparation_summary.json")

    best_r, best_epoch = -float("inf"), None
    with (output / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for epoch in range(1, epochs + 1):
            model.train()
            total, count = 0.0, 0
            for step, (anchor, y) in enumerate(loader):
                if args.smoke_test and step >= 2:
                    break
                optimizer.zero_grad(set_to_none=True)
                prediction = model(None, anchor)
                loss = strict_anchor_pretrain_loss(
                    prediction, y, anchor, d12_scale_uV=scale, **loss_weights
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Non-finite gradient")
                optimizer.step()
                total += float(loss.detach()) * len(anchor)
                count += len(anchor)

            raw = predict(model, anchors, scale)
            metrics, _, _, _ = evaluate_joint_anchor_predictions(raw, target, raw_anchor, "task2")
            checkpoint_r = float(metrics["r_raw_12"])
            if not np.isfinite(checkpoint_r):
                raise FloatingPointError("Non-finite validation r")
            row = {
                "epoch": epoch, "loss": total / count,
                **{key: metrics[key] for key in ["r_raw_12", "r_submit_12", "r_missing11"]},
                "task2_missing11_rmse_uV": metrics["task2_missing_lead_mean_rmse_uV"],
                "submit_rmse_uV": metrics["twelve_lead_mean_rmse_uV"],
            }
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if checkpoint_r > best_r:
                best_r, best_epoch = checkpoint_r, epoch
                torch.save({
                    "state_dict": model.state_dict(), "stage": "P0_anchor_only",
                    "task_id": "task2", "architecture_id": model.architecture_id,
                    "architecture_config": model.architecture_config,
                    "architecture_config_hash": model.architecture_config_hash,
                    "d12_scale_sha256": array_sha(scales_a["d12"]),
                    "preparation_summary_sha256": sha(summary_path),
                    "epoch": epoch, "smoke_test": args.smoke_test,
                }, output / "best_model.pt")
            print(
                f'P0 Task2 | Epoch {epoch}/{epochs} | loss={row["loss"]:.6f} | '
                f'r_raw={row["r_raw_12"]:.6f} | r_submit={row["r_submit_12"]:.6f} | '
                f'r_missing11={row["r_missing11"]:.6f}', flush=True,
            )

    saved = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    raw = predict(model, anchors, scale)
    metrics, raw_details, submit_details, submit = evaluate_joint_anchor_predictions(
        raw, target, raw_anchor, "task2"
    )
    _, _, report_path = write_report(
        output / "evaluation", metrics, submit_details,
        title="B0 Task2 shared P0 submit-view validation",
    )
    subject_rows, device_rows = evaluate_task2_diagnostics(submit, target, metadata)
    write_task2_diagnostics(output / "evaluation", subject_rows, device_rows, report_path)
    raw_metrics, _ = evaluate_predictions(raw, target, "task2")
    write_report(output / "evaluation/raw_prediction", raw_metrics, raw_details)
    centered, centered_details = evaluate_centered_diagnostic(submit, target, "task2")
    _, _, centered_report = write_report(
        output / "evaluation/centered_diagnostic", centered, centered_details,
        title="Centered diagnostic - not official",
    )
    centered_subjects, centered_devices = evaluate_task2_diagnostics(
        submit - np.median(submit, axis=2, keepdims=True),
        target - np.median(target, axis=2, keepdims=True), metadata,
    )
    write_task2_diagnostics(
        output / "evaluation/centered_diagnostic", centered_subjects,
        centered_devices, centered_report,
    )
    np.save(output / "validation_prediction_raw.npy", raw)
    np.save(output / "validation_prediction_submit.npy", submit)
    np.save(output / "validation_anchor_i.npy", raw_anchor)
    with (output / "validation_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    write_json(output / "training_summary.json", {
        "completed": True, "stage": "P0_anchor_only", "task_id": "task2",
        "smoke_test": args.smoke_test, "epochs_completed": epochs,
        "best_epoch": best_epoch, "validation": metrics,
        "raw_prediction": raw_metrics, "device_diagnostics": device_rows,
        "architecture_id": model.architecture_id,
        "architecture_config_hash": model.architecture_config_hash,
        "best_model_sha256": sha(output / "best_model.pt"),
        "d12_scale_sha256": array_sha(scales_a["d12"]),
        "preparation_summary_sha256": sha(summary_path),
    })
    print(f"Completed. Results: {output}")


if __name__ == "__main__":
    main()
