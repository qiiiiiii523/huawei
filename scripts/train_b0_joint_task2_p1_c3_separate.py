"""B0 Task2 P1-C3 for context variant A or B.

Place in scripts/.  Both experiments load the same six-channel Task2 P0
checkpoint.  Context is same-subject/cross-time conditioning only.
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

from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.evaluate import (
    evaluate_centered_diagnostic,
    evaluate_joint_anchor_predictions,
    evaluate_predictions,
    evaluate_task2_diagnostics,
    write_report,
    write_task2_diagnostics,
)
from ecg12gen.losses import joint_anchor_sync_loss
from ecg12gen.models.b0_joint_anchor import B0JointAnchor
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


VARIANT_NAMES = {
    "A": "A_raw_window",
    "B": "B_detrend_0p2Hz_then_window",
}


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


class JointTorchDataset(Dataset):
    def __init__(self, source, preprocessor, expected_split, subject_split, input_type):
        self.source = source
        self.preprocessor = preprocessor
        self.expected_split = expected_split
        self.subject_split = subject_split
        self.input_type = input_type
        self.indices = [
            index for index in range(len(source))
            if source[index].input_type == input_type
        ]
        if not self.indices:
            raise ValueError(f"No {expected_split} samples for {input_type}")

    def __len__(self):
        return len(self.indices)

    def raw_sample(self, index):
        return self.source[self.indices[index]]

    def __getitem__(self, index):
        sample = self.raw_sample(index)
        sid = str(sample.subject_id)
        if sample.split != self.expected_split or self.subject_split.get(sid) != self.expected_split:
            raise ValueError("Task2 joint subject/split mismatch")
        if sample.context_source_type not in {"ecg_machine_d6", "body_scale_d6"}:
            raise ValueError("Unexpected Task2 context source")
        if sample.anchor_source_type != "ecg_machine_i":
            raise ValueError("Unexpected Task2 anchor source")
        if not np.array_equal(sample.anchor_i_ecg, sample.Y_12lead[:1]):
            raise ValueError("Anchor and target I must be same-record/same-window identical")
        context_mask = np.asarray(sample.context_lead_mask, dtype=bool)
        anchor_mask = np.asarray(sample.anchor_lead_mask, dtype=bool)
        if context_mask.shape != (6,) or not context_mask.all():
            raise ValueError("Task2 B0 requires six present context leads")
        if anchor_mask.shape != (12,) or not anchor_mask[0] or anchor_mask[1:].any():
            raise ValueError("Target-time observed mask must be I-only")
        context = self.preprocessor.transform_window(
            sample.context_ecg, sample.context_source_type
        ).model_signal
        anchor = self.preprocessor.transform_window(
            sample.anchor_i_ecg, "ecg_machine_i"
        ).model_signal
        target = self.preprocessor.transform_d12_target(sample.Y_12lead).model_signal
        return (
            tensor(context), tensor(anchor), tensor(target),
            torch.from_numpy(context_mask.copy()), torch.from_numpy(anchor_mask.copy()),
        )


def validation_arrays(dataset):
    contexts, anchors, targets, raw_anchors = [], [], [], []
    context_masks, anchor_masks, metadata = [], [], []
    for index in range(len(dataset)):
        sample = dataset.raw_sample(index)
        context, anchor, _, context_mask, anchor_mask = dataset[index]
        contexts.append(context.numpy())
        anchors.append(anchor.numpy())
        targets.append(sample.Y_12lead.copy())
        raw_anchors.append(sample.anchor_i_ecg.copy())
        context_masks.append(context_mask.numpy())
        anchor_masks.append(anchor_mask.numpy())
        metadata.append({
            "subject_id": str(sample.subject_id), "window_id": sample.window_id,
            "target_record_id": sample.target_record_id, "input_type": sample.input_type,
            "input_processing_variant": sample.meta.get("input_processing_variant", "not_applicable"),
        })
    if not contexts:
        raise ValueError("No Task2 validation samples")
    return (
        tensor(np.stack(contexts)), tensor(np.stack(anchors)), np.stack(targets),
        np.stack(raw_anchors), torch.from_numpy(np.stack(context_masks)),
        torch.from_numpy(np.stack(anchor_masks)), metadata,
    )


@torch.no_grad()
def predict(model, contexts, anchors, context_masks, anchor_masks, scale):
    model.eval()
    predictions = []
    for start in range(0, len(anchors), 16):
        stop = start + 16
        normalized = model(
            contexts[start:stop], anchors[start:stop],
            context_masks[start:stop], anchor_masks[start:stop],
        )
        predictions.append((normalized * scale[None, :, None]).cpu().numpy())
    return np.concatenate(predictions)


def device_stratified_permutation(metadata):
    """Deterministically shuffle contexts within device, never across devices."""
    permutation = np.arange(len(metadata), dtype=np.int64)
    for input_type in sorted({row["input_type"] for row in metadata}):
        indices = np.asarray(
            [index for index, row in enumerate(metadata) if row["input_type"] == input_type],
            dtype=np.int64,
        )
        if len(indices) < 2:
            raise ValueError(f"Cannot shuffle singleton device stratum: {input_type}")
        permutation[indices] = np.roll(indices, 1)
    if np.any(permutation == np.arange(len(metadata))):
        raise ValueError("Shuffled-context permutation contains a fixed point")
    return torch.from_numpy(permutation)


def write_full_evaluation(output, model, contexts, anchors, context_masks, anchor_masks,
                          scale, target, raw_anchor, metadata, title):
    raw = predict(model, contexts, anchors, context_masks, anchor_masks, scale)
    metrics, raw_details, submit_details, submit = evaluate_joint_anchor_predictions(
        raw, target, raw_anchor, "task2"
    )
    _, _, report_path = write_report(output, metrics, submit_details, title=title)
    subject_rows, device_rows = evaluate_task2_diagnostics(submit, target, metadata)
    write_task2_diagnostics(output, subject_rows, device_rows, report_path)
    raw_metrics, _ = evaluate_predictions(raw, target, "task2")
    write_report(output / "raw_prediction", raw_metrics, raw_details)
    centered, centered_details = evaluate_centered_diagnostic(submit, target, "task2")
    _, _, centered_report = write_report(
        output / "centered_diagnostic", centered, centered_details,
        title="Centered diagnostic - not official",
    )
    centered_submit = submit - np.median(submit, axis=2, keepdims=True)
    centered_target = target - np.median(target, axis=2, keepdims=True)
    centered_subjects, centered_devices = evaluate_task2_diagnostics(
        centered_submit, centered_target, metadata
    )
    write_task2_diagnostics(
        output / "centered_diagnostic", centered_subjects, centered_devices, centered_report
    )
    return raw, submit, metrics, raw_metrics, device_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", default="outputs/b0_joint_anchor/task2/preprocessing")
    parser.add_argument("--p0-checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=("A", "B"))
    parser.add_argument(
        "--input-type", required=True,
        choices=("ecg_machine_d6", "body_scale_d6"),
        help="Train and validate one Task2 context source only",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")

    variant_name = VARIANT_NAMES[args.variant]
    if args.input_type == "ecg_machine_d6" and args.variant != "A":
        parser.error("ecg_machine_d6 is unchanged between variants; run it once with --variant A")
    output = (ROOT / args.output_dir).resolve()
    prep = (ROOT / args.preparation_dir).resolve()
    checkpoint_path = (ROOT / args.p0_checkpoint).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"P0 checkpoint not found: {checkpoint_path}")

    summary_path = prep / "preparation_summary.json"
    prepared = json.loads(summary_path.read_text(encoding="utf-8"))
    scale_path = prep / f"preprocessing_scales_{args.variant}.npz"
    if (
        prepared.get("completed") is not True
        or prepared.get("task_id") != "task2"
        or prepared.get("validation_used_to_fit_scales") is not False
        or prepared.get("variants_are_separate_experiments") is not True
        or sha(scale_path) != prepared["scales"][variant_name]["sha256"]
    ):
        raise ValueError("Invalid or modified Task2 preparation")
    with np.load(scale_path, allow_pickle=False) as handle:
        scales = {key: handle[key].copy() for key in handle.files}
    for key, channels in [
        ("d12", 12), ("ecg_machine_i", 1),
        ("ecg_machine_d6", 6), ("body_scale_d6", 6),
    ]:
        value = scales[key]
        if value.shape != (channels,) or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError(f"Invalid frozen scale: {key}")
    if not np.array_equal(scales["ecg_machine_i"], scales["d12"][:1]):
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
        PreprocessingConfig.from_yaml(cfg.path("preprocessing_config")), scales
    )
    with cfg.path("subject_split_csv").open(encoding="utf-8-sig", newline="") as handle:
        subject_split = {row["subject_id"]: row["split"] for row in csv.DictReader(handle)}
    train = JointTorchDataset(
        JointAnchorDataset(cfg, "task2", "train", body_scale_variant=variant_name),
        preprocessor, "train", subject_split, args.input_type,
    )
    validation = JointTorchDataset(
        JointAnchorDataset(cfg, "task2", "validation", body_scale_variant=variant_name),
        preprocessor, "validation", subject_split, args.input_type,
    )
    contexts, anchors, target, raw_anchor, context_masks, anchor_masks, metadata = validation_arrays(validation)
    train_subjects = {str(train.raw_sample(index).subject_id) for index in range(len(train))}
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != "P0_anchor_only" or checkpoint.get("task_id") != "task2":
        raise ValueError("Checkpoint is not a Task2 P0 checkpoint")
    if (
        checkpoint.get("architecture_id") != model.architecture_id
        or checkpoint.get("architecture_config_hash") != model.architecture_config_hash
    ):
        raise ValueError("P0 checkpoint architecture mismatch")
    if checkpoint.get("d12_scale_sha256") != array_sha(scales["d12"]):
        raise ValueError("P0 checkpoint D12 scale mismatch")
    if checkpoint.get("preparation_summary_sha256") != sha(summary_path):
        raise ValueError("P0 checkpoint preparation-summary mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.set_stage("P1-C3")

    loss_cfg = yaml.safe_load(
        (ROOT / "configs/losses.yaml").read_text(encoding="utf-8")
    )["joint_anchor_sync"]
    loss_weights = {
        "huber_weight": float(loss_cfg["huber_full_d12"]),
        "pcc_weight": float(loss_cfg["pcc_full_d12"]),
        "physiology_weight": float(loss_cfg["physiology"]),
        "observed_weight": float(loss_cfg["observed_anchor_consistency"]),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loader = DataLoader(
        train, batch_size=16, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    scale = tensor(scales["d12"])
    epochs = 1 if args.smoke_test else args.epochs

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scale_path, output / scale_path.name)
    shutil.copy2(checkpoint_path, output / "p0_initialization.pt")
    write_json(output / "config_snapshot.json", {
        **vars(args), "stage": "P1-C3", "task_id": "task2", "seed": seed,
        "body_scale_variant": variant_name, "epochs": epochs, "batch_size": 16,
        "training_scope": "single_context_source", "input_type": args.input_type,
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
        "scheduler": None, "early_stopping": False, "baseline_uV": 0,
        "initialization": "compatible_same_architecture_Task2_P0_checkpoint",
        "loss": "joint_anchor_sync_loss", "loss_weights": loss_weights,
        "context_target_relation": "same_subject_cross_time",
        "anchor_target_relation": "same_record_same_window",
        "context_target_pointwise_loss": "forbidden",
        "training_output_i_replacement": "forbidden",
        "fusion_mode": "film_gated_residual", "context_enabled": True,
        "architecture_id": model.architecture_id,
        "architecture_config": model.architecture_config,
        "architecture_config_hash": model.architecture_config_hash,
        "checkpoint_metric": "r_raw_12_official_raw_uV_v0",
        "train_windows": len(train), "validation_windows": len(validation),
        "d12_scale_sha256": array_sha(scales["d12"]),
        "variant_scale_sha256": sha(scale_path),
        "p0_checkpoint_sha256": sha(checkpoint_path),
    })
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for relative in [
        "scripts/train_b0_joint_task2_p1_c3_separate.py", "ecg12gen/models/b0_joint_anchor.py",
        "ecg12gen/losses.py", "ecg12gen/evaluate.py", "ecg12gen/preprocessing.py",
        "ecg12gen/dataset.py", "configs/losses.yaml",
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
            for step, (context, anchor, y, context_mask, anchor_mask) in enumerate(loader):
                if args.smoke_test and step >= 2:
                    break
                optimizer.zero_grad(set_to_none=True)
                prediction = model(context, anchor, context_mask, anchor_mask)
                loss = joint_anchor_sync_loss(
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

            raw = predict(model, contexts, anchors, context_masks, anchor_masks, scale)
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
                    "state_dict": model.state_dict(), "stage": "P1-C3", "task_id": "task2",
                    "variant": args.variant, "body_scale_variant": variant_name,
                    "architecture_id": model.architecture_id,
                    "architecture_config": model.architecture_config,
                    "architecture_config_hash": model.architecture_config_hash,
                    "d12_scale_sha256": array_sha(scales["d12"]),
                    "variant_scale_sha256": sha(scale_path),
                    "p0_checkpoint_sha256": sha(checkpoint_path),
                    "epoch": epoch, "smoke_test": args.smoke_test,
                }, output / "best_model.pt")
            print(
                f'P1-C3-{args.variant}-{args.input_type} | Epoch {epoch}/{epochs} | loss={row["loss"]:.6f} | '
                f'r_raw={row["r_raw_12"]:.6f} | r_submit={row["r_submit_12"]:.6f} | '
                f'r_missing11={row["r_missing11"]:.6f}', flush=True,
            )

    saved = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    raw, submit, metrics, raw_metrics, device_rows = write_full_evaluation(
        output / "evaluation", model, contexts, anchors, context_masks, anchor_masks,
        scale, target, raw_anchor, metadata,
        f"B0 Task2 P1-C3-{args.variant} submit-view validation",
    )

    permutation = device_stratified_permutation(metadata)
    shuffled_raw = predict(
        model, contexts[permutation], anchors, context_masks[permutation], anchor_masks, scale
    )
    shuffled_metrics, _, shuffled_details, shuffled_submit = evaluate_joint_anchor_predictions(
        shuffled_raw, target, raw_anchor, "task2"
    )
    _, _, shuffled_report = write_report(
        output / "evaluation/shuffled_context", shuffled_metrics, shuffled_details,
        title=f"B0 Task2 P1-C3-{args.variant} device-stratified shuffled-context diagnostic",
    )
    shuffled_subjects, shuffled_devices = evaluate_task2_diagnostics(
        shuffled_submit, target, metadata
    )
    write_task2_diagnostics(
        output / "evaluation/shuffled_context", shuffled_subjects,
        shuffled_devices, shuffled_report,
    )
    shuffled_delta = {
        key: float(metrics[key]) - float(shuffled_metrics[key])
        for key in ["r_raw_12", "r_submit_12", "r_missing11"]
    }

    np.save(output / "validation_prediction_raw.npy", raw)
    np.save(output / "validation_prediction_submit.npy", submit)
    np.save(output / "validation_anchor_i.npy", raw_anchor)
    np.save(output / "validation_context_shuffle_permutation.npy", permutation.numpy())
    with (output / "validation_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    write_json(output / "training_summary.json", {
        "completed": True, "stage": "P1-C3", "task_id": "task2",
        "variant": args.variant, "body_scale_variant": variant_name,
        "training_scope": "single_context_source", "input_type": args.input_type,
        "smoke_test": args.smoke_test, "epochs_completed": epochs,
        "best_epoch": best_epoch, "validation": metrics,
        "raw_prediction": raw_metrics, "device_diagnostics": device_rows,
        "shuffled_context_validation": shuffled_metrics,
        "shuffled_context_device_diagnostics": shuffled_devices,
        "real_minus_shuffled_context_r": shuffled_delta,
        "architecture_id": model.architecture_id,
        "architecture_config_hash": model.architecture_config_hash,
        "best_model_sha256": sha(output / "best_model.pt"),
        "p0_checkpoint_sha256": sha(checkpoint_path),
        "d12_scale_sha256": array_sha(scales["d12"]),
        "variant_scale_sha256": sha(scale_path),
    })
    print(f"Completed. Results: {output}")


if __name__ == "__main__":
    main()
