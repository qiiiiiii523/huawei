"""B2-v1 P0/P1 training under the latest main joint-anchor contract."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b2_data import B2PreparedDataset, b2_collate
from .b2_model import B2MaskedPatchTransformer, architecture_metadata
from .evaluate import evaluate_centered_diagnostic, evaluate_joint_anchor_predictions, evaluate_task2_diagnostics
from .losses import joint_anchor_sync_loss, strict_anchor_pretrain_loss
from .training import seed_everything

P0_STAGE = "P0_anchor_only"
P1_STAGES = {"P1-C1", "P1-C2", "P1-C3"}
MAX_EPOCHS = 100
GRAD_CLIP_NORM = 1.0


def _loader(dataset: B2PreparedDataset, shuffle: bool, seed: int, batch_size: int = 16) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=False, collate_fn=b2_collate, generator=generator)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _forward(model: B2MaskedPatchTransformer, batch: dict[str, Any], stage: str) -> torch.Tensor:
    if stage == P0_STAGE:
        return model.forward_anchor(batch["anchor_model"], batch["anchor_lead_mask"])
    return model(batch["context_model"], batch["anchor_model"], batch["context_lead_mask"], batch["anchor_lead_mask"])


def _raw_prediction(output: np.ndarray, d12_scale_uV: np.ndarray) -> np.ndarray:
    # B2-v1 has no baseline head.  Its centered morphology is the documented
    # first-comparison raw view with a fixed zero-uV predicted baseline.
    return output.astype(np.float32) * np.asarray(d12_scale_uV, np.float32)[None, :, None]


@dataclass
class ValidationResult:
    metric_value: float
    summary: dict[str, Any]
    prediction_raw: np.ndarray
    prediction_submit: np.ndarray
    target_raw: np.ndarray
    anchor_raw: np.ndarray
    metadata: list[dict[str, Any]]


@torch.no_grad()
def validate_v0(model: B2MaskedPatchTransformer, dataset: B2PreparedDataset,
                d12_scale_uV: np.ndarray, task_id: str, device: torch.device,
                stage: str | None = None, batch_size: int = 16) -> ValidationResult:
    model.eval()
    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    stage = stage or (P0_STAGE if dataset.mode == "strict_anchor_pretrain" else "P1-C3")
    for batch in _loader(dataset, False, 42, batch_size):
        moved = _move(batch, device)
        outputs.append(_raw_prediction(_forward(model, moved, stage).cpu().numpy(), d12_scale_uV))
        targets.append(batch["raw_target_uV"].numpy())
        anchors.append(batch["raw_anchor_i_uV"].numpy())
        metadata.extend(batch["meta"])
    prediction_raw = np.concatenate(outputs).astype(np.float32)
    target_raw = np.concatenate(targets).astype(np.float32)
    anchor_raw = np.concatenate(anchors).astype(np.float32)
    summary, raw_details, submit_details, prediction_submit = evaluate_joint_anchor_predictions(
        prediction_raw, target_raw, anchor_raw, task_id)
    centered, centered_details = evaluate_centered_diagnostic(prediction_submit, target_raw, task_id)
    summary.update({"centered_r_12": float(centered["twelve_lead_mean_pearson_r"]),
                    "centered_rmse_uV": float(centered["twelve_lead_mean_rmse_uV"]),
                    "centered_r_missing11": float(np.nanmean([float(x["pearson_r"]) for x in centered_details[1:]])),
                    "raw_baseline_policy": "fixed_zero_uV"})
    if task_id == "task2":
        subjects, devices = evaluate_task2_diagnostics(prediction_submit, target_raw,
            [{"subject_id": str(x.get("subject_id", "")), "input_type": str(x.get("input_type", ""))} for x in metadata])
        summary["task2_subject_macro_r_submit_12"] = float(np.nanmean([x["twelve_lead_mean_pearson_r"] for x in subjects]))
        summary["task2_device_rows"] = len(devices)
    return ValidationResult(float(summary["r_submit_12"]), summary, prediction_raw,
                            prediction_submit, target_raw, anchor_raw, metadata)


def _save_validation(output: Path, result: ValidationResult) -> None:
    np.save(output / "prediction_raw.npy", result.prediction_raw)
    np.save(output / "prediction_submit.npy", result.prediction_submit)
    np.save(output / "validation_target.npy", result.target_raw)
    np.save(output / "validation_anchor_i.npy", result.anchor_raw)
    with (output / "validation_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["array_index", "subject_id", "input_type", "split"])
        writer.writeheader()
        for index, row in enumerate(result.metadata):
            writer.writerow({"array_index": index, "subject_id": row.get("subject_id", ""),
                             "input_type": row.get("input_type", row.get("device_type", "")),
                             "split": "validation"})


def _load_p0(model: B2MaskedPatchTransformer, checkpoint: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != model.checkpoint_schema or payload.get("stage") != P0_STAGE:
        raise ValueError("P1 requires a compatible B2 P0 checkpoint")
    expected = architecture_metadata(model.config)
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"P0 architecture mismatch: expected {expected}, got {actual}")
    model.load_state_dict(payload["model"], strict=True)
    return actual


def fit_b2(model: B2MaskedPatchTransformer, train_dataset: B2PreparedDataset,
           validation_dataset: B2PreparedDataset, d12_scale_uV: np.ndarray,
           task_id: str, output_dir: str | Path, *, stage: str,
           p0_checkpoint: str | Path | None = None, device: str = "cpu",
           epochs: int = MAX_EPOCHS, validation_views: Mapping[str, B2PreparedDataset] | None = None) -> Path:
    if stage not in {P0_STAGE, *P1_STAGES} or not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError("invalid B2 stage or epoch count")
    if stage == P0_STAGE and train_dataset.mode != "strict_anchor_pretrain":
        raise ValueError("P0 requires strict train-only data")
    if stage in P1_STAGES and (train_dataset.mode != "joint_anchor" or p0_checkpoint is None):
        raise ValueError("P1 requires joint-anchor data and an explicit P0 checkpoint")
    seed_everything(42, deterministic=True)
    device_t = torch.device(device)
    model.to(device_t)
    p0_architecture = _load_p0(model, p0_checkpoint, device_t) if stage in P1_STAGES else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    train_loader = _loader(train_dataset, True, 42)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    best = -float("inf"); best_path = output / "b2_best.pt"; history: list[dict[str, Any]] = []
    scale_t = torch.as_tensor(d12_scale_uV, dtype=torch.float32, device=device_t)
    for epoch in range(1, epochs + 1):
        model.train(); losses: list[float] = []
        for batch in train_loader:
            moved = _move(batch, device_t); optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, moved, stage)
            if stage == P0_STAGE:
                loss = strict_anchor_pretrain_loss(prediction, moved["target_model"], moved["anchor_model"], d12_scale_uV=scale_t)
            else:
                loss = joint_anchor_sync_loss(prediction, moved["target_model"], moved["anchor_model"], d12_scale_uV=scale_t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = validate_v0(model, validation_dataset, d12_scale_uV, task_id, device_t, stage)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation.summary}
        history.append(row)
        if validation.metric_value > best:
            best = validation.metric_value; _save_validation(output, validation)
            architecture = architecture_metadata(model.config)
            torch.save({"schema": model.checkpoint_schema, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "stage": stage, "task_id": task_id, "model_config": asdict(model.config),
                        "architecture_id": architecture["architecture_id"], "architecture_config_hash": architecture["architecture_config_hash"],
                        "p0_architecture": p0_architecture, "d12_scale_uV": np.asarray(d12_scale_uV, np.float32).tolist(),
                        "checkpoint_selection": "best_validation_official_raw_uV_submit_anchor_i_v0",
                        "raw_baseline_policy": "fixed_zero_uV"}, best_path)
    (output / "history.json").write_text(json.dumps({"stage": stage, "history": history}, indent=2, default=str), encoding="utf-8")
    return best_path
