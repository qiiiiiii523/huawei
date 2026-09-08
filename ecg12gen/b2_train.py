"""B2 P0/P1-C1/P1-C2/P1-C3 training using main's losses."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b2_data import B2PreparedDataset, b2_collate
from .b2_model import B2JointAnchorPatchTransformer, architecture_metadata
from .evaluate import evaluate_centered_diagnostic, evaluate_joint_anchor_predictions, evaluate_task2_diagnostics
from .losses import joint_anchor_sync_loss, strict_anchor_pretrain_loss
from .training import seed_everything

P0_STAGE, P1_STAGE = "P0_anchor_only", "P1_joint_anchor"
MAX_EPOCHS = 200
GRAD_CLIP_NORM = 1.0


def _loader(dataset: B2PreparedDataset, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(dataset, batch_size=16, shuffle=shuffle, num_workers=0, pin_memory=False,
                      collate_fn=b2_collate, generator=torch.Generator().manual_seed(seed))


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _forward(model: B2JointAnchorPatchTransformer, batch: dict[str, Any], task_id: str) -> torch.Tensor:
    return model(batch["anchor_model"], batch["anchor_lead_mask"], task_id=task_id,
                 watch_context=batch["watch_context_model"], watch_available=batch["watch_available"],
                 machine_d6_context=batch["machine_d6_model"], machine_d6_mask=batch["machine_d6_mask"], machine_available=batch["machine_available"],
                 body_d6_context=batch["body_d6_model"], body_d6_mask=batch["body_d6_mask"], body_available=batch["body_available"])


@dataclass
class ValidationResult:
    metric_value: float
    overall: dict[str, Any]
    raw: np.ndarray
    submit: np.ndarray
    anchor: np.ndarray
    target: np.ndarray
    metadata: list[dict[str, Any]]


@torch.no_grad()
def validate_v0(model: B2JointAnchorPatchTransformer, dataset: B2PreparedDataset, d12_scale_uV: np.ndarray,
                task_id: str, device: torch.device) -> ValidationResult:
    model.eval(); raw, target, anchor, metadata = [], [], [], []
    for batch in _loader(dataset, False, 42):
        moved = _move(batch, device)
        raw.append(_forward(model, moved, task_id).cpu().numpy() * d12_scale_uV[None, :, None])
        target.append(batch["raw_target_uV"].numpy()); anchor.append(batch["raw_anchor_i_uV"].numpy()); metadata.extend(batch["meta"])
    prediction, target_array, anchor_array = np.concatenate(raw), np.concatenate(target), np.concatenate(anchor)
    overall, _, _, submit = evaluate_joint_anchor_predictions(prediction, target_array, anchor_array, task_id)
    centered, centered_details = evaluate_centered_diagnostic(submit, target_array, task_id)
    overall["centered_r_12"] = float(centered["twelve_lead_mean_pearson_r"])
    overall["centered_rmse_uV"] = float(centered["twelve_lead_mean_rmse_uV"])
    overall["centered_r_missing11"] = float(np.nanmean([float(row["pearson_r"]) for row in centered_details[1:]]))
    if task_id == "task2":
        subjects, devices = evaluate_task2_diagnostics(
            submit, target_array,
            [{"subject_id": str(x.get("subject_id", "")), "input_type": str(x.get("input_type", "strict"))} for x in metadata],
        )
        overall["task2_subject_macro_rows"], overall["task2_device_rows"] = len(subjects), len(devices)
    metric = float(overall["task1_r1" if task_id == "task1" else "task2_r2"])
    return ValidationResult(metric, overall, prediction.astype(np.float32), submit.astype(np.float32), anchor_array.astype(np.float32), target_array.astype(np.float32), metadata)


def _load_p0(model: B2JointAnchorPatchTransformer, checkpoint: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != model.checkpoint_schema or payload.get("stage") != P0_STAGE:
        raise ValueError("P1 requires a compatible B2 P0_anchor_only checkpoint")
    expected = architecture_metadata(model.config)
    legacy_metadata = False
    actual = {key: payload.get(key) for key in ("architecture_id", "architecture_config_hash")}
    if not all(actual.values()):
        model_config = payload.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("P0 checkpoint is missing architecture metadata and model_config")
        actual = architecture_metadata(model_config)
        legacy_metadata = True
    if actual != expected:
        raise ValueError(f"P0 architecture mismatch: expected {expected}, got {actual}")
    model.load_state_dict(payload["model"], strict=True)
    if model.config.fusion_mode in {"gated_residual", "film_gated_residual"}:
        with torch.no_grad():
            model.gate.bias.fill_(math.log(model.config.initial_gate / (1.0 - model.config.initial_gate)))
    return {**actual, "legacy_metadata_derived": legacy_metadata}


def _save_validation(output: Path, validation: ValidationResult) -> None:
    np.save(output / "prediction_raw.npy", validation.raw); np.save(output / "prediction_submit.npy", validation.submit)
    np.save(output / "validation_anchor_i.npy", validation.anchor); np.save(output / "validation_target.npy", validation.target)
    with (output / "validation_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["array_index", "subject_id", "input_type", "split"]); writer.writeheader()
        for i, row in enumerate(validation.metadata): writer.writerow({"array_index": i, "subject_id": row.get("subject_id", ""), "input_type": row.get("input_type", ""), "split": row.get("split", "validation")})


def _strict_train_metrics(model: B2JointAnchorPatchTransformer, dataset: B2PreparedDataset,
                          d12_scale_uV: np.ndarray, task_id: str, device: torch.device,
                          checkpoint_epoch: int) -> dict[str, Any]:
    """Report strict-pretraining R on the train-only strict diagnostic set.

    The strict protocol intentionally has no strict validation split.  These
    metrics are therefore diagnostic train-set R values and are never used for
    checkpoint selection.
    """
    result = validate_v0(model, dataset, d12_scale_uV, task_id, device)
    return {
        "split": "train",
        "diagnostic_only": True,
        "evaluation_input_contract": "strict_anchor_pretrain_train_diagnostic",
        "checkpoint_epoch": checkpoint_epoch,
        "n_windows": len(dataset),
        "r_raw_12": result.overall["r_raw_12"],
        "r_submit_12": result.overall["r_submit_12"],
        "r_missing11": result.overall["r_missing11"],
        "twelve_lead_mean_rmse_uV": result.overall["twelve_lead_mean_rmse_uV"],
    }


def fit_b2(model: B2JointAnchorPatchTransformer, train_dataset: B2PreparedDataset, validation_dataset: B2PreparedDataset,
           d12_scale_uV: np.ndarray, task_id: str, output_dir: str | Path, *, stage: str,
           p0_checkpoint: str | Path | None = None, device: str = "cpu", epochs: int = MAX_EPOCHS,
           validation_views: Mapping[str, B2PreparedDataset] | None = None) -> Path:
    if stage not in {P0_STAGE, P1_STAGE} or not 1 <= epochs <= MAX_EPOCHS: raise ValueError("invalid B2 stage or epoch count")
    if stage == P0_STAGE and train_dataset.mode != "strict_anchor_pretrain": raise ValueError("P0 must use strict_anchor_pretrain data")
    if stage == P1_STAGE and (train_dataset.mode != "joint_anchor" or p0_checkpoint is None): raise ValueError("P1 must use joint-anchor data and an explicit P0 checkpoint")
    seed_everything(42, deterministic=True); device_t = torch.device(device); model.to(device_t)
    p0_architecture: dict[str, Any] | None = None
    if stage == P1_STAGE: p0_architecture = _load_p0(model, p0_checkpoint, device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    best, history = -float("inf"), []
    best_path = output / "b2_best.pt"
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for batch in _loader(train_dataset, True, 42):
            moved = _move(batch, device_t); optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, moved, task_id)
            loss = (strict_anchor_pretrain_loss(prediction, moved["target_model"], moved["anchor_model"],
                                                d12_scale_uV=d12_scale_uV) if stage == P0_STAGE
                    else joint_anchor_sync_loss(prediction, moved["target_model"], moved["anchor_model"],
                                                d12_scale_uV=d12_scale_uV))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step(); total += float(loss.detach().cpu())
        validation = validate_v0(model, validation_dataset, np.asarray(d12_scale_uV, np.float32), task_id, device_t)
        history.append({"epoch": epoch, "train_loss": total / max(len(_loader(train_dataset, False, 42)), 1), "validation": validation.overall})
        if validation.metric_value > best:
            best = validation.metric_value; _save_validation(output, validation)
            architecture = architecture_metadata(model.config)
            torch.save({"schema": model.checkpoint_schema, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "stage": stage, "task_id": task_id, "model_config": asdict(model.config), "p0_checkpoint": str(p0_checkpoint) if p0_checkpoint else None,
                        "architecture_id": architecture["architecture_id"], "architecture_config_hash": architecture["architecture_config_hash"],
                        "p0_architecture": p0_architecture,
                        "d12_scale_uV": np.asarray(d12_scale_uV, dtype=np.float32).tolist(),
                        "checkpoint_selection": "best_validation_raw_uV_submit_anchor_i_replaced_v0"}, best_path)
    history_payload: dict[str, Any] = {"stage": stage, "epochs_requested": epochs,
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                       "gradient_clip_norm": GRAD_CLIP_NORM},
        "checkpoint_selection": "raw_uV_submit_anchor_i_replaced_v0", "history": history}
    best_payload = None
    if stage == P0_STAGE or validation_views:
        best_payload = torch.load(best_path, map_location=device_t, weights_only=False)
        model.load_state_dict(best_payload["model"], strict=True)
    validation_view_metrics: dict[str, dict[str, Any]] = {}
    if validation_views:
        for view_name, view_dataset in validation_views.items():
            view_result = validate_v0(model, view_dataset, np.asarray(d12_scale_uV, np.float32), task_id, device_t)
            view_metrics = {**view_result.overall, "context_view": view_name,
                            "checkpoint_epoch": int(best_payload["epoch"])}
            validation_view_metrics[view_name] = view_metrics
            (output / f"validation_{view_name}_metrics.json").write_text(
                json.dumps(view_metrics, indent=2, default=str), encoding="utf-8")
    if stage == P0_STAGE:
        strict_metrics = _strict_train_metrics(model, train_dataset, np.asarray(d12_scale_uV, np.float32),
                                                task_id, device_t, int(best_payload["epoch"]))
        history_payload["strict_train_best_checkpoint"] = strict_metrics
        (output / "strict_train_metrics.json").write_text(json.dumps(strict_metrics, indent=2, default=str), encoding="utf-8")
    if validation_view_metrics:
        history_payload["validation_views"] = validation_view_metrics
    (output / "history.json").write_text(json.dumps(history_payload, indent=2, default=str), encoding="utf-8")
    return best_path
