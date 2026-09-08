"""B2 P0/C1/C2 training using main's strict and joint-anchor losses."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b2_data import B2PreparedDataset, b2_collate
from .b2_model import B2JointAnchorPatchTransformer
from .evaluate import evaluate_joint_anchor_predictions, evaluate_task2_diagnostics
from .losses import joint_anchor_sync_loss, strict_anchor_pretrain_loss
from .training import seed_everything

P0_STAGE, P1_STAGE = "P0_anchor_only", "P1_joint_anchor"


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
    if task_id == "task2":
        subjects, devices = evaluate_task2_diagnostics(submit, target_array, [{"subject_id": str(x["subject_id"]), "input_type": str(x["input_type"])} for x in metadata])
        overall["task2_subject_macro_rows"], overall["task2_device_rows"] = len(subjects), len(devices)
    metric = float(overall["task1_r1" if task_id == "task1" else "task2_r2"])
    return ValidationResult(metric, overall, prediction.astype(np.float32), submit.astype(np.float32), anchor_array.astype(np.float32), target_array.astype(np.float32), metadata)


def _load_p0(model: B2JointAnchorPatchTransformer, checkpoint: str | Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != model.checkpoint_schema or payload.get("stage") != P0_STAGE:
        raise ValueError("P1 requires a compatible B2 P0_anchor_only checkpoint")
    model.load_state_dict(payload["model"], strict=True)


def _save_validation(output: Path, validation: ValidationResult) -> None:
    np.save(output / "prediction_raw.npy", validation.raw); np.save(output / "prediction_submit.npy", validation.submit)
    np.save(output / "validation_anchor_i.npy", validation.anchor); np.save(output / "validation_target.npy", validation.target)
    with (output / "validation_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["array_index", "subject_id", "input_type", "split"]); writer.writeheader()
        for i, row in enumerate(validation.metadata): writer.writerow({"array_index": i, "subject_id": row.get("subject_id", ""), "input_type": row.get("input_type", ""), "split": row.get("split", "validation")})


def fit_b2(model: B2JointAnchorPatchTransformer, train_dataset: B2PreparedDataset, validation_dataset: B2PreparedDataset,
           d12_scale_uV: np.ndarray, task_id: str, output_dir: str | Path, *, stage: str,
           p0_checkpoint: str | Path | None = None, device: str = "cpu", epochs: int = 100) -> Path:
    if stage not in {P0_STAGE, P1_STAGE} or not 1 <= epochs <= 100: raise ValueError("invalid B2 stage or epoch count")
    if stage == P0_STAGE and train_dataset.mode != "strict_anchor_pretrain": raise ValueError("P0 must use strict_anchor_pretrain data")
    if stage == P1_STAGE and (train_dataset.mode != "joint_anchor" or p0_checkpoint is None): raise ValueError("P1 must use joint-anchor data and an explicit P0 checkpoint")
    seed_everything(42, deterministic=True); device_t = torch.device(device); model.to(device_t)
    if stage == P1_STAGE: _load_p0(model, p0_checkpoint, device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    best, history = -float("inf"), []
    best_path = output / "b2_best.pt"
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for batch in _loader(train_dataset, True, 42):
            moved = _move(batch, device_t); optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, moved, task_id)
            loss = (strict_anchor_pretrain_loss(prediction, moved["target_model"], moved["anchor_model"]) if stage == P0_STAGE
                    else joint_anchor_sync_loss(prediction, moved["target_model"], moved["anchor_model"]))
            loss.backward(); optimizer.step(); total += float(loss.detach().cpu())
        validation = validate_v0(model, validation_dataset, np.asarray(d12_scale_uV, np.float32), task_id, device_t)
        history.append({"epoch": epoch, "train_loss": total / max(len(_loader(train_dataset, False, 42)), 1), "validation": validation.overall})
        if validation.metric_value > best:
            best = validation.metric_value; _save_validation(output, validation)
            torch.save({"schema": model.checkpoint_schema, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "stage": stage, "task_id": task_id, "model_config": asdict(model.config), "p0_checkpoint": str(p0_checkpoint) if p0_checkpoint else None,
                        "checkpoint_selection": "best_validation_raw_uV_submit_anchor_i_replaced_v0"}, best_path)
    (output / "history.json").write_text(json.dumps({"stage": stage, "checkpoint_selection": "raw_uV_submit_anchor_i_replaced_v0", "history": history}, indent=2, default=str), encoding="utf-8")
    return best_path
