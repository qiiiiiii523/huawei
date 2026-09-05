"""B2 training routes, loss contracts, and validation checkpoint selection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .b2_data import B2PreparedDataset, b2_collate
from .b2_model import B2MaskedPatchTransformer
from .evaluate import evaluate_predictions
from .losses import (
    masked_huber_loss,
    masked_pcc_loss,
    observed_consistency_loss,
    pair_invariant_stat_loss,
    physiology_constraint_loss,
    spectral_stat_loss,
)
from .training import seed_everything


class WeakProtocolConflictError(RuntimeError):
    """Raised when the shared weak-pair observed-consistency rule is invalid."""


def check_weak_protocol_compatibility(
    training_protocol_path: str | Path,
    losses_path: str | Path,
) -> None:
    with Path(training_protocol_path).open(encoding="utf-8") as handle:
        training = yaml.safe_load(handle)
    with Path(losses_path).open(encoding="utf-8") as handle:
        losses = yaml.safe_load(handle)
    policy = training["supervision_and_loss"]["observed_lead_consistency"]
    expected_policy = {
        "synchronous_d12": "permitted",
        "cross_device_weak_pair": "permitted_as_low_weight_input_only",
        "paired_d12_pointwise_reconstruction": "forbidden",
    }
    if policy != expected_policy:
        raise WeakProtocolConflictError(
            "Weak-pair training is blocked: training_protocol_v1 does not declare the "
            "方案 1 observed-consistency policy."
        )
    for profile in ("raw_weak", "raw_weak_a0", "raw_weak_a1"):
        weak_weight = float(losses.get(profile, {}).get("observed_consistency", 0.0))
        if weak_weight > 0.20:
            raise WeakProtocolConflictError(
                f"Weak-pair observed_consistency for {profile} must remain low-weight; "
                f"got {weak_weight}."
            )

def strict_missing_loss(prediction: torch.Tensor, target: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    """Strict reconstruction objective; every reconstruction term is missing-lead-only."""
    return masked_huber_loss(prediction, target, missing_mask) + 0.10 * masked_pcc_loss(prediction, target, missing_mask)


def weak_route_loss(
    prediction: torch.Tensor,
    batch: dict[str, Any],
    strict_reference: torch.Tensor,
    profile: str = "A0",
) -> torch.Tensor:
    """A0/A1 weak objective without any paired-d12 pointwise loss."""
    if profile not in {"A0", "A1"}:
        raise ValueError("weak profile must be A0 or A1")
    loss = 0.10 * observed_consistency_loss(prediction, batch["observed_d12_model"], batch["lead_mask"])
    loss = loss + 0.20 * spectral_stat_loss(prediction, strict_reference)
    loss = loss + 0.05 * physiology_constraint_loss(prediction)
    if profile == "A1":
        loss = loss + 0.20 * pair_invariant_stat_loss(prediction, batch["target_model"])
    return loss


def quality_weighted_pseudo_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    """Pseudo pointwise loss, weighted by alignment quality and missing leads only."""
    if quality.ndim != 1 or quality.shape[0] != prediction.shape[0]:
        raise ValueError("quality must have shape [batch]")
    lead_weight = missing_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    sample_weight = quality.to(dtype=prediction.dtype).view(-1, 1, 1)
    pointwise = torch.nn.functional.huber_loss(prediction, target, reduction="none")
    denominator = (lead_weight * sample_weight).sum().clamp_min(1.0) * prediction.shape[-1]
    huber = (pointwise * lead_weight * sample_weight).sum() / denominator

    p = prediction - prediction.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    correlation = (p * t).sum(dim=-1) / torch.sqrt((p.square().sum(dim=-1) * t.square().sum(dim=-1)).clamp_min(1e-8))
    pcc_weight = missing_mask.to(dtype=prediction.dtype) * quality[:, None]
    pcc = 1.0 - (correlation * pcc_weight).sum() / pcc_weight.sum().clamp_min(1.0)
    return huber + 0.10 * pcc


def _quality_from_batch(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    values = [float(meta.get("alignment_quality_score", 0.0)) for meta in batch["meta"]]
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _loader(dataset: B2PreparedDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False,
                      collate_fn=b2_collate, generator=generator)


def _model_to_morphology(model_output: np.ndarray, d12_scale_uV: np.ndarray) -> np.ndarray:
    return np.asarray(model_output, dtype=np.float32) * np.asarray(d12_scale_uV, dtype=np.float32)[None, :, None]


@dataclass(frozen=True)
class ValidationResult:
    metric_name: str
    metric_value: float
    overall: dict[str, Any]


@torch.no_grad()
def validate_v0(
    model: B2MaskedPatchTransformer,
    dataset: B2PreparedDataset,
    d12_scale_uV: np.ndarray,
    task_id: str,
    device: torch.device,
    batch_size: int = 16,
) -> ValidationResult:
    loader = _loader(dataset, batch_size, False, 42)
    model.eval()
    predictions, targets, masks = [], [], []
    for batch in loader:
        moved = _move_batch(batch, device)
        output = model(moved["input_model"], moved["lead_mask"], moved["missing_mask"])
        predictions.append(output.detach().cpu().numpy())
        targets.append(batch["raw_target_uV"].numpy())
        masks.append(batch["lead_mask"].numpy())
    prediction_uV = _model_to_morphology(np.concatenate(predictions), d12_scale_uV)
    target_uV = np.concatenate(targets)
    lead_mask = np.concatenate(masks)
    overall, _ = evaluate_predictions(prediction_uV, target_uV, task_id, lead_mask)
    metric_name = "task1_r1" if task_id == "task1" else "task2_r2"
    return ValidationResult(metric_name, float(overall[metric_name]), overall)


def _next_batch(iterator: Any, loader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _run_epoch(
    model: B2MaskedPatchTransformer,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    strict_loader: DataLoader | None = None,
    weak_loader: DataLoader | None = None,
    pseudo_loader: DataLoader | None = None,
    strict_reference: torch.Tensor | None = None,
    mode: str = "strict",
    weak_profile: str = "A0",
) -> float:
    model.train()
    total = 0.0
    steps = 0
    strict_iterator = iter(strict_loader) if strict_loader is not None else None
    weak_iterator = iter(weak_loader) if weak_loader is not None else None
    pseudo_iterator = iter(pseudo_loader) if pseudo_loader is not None else None
    if mode == "mixed" and (strict_loader is None or weak_loader is None or strict_reference is None):
        raise ValueError("mixed mode requires strict and weak loaders plus a reference bank")
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if mode == "weak":
            if strict_reference is None:
                raise ValueError("weak mode requires a strict reference bank")
            moved = _move_batch(batch, device)
            output = model(moved["input_model"], moved["lead_mask"], moved["missing_mask"])
            loss = weak_route_loss(output, moved, strict_reference, weak_profile)
        elif mode == "strict":
            moved = _move_batch(batch, device)
            output = model(moved["input_model"], moved["lead_mask"], moved["missing_mask"])
            loss = strict_missing_loss(output, moved["target_model"], moved["missing_mask"])
        elif mode == "mixed":
            strict_batch, strict_iterator = _next_batch(strict_iterator, strict_loader)  # type: ignore[arg-type]
            weak_batch, weak_iterator = _next_batch(weak_iterator, weak_loader)  # type: ignore[arg-type]
            strict_batch = _move_batch(strict_batch, device)
            weak_batch = _move_batch(weak_batch, device)
            strict_output = model(strict_batch["input_model"], strict_batch["lead_mask"], strict_batch["missing_mask"])
            weak_output = model(weak_batch["input_model"], weak_batch["lead_mask"], weak_batch["missing_mask"])
            strict_loss = strict_missing_loss(strict_output, strict_batch["target_model"], strict_batch["missing_mask"])
            weak_loss = weak_route_loss(weak_output, weak_batch, strict_reference, weak_profile)
            loss = strict_loss + 0.20 * weak_loss
        elif mode == "pseudo":
            strict_batch, strict_iterator = _next_batch(strict_iterator, strict_loader)  # type: ignore[arg-type]
            pseudo_batch, pseudo_iterator = _next_batch(pseudo_iterator, pseudo_loader)  # type: ignore[arg-type]
            strict_batch = _move_batch(strict_batch, device)
            pseudo_batch = _move_batch(pseudo_batch, device)
            strict_output = model(strict_batch["input_model"], strict_batch["lead_mask"], strict_batch["missing_mask"])
            pseudo_output = model(pseudo_batch["input_model"], pseudo_batch["lead_mask"], pseudo_batch["missing_mask"])
            strict_loss = strict_missing_loss(strict_output, strict_batch["target_model"], strict_batch["missing_mask"])
            pseudo_loss = quality_weighted_pseudo_loss(
                pseudo_output, pseudo_batch["target_model"], pseudo_batch["missing_mask"], _quality_from_batch(pseudo_batch, device)
            )
            loss = strict_loss + 0.20 * pseudo_loss
        else:
            raise ValueError(f"unknown training mode: {mode}")
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


def fit_b2(
    model: B2MaskedPatchTransformer,
    train_dataset: B2PreparedDataset,
    validation_dataset: B2PreparedDataset,
    d12_scale_uV: np.ndarray,
    task_id: str,
    output_dir: str | Path,
    device: str = "cpu",
    epochs: int = 100,
    route: str = "B",
    stage: str = "strict",
    weak_dataset: B2PreparedDataset | None = None,
    pseudo_dataset: B2PreparedDataset | None = None,
    strict_reference_dataset: B2PreparedDataset | None = None,
    weak_profile: str = "A0",
) -> Path:
    """Train a permitted route with fixed optimizer and V0 checkpoint selection."""
    if epochs < 1 or epochs > 100:
        raise ValueError("B2 epochs must be between 1 and 100")
    if route == "A" or (route == "B" and stage == "mixed"):
        check_weak_protocol_compatibility(
            Path(__file__).resolve().parents[1] / "configs" / "training_protocol_v1.yaml",
            Path(__file__).resolve().parents[1] / "configs" / "losses.yaml",
        )
    if route == "B" and stage == "mixed" and weak_dataset is None:
        raise ValueError("Route B mixed stage requires weak_dataset")
    if route == "C" and (task_id != "task1" or pseudo_dataset is None):
        raise ValueError("Route C is task1-only and requires pseudo_dataset")

    seed_everything(42, deterministic=True)
    target_device = torch.device(device)
    model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    strict_loader = _loader(train_dataset, 16, True, 42)
    weak_loader = _loader(weak_dataset, 16, True, 43) if weak_dataset is not None else None
    pseudo_loader = _loader(pseudo_dataset, 16, True, 44) if pseudo_dataset is not None else None
    train_loader = strict_loader
    if route == "A":
        train_loader = weak_loader  # type: ignore[assignment]
    if train_loader is None:
        raise ValueError("no training loader")

    reference_tensor: torch.Tensor | None = None
    if strict_reference_dataset is not None:
        references = [strict_reference_dataset[index].target_model.numpy() for index in range(len(strict_reference_dataset))]
        reference_tensor = torch.from_numpy(np.stack(references)).to(target_device)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_value = -float("inf")
    best_epoch = None
    best_path = output / "b2_best.pt"
    for epoch in range(1, epochs + 1):
        if route == "A":
            loss = _run_epoch(model, optimizer, train_loader, target_device,
                              strict_reference=reference_tensor, mode="weak", weak_profile=weak_profile)
        elif route == "B" and stage == "strict":
            loss = _run_epoch(model, optimizer, train_loader, target_device, mode="strict")
        elif route == "B" and stage == "mixed":
            loss = _run_epoch(model, optimizer, train_loader, target_device, strict_loader=strict_loader,
                              weak_loader=weak_loader, strict_reference=reference_tensor, mode="mixed", weak_profile=weak_profile)
        elif route == "C":
            loss = _run_epoch(model, optimizer, train_loader, target_device, strict_loader=strict_loader,
                              pseudo_loader=pseudo_loader, mode="pseudo")
        else:
            raise ValueError("unsupported B2 route/stage")
        validation = validate_v0(model, validation_dataset, d12_scale_uV, task_id, target_device)
        row = {"epoch": epoch, "train_loss": loss, "validation": validation.overall}
        history.append(row)
        if validation.metric_value > best_value:
            best_value = validation.metric_value
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "parameter_count": model.parameter_count, "task_id": task_id, "route": route, "stage": stage}, best_path)
    (output / "history.json").write_text(json.dumps({"best_epoch": best_epoch, "history": history}, indent=2, default=str), encoding="utf-8")
    return best_path
