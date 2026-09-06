"""Training routes for M1 using the frozen B2 data and loss infrastructure."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b2_data import B2PreparedDataset, b2_collate
from .b2_train import (
    check_weak_protocol_compatibility,
    quality_weighted_pseudo_loss,
    strict_missing_loss,
    weak_route_loss,
)
from .evaluate import evaluate_predictions
from .losses import observed_consistency_loss, physiology_constraint_loss
from .m1_model import M1MaskedCNNLeadTimeTransformer
from .training import seed_everything


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _loader(dataset: B2PreparedDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=False, collate_fn=b2_collate, generator=generator,
    )


def _forward(model: M1MaskedCNNLeadTimeTransformer, batch: dict[str, Any]) -> torch.Tensor:
    """The B2 prepared batch supplies the mandatory common d12 observed view."""
    return model(
        batch["input_model"], batch["lead_mask"], batch["missing_mask"],
        observed_d12_model=batch["observed_d12_model"],
    )


def strict_route_loss(prediction: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Strict same-window reconstruction using shared B2 primitives and weights."""
    return (
        strict_missing_loss(prediction, batch["target_model"], batch["missing_mask"])
        + 0.05 * physiology_constraint_loss(prediction)
        + 0.02 * observed_consistency_loss(prediction, batch["observed_d12_model"], batch["lead_mask"])
    )


def pseudo_route_loss(prediction: torch.Tensor, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Accepted pseudo-pairs: quality-weighted missing-lead loss plus shared constraints."""
    quality = torch.as_tensor(
        [float(meta.get("alignment_quality_score", 0.0)) for meta in batch["meta"]],
        dtype=torch.float32, device=device,
    )
    return (
        quality_weighted_pseudo_loss(prediction, batch["target_model"], batch["missing_mask"], quality)
        + 0.05 * physiology_constraint_loss(prediction)
        + 0.10 * observed_consistency_loss(prediction, batch["observed_d12_model"], batch["lead_mask"])
    )


def _model_to_morphology(model_output: np.ndarray, d12_scale_uV: np.ndarray) -> np.ndarray:
    return np.asarray(model_output, dtype=np.float32) * np.asarray(d12_scale_uV, dtype=np.float32)[None, :, None]


@dataclass(frozen=True)
class ValidationResult:
    metric_name: str
    metric_value: float
    overall: dict[str, Any]


@torch.no_grad()
def validate_v0(
    model: M1MaskedCNNLeadTimeTransformer,
    dataset: B2PreparedDataset,
    d12_scale_uV: np.ndarray,
    task_id: str,
    device: torch.device,
) -> ValidationResult:
    """Run the unchanged official raw-uV V0 implementation on validation only."""
    model.eval()
    predictions, targets, masks = [], [], []
    for batch in _loader(dataset, 16, False, 42):
        moved = _move_batch(batch, device)
        predictions.append(_forward(model, moved).detach().cpu().numpy())
        targets.append(batch["raw_target_uV"].numpy())
        masks.append(batch["lead_mask"].numpy())
    prediction_uV = _model_to_morphology(np.concatenate(predictions), d12_scale_uV)
    target_uV, lead_mask = np.concatenate(targets), np.concatenate(masks)
    overall, _ = evaluate_predictions(prediction_uV, target_uV, task_id, lead_mask)
    metric_name = "task1_r1" if task_id == "task1" else "task2_r2"
    return ValidationResult(metric_name, float(overall[metric_name]), overall)


def _reference_tensor(dataset: B2PreparedDataset | None, device: torch.device) -> torch.Tensor | None:
    if dataset is None:
        return None
    values = [dataset[index].target_model.numpy() for index in range(len(dataset))]
    return torch.from_numpy(np.stack(values)).to(device)


def _next_batch(iterator: Any, loader: DataLoader) -> tuple[dict[str, Any], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _run_epoch(
    model: M1MaskedCNNLeadTimeTransformer,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    strict_loader: DataLoader | None = None,
    adaptation_loader: DataLoader | None = None,
    strict_reference: torch.Tensor | None = None,
    weak_profile: str = "A0",
) -> float:
    model.train()
    total, steps = 0.0, 0
    strict_iterator = iter(strict_loader) if strict_loader is not None else None
    adaptation_iterator = iter(adaptation_loader) if adaptation_loader is not None else None
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if mode == "strict":
            moved = _move_batch(batch, device)
            loss = strict_route_loss(_forward(model, moved), moved)
        elif mode == "weak":
            if strict_reference is None:
                raise ValueError("weak route requires the strict train d12 reference bank")
            moved = _move_batch(batch, device)
            loss = weak_route_loss(_forward(model, moved), moved, strict_reference, weak_profile)
        elif mode in {"mixed_weak", "mixed_pseudo"}:
            if strict_iterator is None or adaptation_iterator is None:
                raise ValueError(f"{mode} requires strict and adaptation data")
            strict_batch, strict_iterator = _next_batch(strict_iterator, strict_loader)  # type: ignore[arg-type]
            adaptation_batch, adaptation_iterator = _next_batch(adaptation_iterator, adaptation_loader)  # type: ignore[arg-type]
            strict_batch, adaptation_batch = _move_batch(strict_batch, device), _move_batch(adaptation_batch, device)
            strict_loss = strict_route_loss(_forward(model, strict_batch), strict_batch)
            if mode == "mixed_weak":
                if strict_reference is None:
                    raise ValueError("mixed weak route requires the strict train d12 reference bank")
                adaptation_loss = weak_route_loss(
                    _forward(model, adaptation_batch), adaptation_batch, strict_reference, weak_profile
                )
            else:
                adaptation_loss = pseudo_route_loss(_forward(model, adaptation_batch), adaptation_batch, device)
            loss = strict_loss + 0.20 * adaptation_loss
        else:
            raise ValueError(f"Unknown M1 training mode: {mode}")
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


def fit_m1(
    model: M1MaskedCNNLeadTimeTransformer,
    strict_dataset: B2PreparedDataset,
    validation_dataset: B2PreparedDataset,
    d12_scale_uV: np.ndarray,
    task_id: str,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    epochs: int = 100,
    route: str,
    stage: str,
    weak_dataset: B2PreparedDataset | None = None,
    pseudo_dataset: B2PreparedDataset | None = None,
    strict_reference_dataset: B2PreparedDataset | None = None,
    weak_profile: str = "A0",
) -> Path:
    """Train M1 without adapters, baseline heads, or stochastic lead masking."""
    if task_id not in {"task1", "task2"} or route not in {"A", "B", "C"} or stage not in {"strict", "mixed"}:
        raise ValueError("task_id, route, or stage is invalid")
    if not 1 <= epochs <= 100:
        raise ValueError("M1 epochs must be between 1 and 100")
    if task_id == "task2" and route == "C":
        raise ValueError("Route C is task1-only")
    if route == "C" and stage != "mixed":
        raise ValueError("Route C requires stage=mixed")
    if route == "B" and stage == "mixed" and weak_dataset is None:
        raise ValueError("Route B mixed requires weak_dataset")
    if route == "C" and pseudo_dataset is None:
        raise ValueError("Route C requires pseudo_dataset")
    if route in {"A", "B"}:
        check_weak_protocol_compatibility(
            Path(__file__).resolve().parents[1] / "configs" / "training_protocol_v1.yaml",
            Path(__file__).resolve().parents[1] / "configs" / "losses.yaml",
        )

    seed_everything(42, deterministic=True)
    target_device = torch.device(device)
    model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    strict_loader = _loader(strict_dataset, 16, True, 42)
    weak_loader = _loader(weak_dataset, 16, True, 43) if weak_dataset is not None else None
    pseudo_loader = _loader(pseudo_dataset, 16, True, 44) if pseudo_dataset is not None else None
    strict_reference = _reference_tensor(strict_reference_dataset, target_device)

    if route == "A":
        if weak_loader is None:
            raise ValueError("Route A requires weak_dataset")
        epoch_loader, mode = weak_loader, "weak"
    elif route == "B" and stage == "strict":
        epoch_loader, mode = strict_loader, "strict"
    elif route == "B":
        epoch_loader, mode = strict_loader, "mixed_weak"
    else:
        epoch_loader, mode = strict_loader, "mixed_pseudo"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_value, best_epoch = -float("inf"), None
    best_path = output / "m1_best.pt"
    for epoch in range(1, epochs + 1):
        loss = _run_epoch(
            model, optimizer, epoch_loader, target_device, mode,
            strict_loader=strict_loader if mode.startswith("mixed") else None,
            adaptation_loader=weak_loader if mode == "mixed_weak" else pseudo_loader if mode == "mixed_pseudo" else None,
            strict_reference=strict_reference, weak_profile=weak_profile,
        )
        validation = validate_v0(model, validation_dataset, d12_scale_uV, task_id, target_device)
        history.append({"epoch": epoch, "train_loss": loss, "validation": validation.overall})
        if validation.metric_value > best_value:
            best_value, best_epoch = validation.metric_value, epoch
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "parameter_count": model.parameter_count, "task_id": task_id, "route": route,
                 "stage": stage, "adapter": False, "baseline_head": False},
                best_path,
            )
    (output / "history.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_metric": best_value, "history": history}, indent=2, default=str),
        encoding="utf-8",
    )
    return best_path
