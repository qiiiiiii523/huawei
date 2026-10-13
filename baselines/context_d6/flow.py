"""Flow-matching loss and sampling for the machine-context B5 extension."""
from __future__ import annotations

from typing import Any

import torch

from baselines.B5.flow import integrate, keyed_noise, linear_path
from baselines.B5.objective import validate_mainline_loss
from ecg12gen.losses import masked_huber_loss, masked_pcc_loss, physiology_constraint_loss

CONTEXT_FIELDS = ("context", "context_time_mask", "context_window_mask", "context_window_weights")


def encode_conditions(model: Any, batch: dict[str, torch.Tensor]):
    return model.encode_conditions(
        batch["anchor"], batch["numeric"], batch["sex"], batch["field_mask"], batch["age_topcoded"],
        context=batch["context"], context_time_mask=batch["context_time_mask"],
        context_window_mask=batch["context_window_mask"],
        context_window_weights=batch["context_window_weights"])


def flow_loss(model: Any, batch: dict[str, torch.Tensor], weights: dict[str, Any],
              d12_scale: torch.Tensor, noise: torch.Tensor | None = None,
              time: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    validate_mainline_loss(weights)
    target, mask = batch["target"], batch["quality_mask"].bool()
    if target.ndim != 3 or target.shape[1] != 12 or mask.shape != target.shape[:2]:
        raise ValueError("Training expects [B,12,T] target and [B,12] quality mask")
    if torch.any(mask[:, 1:].sum(dim=1) == 0):
        raise ValueError("Each item needs at least one valid missing lead")
    missing = target[:, 1:]
    noise = torch.randn_like(missing) if noise is None else noise
    time = torch.rand(len(target), device=target.device) if time is None else time
    state, true_velocity = linear_path(missing, mask[:, 1:], noise, time)
    condition = encode_conditions(model, batch)
    predicted = model.velocity(state, time, condition)
    velocity_weight = mask[:, 1:].unsqueeze(-1)
    fm = torch.where(velocity_weight, (predicted - true_velocity).square(), 0.).sum()
    fm = fm / (mask[:, 1:].sum().clamp_min(1) * target.shape[-1])
    endpoint = state + (1 - time[:, None, None]) * predicted
    full = torch.cat((condition.anchor_prediction, endpoint), dim=1)
    safe_target = torch.where(mask.unsqueeze(-1), target, torch.zeros_like(target))
    safe_full = torch.where(mask.unsqueeze(-1), full, torch.zeros_like(full))
    huber = masked_huber_loss(safe_full, safe_target, mask, float(weights["huber_delta"]))
    pcc = masked_pcc_loss(safe_full, safe_target, mask)
    anchor = masked_huber_loss(condition.anchor_prediction, batch["anchor"],
                               torch.ones_like(mask[:, :1]), float(weights["huber_delta"]))
    physiology = full.new_zeros(())
    if float(weights["physiology"]):
        reliable = mask[:, :6].all(dim=1)
        if reliable.any():
            physiology = physiology_constraint_loss(full[reliable], d12_scale)
    parts = {"fm": fm, "huber": huber, "pcc": pcc, "anchor": anchor, "physiology": physiology}
    total = fm + sum(float(weights[key]) * parts[key]
                     for key in ("huber", "pcc", "anchor", "physiology"))
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite context-d6 loss")
    return total, parts


@torch.no_grad()
def sample(model: Any, batch: dict[str, torch.Tensor], keys: list[str], seed: int,
           steps: int, solver: str, samples: int) -> torch.Tensor:
    if model.training or samples < 1:
        raise ValueError("Sampling requires eval mode and positive sample count")
    condition = encode_conditions(model, batch)
    anchor = batch["anchor"]
    total = anchor.new_zeros(len(anchor), 11, anchor.shape[-1])
    for index in range(samples):
        noise = keyed_noise(keys, anchor.shape[-1], seed, index, anchor.device, anchor.dtype)
        total += integrate(model, condition, noise, steps, solver)
    return torch.cat((condition.anchor_prediction, total / samples), dim=1)

