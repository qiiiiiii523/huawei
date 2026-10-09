"""Masked FM objective and v3-compatible auxiliary reconstruction losses."""
from __future__ import annotations

import torch

from ecg12gen.losses import masked_huber_loss, masked_pcc_loss, physiology_constraint_loss
from .flow import linear_path
from .model import B5UNet


def flow_loss(model: B5UNet, batch: dict[str, torch.Tensor], weights: dict[str, float],
              d12_scale: torch.Tensor, noise: torch.Tensor | None = None,
              time: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = batch["target"]
    mask = batch["quality_mask"].bool()
    if target.ndim != 3 or target.shape[1] != 12 or mask.shape != target.shape[:2]:
        raise ValueError("Training expects [B,12,T] target and [B,12] quality mask")
    if torch.any(mask[:, 1:].sum(dim=1) == 0):
        raise ValueError("Each training item needs at least one valid missing lead")
    missing = target[:, 1:]
    noise = torch.randn_like(missing) if noise is None else noise
    time = torch.rand(len(target), device=target.device) if time is None else time
    state, true_velocity = linear_path(missing, mask[:, 1:], noise, time)
    predicted, anchor_prediction = model(state, time, batch["anchor"], batch["numeric"], batch["sex"],
                                         batch["field_mask"], batch["age_topcoded"])
    velocity_weight = mask[:, 1:].unsqueeze(-1)
    fm = torch.where(velocity_weight, (predicted - true_velocity).square(), 0.).sum()
    fm = fm / (mask[:, 1:].sum().clamp_min(1) * target.shape[-1])
    endpoint = state + (1 - time[:, None, None]) * predicted
    full = torch.cat((anchor_prediction, endpoint), dim=1)
    safe_target = torch.where(mask.unsqueeze(-1), target, torch.zeros_like(target))
    # A zeroed invalid prediction avoids NaN*0 in public loss primitives.
    safe_full = torch.where(mask.unsqueeze(-1), full, torch.zeros_like(full))
    huber = masked_huber_loss(safe_full, safe_target, mask, float(weights["huber_delta"]))
    pcc = masked_pcc_loss(safe_full, safe_target, mask)
    anchor = masked_huber_loss(anchor_prediction, batch["anchor"], torch.ones_like(mask[:, :1]),
                               float(weights["huber_delta"]))
    physiology = full.new_zeros(())
    if float(weights["physiology"]):
        # Only items with all limb leads supervised can use this auxiliary constraint.
        reliable = mask[:, :6].all(dim=1)
        if reliable.any():
            physiology = physiology_constraint_loss(full[reliable], d12_scale)
    parts = {"fm": fm, "huber": huber, "pcc": pcc, "anchor": anchor, "physiology": physiology}
    total = fm + sum(float(weights[key]) * parts[key] for key in ("huber", "pcc", "anchor", "physiology"))
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite B5 loss")
    return total, parts
