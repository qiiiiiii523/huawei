"""Full-window FM/MAE and central PCC, weighted on the seven unknown leads."""
from __future__ import annotations

import torch
from .config import LossConfig
from .flow import linear_path


def core_slice(length, sample_rate=500):
    if sample_rate != 500 or length not in (5000, 20000):
        raise ValueError("Training uses real 10s or 40s at 500Hz")
    return slice(0, 5000) if length == 5000 else slice(7500, 12500)


def reconstruction_losses(prediction, target, valid, config=None):
    config = config or LossConfig()
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 7 or valid.shape != target.shape:
        raise ValueError("Matching [B,7,T] tensors and point mask required")
    if not valid.any():
        raise ValueError("No supervised positions")
    weights = prediction.new_tensor((config.ii_weight, 1, 1, 1, 1, 1, 1))[None, :, None]
    p = torch.where(valid, prediction, torch.zeros_like(prediction))
    y = torch.where(valid, target, torch.zeros_like(target))
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError("Nonfinite supervised waveform")
    absolute = torch.where(valid, (p - y).abs(), 0.)
    mae = (absolute * weights).sum() / (valid * weights).sum()
    core = core_slice(target.shape[-1])
    mask = valid[:, :, core]
    n = mask.sum(-1).clamp_min(1)
    pc, yc = p[:, :, core], y[:, :, core]
    pm, ym = pc.sum(-1) / n, yc.sum(-1) / n
    pc = torch.where(mask, pc - pm[:, :, None], 0.)
    yc = torch.where(mask, yc - ym[:, :, None], 0.)
    py, yy = pc.square().sum(-1), yc.square().sum(-1)
    eligible = (mask.sum(-1) >= 2) & (yy / n > config.target_variance_floor)
    denominator = torch.sqrt((py * yy).clamp_min(1e-12))
    corr = ((pc * yc).sum(-1) / denominator).clamp(-1, 1)
    lead_weights = weights.squeeze(-1) * eligible
    pcc = ((1 - corr) * lead_weights).sum() / lead_weights.sum().clamp_min(1)
    # When there is no varying target in the core, PCC contributes zero.
    counts = valid.sum((0, 2)).clamp_min(1)
    per_lead_mae = absolute.sum((0, 2)) / counts
    return {"mae": mae, "pcc": pcc, "pcc_valid_leads": eligible.sum().to(prediction.dtype),
            "mae_ii": per_lead_mae[0], "mae_chest": per_lead_mae[1:].mean()}


def flow_loss(model, batch, config=None, *, noise=None, time=None):
    config = config or LossConfig()
    target, anchor = batch["target"], batch["anchor"]
    if target.ndim != 3 or target.shape[1] != 7:
        raise ValueError("B7 target contains II,V1..V6 only")
    core_slice(target.shape[-1])
    mask = batch["lead_mask"].bool()
    tm = batch.get("time_mask", torch.ones(len(target), target.shape[-1], dtype=torch.bool, device=target.device)).bool()
    if mask.shape != target.shape[:2] or tm.shape != (len(target), target.shape[-1]):
        raise ValueError("Mask shape mismatch")
    valid = mask[:, :, None] & tm[:, None, :]
    if not valid.flatten(1).any(1).all():
        raise ValueError("Each training item needs valid supervision")
    noise = torch.randn_like(target) if noise is None else noise
    time = torch.rand(len(target), device=target.device) if time is None else time
    state, velocity = linear_path(target, valid, noise, time)
    predicted = model(state, time, anchor, batch.get("metadata"))
    if predicted.shape != target.shape:
        raise ValueError("Velocity output must be [B,7,T]")
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("Nonfinite model velocity")
    weights = target.new_tensor((config.ii_weight, 1, 1, 1, 1, 1, 1))[None, :, None]
    error = torch.where(valid, (predicted - velocity).square(), 0.)
    fm = (error * weights).sum() / (valid * weights).sum()
    endpoint = state + (1 - time[:, None, None]) * predicted
    parts = reconstruction_losses(endpoint, target, valid, config)
    parts["fm"] = fm
    total = fm + config.mae_weight * parts["mae"] + config.pcc_weight * parts["pcc"]
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite B7 loss")
    return total, parts
