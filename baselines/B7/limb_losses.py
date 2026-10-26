"""Differentiable physical-uV endpoint losses; no additional generated channels."""
import torch
from . import GENERATED_INDICES
from .leads import compose_d12


def physical_losses(endpoint, anchor, batch, time_mask, config):
    scales = batch["scales_uV"].to(endpoint)
    if scales.ndim == 1:
        scales = scales[None].expand(len(endpoint), -1)
    qc = batch["lead_mask_d12"].bool()
    target = batch["target_d12_uV"].to(endpoint)
    if target.shape != (len(endpoint), 12, endpoint.shape[-1]) or scales.shape != (len(endpoint), 12):
        raise ValueError("Physical losses require target[B,12,T] and scales[B,12]")
    if qc.shape != target.shape[:2] or time_mask.shape != (len(endpoint), endpoint.shape[-1]):
        raise ValueError("Physical loss mask shape mismatch")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid physical scales")
    if not torch.equal(qc[:, GENERATED_INDICES], batch["lead_mask"].bool()):
        raise ValueError("Physical and generated QC disagree")
    # Invalid labels may contain NaN: mask BEFORE any arithmetic.
    valid = qc[:, :, None] & time_mask[:, None, :]
    y = torch.where(valid, target, 0.)
    if not torch.isfinite(y).all():
        raise ValueError("Nonfinite eligible physical targets")
    a = anchor * scales[:, :1, None]
    p = compose_d12(a, endpoint * scales[:, GENERATED_INDICES, None])
    zero = endpoint.sum() * 0.
    result = {}
    if config.limb_wave_weight:
        mask = valid[:, [2, 5]]
        truth = y[:, [2, 5]]
        count = mask.sum(-1)
        n = count.clamp_min(1)
        mean = truth.sum(-1) / n
        centered = torch.where(mask, truth - mean[:, :, None], 0.)
        std = torch.sqrt(centered.square().sum(-1) / n)
        eligible = (count >= 2) & (std > 0)
        denominator = std.clamp_min(config.limb_std_floor_uV)
        # Full-waveform MAE: unlike PCC, retains amplitude and DC error.
        error = torch.where(mask, (p[:, [2, 5]] - truth).abs(), 0.).sum(-1) / n
        values = error / denominator
        result["limb_wave"] = torch.where(eligible, values, 0.).sum() / eligible.sum().clamp_min(1)
        result["limb_wave_valid"] = eligible.sum().to(endpoint.dtype)
    if config.projection_weight:
        # A common time mask is essential for the projection and its normalization.
        mask = time_mask & qc[:, 0, None] & qc[:, 1, None]
        count = mask.sum(-1)
        n = count.clamp_min(1)
        av = torch.where(mask, a[:, 0], 0.)
        pv = torch.where(mask, p[:, 1], 0.)
        yv = torch.where(mask, y[:, 1], 0.)
        ac = torch.where(mask, av - av.sum(-1)[:, None] / n[:, None], 0.)
        pc = torch.where(mask, pv - pv.sum(-1)[:, None] / n[:, None], 0.)
        yc = torch.where(mask, yv - yv.sum(-1)[:, None] / n[:, None], 0.)
        aa = ac.square().sum(-1)
        ast = torch.sqrt(aa / n)
        yst = torch.sqrt(yc.square().sum(-1) / n)
        eligible = (count >= 2) & (ast >= config.anchor_std_floor_uV) & (yst > 0)
        delta = ((pc - yc) * ac).sum(-1) / aa.clamp_min(1e-12)
        # Error in the I-directed component, relative to true II amplitude.
        normalized = delta.abs() * ast / yst.clamp_min(config.limb_std_floor_uV)
        result["projection"] = torch.where(eligible, normalized, 0.).sum() / eligible.sum().clamp_min(1)
        result["projection_valid"] = eligible.sum().to(endpoint.dtype)
    return {k: v + zero if k in ("limb_wave", "projection") else v for k, v in result.items()}
