"""Differentiable zero-DC DCT low-pass supervision at the original 500Hz.

No resampling, scipy filtering, label substitution or extra ODE rollout occurs.
The cosine basis uses reflected boundary conditions. A hard DCT cutoff is an
experimental loss definition, not the Butterworth oracle used in diagnostics.
"""
from __future__ import annotations

from functools import lru_cache
import math

import torch


@lru_cache(maxsize=16)
def cosine_basis(length, cutoff_hz, device, dtype):
    # DCT-II frequency k is k*fs/(2*T). Exclude DC (k=0).
    count = min(length - 1, math.floor(2 * length * cutoff_hz / 500 + 1e-10))
    positions = torch.arange(length, device=device, dtype=torch.float64) + 0.5
    modes = torch.arange(1, count + 1, device=device, dtype=torch.float64)
    return (torch.cos(math.pi / length * modes[:, None] * positions) * math.sqrt(2 / length)).to(dtype)


def project_slow(error, cutoff_hz):
    """Project a contiguous valid segment, without DC, retaining its full length."""
    basis = cosine_basis(error.shape[-1], cutoff_hz, str(error.device), error.dtype)
    centered = error - error.mean(-1, keepdim=True)
    return (centered @ basis.T) @ basis


def low_frequency_loss(prediction, target, valid, config):
    if prediction.shape != target.shape or valid.shape != target.shape or target.shape[1] != 7:
        raise ValueError("Low frequency supervision needs matching [B,7,T] waveforms and mask")
    # Mask BEFORE arithmetic, so excluded NaN/extreme labels cannot contaminate
    # valid projections or gradients. Never interpolate/filter across invalid gaps.
    error = torch.where(valid, prediction, 0.) - torch.where(valid, target, 0.)
    if not torch.isfinite(error).all():
        raise ValueError("Nonfinite low frequency supervision")
    weights = prediction.new_tensor((config.ii_weight, 1, 1, 1, 1, 1, 1)).expand(len(error), -1).reshape(-1)
    rows, masks = error.reshape(-1, error.shape[-1]), valid.reshape(-1, error.shape[-1])
    full = masks.all(-1)
    total = error.sum() * 0.
    denominator = prediction.new_zeros(())
    points = prediction.new_zeros(())
    if full.any():
        projected = project_slow(rows[full], config.low_frequency_cutoff_hz)
        total = total + (projected.abs().sum(-1) * weights[full]).sum()
        denominator = denominator + weights[full].sum() * error.shape[-1]
        points = points + full.sum() * error.shape[-1]
    # QC-v2 currently masks whole leads. This fallback also supports future time
    # masks: each contiguous valid segment gets its own physical-frequency basis.
    partial = (~full) & masks.any(-1)
    for index in partial.nonzero().flatten().tolist():
        m = masks[index]
        transitions = torch.diff(torch.cat((m.new_zeros(1), m, m.new_zeros(1))).to(torch.int8))
        starts = (transitions == 1).nonzero().flatten().tolist()
        stops = (transitions == -1).nonzero().flatten().tolist()
        for start, stop in zip(starts, stops):
            if 2 * (stop - start) * config.low_frequency_cutoff_hz < 500:
                continue  # no non-DC mode below cutoff on such a short segment
            projected = project_slow(rows[index, start:stop], config.low_frequency_cutoff_hz)
            total = total + projected.abs().sum() * weights[index]
            denominator = denominator + (stop - start) * weights[index]
            points = points + stop - start
    return {"low_frequency_mae": total / denominator.clamp_min(1),
            "low_frequency_valid_points": points}
