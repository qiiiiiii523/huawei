"""Differentiable lead recovery, always in physical microvolts."""
from __future__ import annotations

import torch
from . import GENERATED_INDICES


def generated_from_d12(target):
    if target.ndim != 3 or target.shape[1] != 12:
        raise ValueError("Expected [B,12,T]")
    return target[:, GENERATED_INDICES, :]


def compose_d12(anchor_uV, generated_uV):
    if generated_uV.ndim != 3 or generated_uV.shape[1] != 7:
        raise ValueError("Expected seven generated leads")
    if anchor_uV.shape != (len(generated_uV), 1, generated_uV.shape[-1]):
        raise ValueError("Anchor shape mismatch")
    ii = generated_uV[:, :1]
    return torch.cat((anchor_uV, ii, ii - anchor_uV, -(anchor_uV + ii) / 2,
                      anchor_uV - ii / 2, ii - anchor_uV / 2, generated_uV[:, 1:]), dim=1)


def validate_scales(scales):
    value = torch.as_tensor(scales, dtype=torch.float32)
    if value.shape != (12,) or not torch.isfinite(value).all() or (value <= 0).any():
        raise ValueError("Twelve positive finite train-only scales required")
    return value
