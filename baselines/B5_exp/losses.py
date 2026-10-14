"""Training-only C/M/F losses. Targets never enter the predictor interface."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .core import objective

LOSS_PROTOCOL = 'B5-slow-loss-pair-v1'


def smooth_native(x, width=51):
    """Centered edge-repeat average on full 120s at 50Hz; deterministic backward."""
    if width < 3 or width % 2 != 1 or width > x.shape[-1]:
        raise ValueError('Residual width must be odd, >=3 and <= record length')
    r = width // 2
    extended = torch.cat((x[..., :1].expand(*x.shape[:-1], r), x,
                          x[..., -1:].expand(*x.shape[:-1], r)), dim=-1)
    return F.avg_pool1d(extended, width, stride=1)


def auxiliary_terms(prediction, target, mask, width=51):
    if prediction.shape != target.shape or mask.shape != prediction.shape[:-1]:
        raise ValueError('Expected prediction/target [B,11,T] and mask [B,11]')
    valid = mask.bool()[..., None]
    # Mask BEFORE averaging/filtering, so flagged NaN labels cannot leak gradients.
    p = torch.where(valid, prediction, 0.)
    y = torch.where(valid, target, 0.)
    denominator = mask.sum().clamp_min(1)
    mean = F.huber_loss(p.mean(-1), y.mean(-1), reduction='sum') / denominator
    hp, hy = p - smooth_native(p, width), y - smooth_native(y, width)
    residual = F.huber_loss(hp, hy, reduction='sum') / (denominator * y.shape[-1])
    return mean, residual


def paired_objective(prediction, target, mask, group, weights, width=51):
    if group not in ('C', 'M', 'F'):
        raise ValueError('Expected C, M or F')
    base, parts = objective(prediction, target, mask)
    mean, residual = auxiliary_terms(prediction, target, mask, width)
    mean_weight = float(weights['mean']) if group == 'M' else 0.
    residual_weight = float(weights['residual']) if group == 'F' else 0.
    if min(mean_weight, residual_weight) < 0:
        raise ValueError('Loss weights must be nonnegative')
    total = base + mean_weight * mean + residual_weight * residual
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite paired loss')
    return total, {**parts, 'mean': float(mean.detach()), 'residual': float(residual.detach()),
                   'weighted_mean': float((mean_weight * mean).detach()),
                   'weighted_residual': float((residual_weight * residual).detach())}
