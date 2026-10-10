"""Training-only centered slow-trend supervision; no inference dependency."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def centered_slow_curve(signal: torch.Tensor, width: int) -> torch.Tensor:
    if signal.ndim != 3 or width < 3 or width % 2 != 1 or width > signal.shape[-1]:
        raise ValueError('Expected [B,C,T] and an odd pooling width fitting T')
    # FP32 pooling avoids half-precision accumulation. Center before/after pooling
    # for numerical stability and exact constant-offset invariance in the definition.
    signal = signal.float()
    centered = signal - signal.mean(dim=-1, keepdim=True)
    radius = width // 2
    # Explicit edge repeats avoid ReplicationPad CUDA backward nondeterminism.
    padded = torch.cat((centered[..., :1].expand(-1, -1, radius), centered,
                        centered[..., -1:].expand(-1, -1, radius)), dim=-1)
    slow = F.avg_pool1d(padded, kernel_size=width, stride=1)
    return slow - slow.mean(dim=-1, keepdim=True)


def masked_slow_trend_loss(prediction: torch.Tensor, target: torch.Tensor,
                           quality_mask: torch.Tensor, width: int = 501,
                           delta: float = 1., leads: str = 'chest6') -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 11:
        raise ValueError('Slow supervision expects missing II--V6 [B,11,T]')
    if quality_mask.shape != prediction.shape[:2]:
        raise ValueError('Slow supervision mask must be [B,11]')
    if delta <= 0 or leads not in ('chest6', 'missing11'):
        raise ValueError('Invalid slow supervision settings')
    mask = quality_mask.bool().clone()
    if leads == 'chest6':
        mask[:, :5] = False  # Missing lead indices 5..10 are canonical V1..V6.
    valid = mask.unsqueeze(-1)
    # Sanitize invalid targets BEFORE temporal pooling, not NaN*0 after the loss.
    p = torch.where(valid, prediction, torch.zeros_like(prediction))
    y = torch.where(valid, target, torch.zeros_like(target))
    ps, ys = centered_slow_curve(p, width), centered_slow_curve(y, width)
    error = F.huber_loss(ps, ys, reduction='none', delta=delta)
    return torch.where(valid, error, torch.zeros_like(error)).sum() / (mask.sum().clamp_min(1) * target.shape[-1])
