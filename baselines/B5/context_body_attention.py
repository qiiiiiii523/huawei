"""Local-attention body context encoder pilot.

The B5 backbone, metadata path, bottleneck FiLM, masks, and record-level
weighted mean/std pooling contract stay unchanged. Only within-window temporal
pooling changes from a uniform average to a learned masked attention pool.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .context_body import B5BodyContext, BodyContextEncoder


class BodyContextEncoderAttention(BodyContextEncoder):
    """Encode salient local waveform segments before record-level pooling."""

    def __init__(self, dimension: int = 128) -> None:
        super().__init__(dimension)
        self.attention = nn.Conv1d(64, 1, kernel_size=1)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        self.fuse = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, context: torch.Tensor, time_mask: torch.Tensor,
                window_weights: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        if context.ndim != 4 or context.shape[2] != 6 or context.shape[-1] < 8:
            raise ValueError("Body context must be [B,W,6,T] with T>=8")
        b, w, _, length = context.shape
        if time_mask.shape != context.shape or window_weights.shape != (b, w) or available.shape != (b,):
            raise ValueError("Body context mask/weight/availability shape mismatch")
        mask = time_mask.bool()
        valid = torch.where(mask, context, torch.zeros_like(context))
        if not torch.isfinite(valid).all():
            raise ValueError("Observed context values must be finite")
        counts = mask.sum(dim=(2, 3)).to(context.dtype)
        expected = counts / counts.sum(dim=1, keepdim=True).clamp_min(1)
        if not torch.isfinite(window_weights).all() or not torch.allclose(window_weights, expected, atol=1e-5):
            raise ValueError("Context window weights must match valid-point counts")
        if not torch.equal(available.bool(), counts.sum(dim=1) > 0):
            raise ValueError("Context availability disagrees with valid-point mask")

        features = self.cnn(torch.cat((valid, mask.to(context.dtype)), dim=2)
                            .reshape(b * w, 12, length))
        support = F.adaptive_avg_pool1d(
            mask.reshape(b * w, 6, length).any(dim=1, keepdim=True).float(),
            features.shape[-1])
        logits = self.attention(features).squeeze(1)
        # A zero-initialized attention map reproduces the previous support-
        # weighted temporal mean exactly, giving a function-preserving start.
        logits = logits + torch.log(support.squeeze(1).clamp_min(1e-8))
        valid_positions = support.squeeze(1) > 0
        logits = logits.masked_fill(~valid_positions, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1) * valid_positions.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = (features * weights.unsqueeze(1)).sum(dim=-1)
        windows = self.project(pooled.reshape(b, w, -1))

        mean = (windows * expected.unsqueeze(-1)).sum(dim=1)
        variance = ((windows - mean.unsqueeze(1)).square()
                    * expected.unsqueeze(-1)).sum(dim=1)
        std = torch.sqrt(variance.clamp_min(0) + 1e-8)
        return mean + self.fuse(torch.cat((mean, std), dim=-1))


class B5BodyContextAttention(B5BodyContext):
    """B5 with only its body-context encoder replaced by local attention."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__(config)
        self.body_encoder = BodyContextEncoderAttention(config.metadata_dim)
