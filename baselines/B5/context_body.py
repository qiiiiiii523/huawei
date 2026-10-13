"""Body-scale, independent-record context adapter for B5.

One entire historical d6 record conditions each target window. No context time
index is aligned to an ECG target index, and no context waveform is added to the
generated leads.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .conditions import FiLM
from .config import ModelConfig
from .flow import integrate, keyed_noise
from .model import B5UNet, ConditionCache


class BodyContextEncoder(nn.Module):
    """Masked per-window CNN followed by permutation-invariant weighted pooling."""

    def __init__(self, dimension: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 32, 7, stride=2, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.SiLU(),
        )
        self.project = nn.Sequential(nn.Linear(64, dimension), nn.SiLU(), nn.Linear(dimension, dimension))

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
        # The validity channels let the CNN distinguish true zero voltage from padding.
        features = self.cnn(torch.cat((valid, mask.to(context.dtype)), dim=2).reshape(b * w, 12, length))
        # Pool only positions supported by observed input; padded tails have zero mass.
        support = F.adaptive_avg_pool1d(mask.reshape(b * w, 6, length).any(dim=1, keepdim=True).float(),
                                         features.shape[-1])
        pooled = (features * support).sum(dim=-1) / support.sum(dim=-1).clamp_min(1)
        windows = self.project(pooled.reshape(b, w, -1))
        return (windows * expected.unsqueeze(-1)).sum(dim=1)


@dataclass
class BodyConditionCache(ConditionCache):
    body_embedding: torch.Tensor
    context_available: torch.Tensor


class B5BodyContext(B5UNet):
    """B5 with an initially neutral, explicitly gated body-context FiLM path."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__(config)
        self.body_encoder = BodyContextEncoder(config.metadata_dim)
        self.body_film = FiLM(config.metadata_dim, config.channels[-1])

    def encode_body_conditions(self, anchor: torch.Tensor, numeric: torch.Tensor, sex: torch.Tensor,
                               field_mask: torch.Tensor, age_topcoded: torch.Tensor,
                               context: torch.Tensor, context_time_mask: torch.Tensor,
                               context_window_weights: torch.Tensor,
                               context_available: torch.Tensor) -> BodyConditionCache:
        base = self.encode_conditions(anchor, numeric, sex, field_mask, age_topcoded)
        if context.shape[0] != anchor.shape[0]:
            raise ValueError("Anchor and context batch mismatch")
        embedding = self.body_encoder(context, context_time_mask, context_window_weights, context_available)
        return BodyConditionCache(base.anchor_features, base.metadata, base.anchor_prediction,
                                  embedding, context_available.bool())

    def _condition_bottleneck(self, features: torch.Tensor, condition: ConditionCache) -> torch.Tensor:
        if not isinstance(condition, BodyConditionCache):
            return features
        gamma, beta = self.body_film.projection(condition.body_embedding).chunk(2, dim=-1)
        gate = condition.context_available.to(features.dtype).unsqueeze(-1)
        return features * (1 + (gamma * gate).unsqueeze(-1)) + (beta * gate).unsqueeze(-1)

    def forward_body(self, state: torch.Tensor, time: torch.Tensor, anchor: torch.Tensor,
                     numeric: torch.Tensor, sex: torch.Tensor, field_mask: torch.Tensor,
                     age_topcoded: torch.Tensor, context: torch.Tensor,
                     context_time_mask: torch.Tensor, context_window_weights: torch.Tensor,
                     context_available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.encode_body_conditions(anchor, numeric, sex, field_mask, age_topcoded,
                                                context, context_time_mask, context_window_weights,
                                                context_available)
        return self.velocity(state, time, condition), condition.anchor_prediction


@torch.no_grad()
def sample_body(model: B5BodyContext, inputs: dict[str, torch.Tensor], keys: list[str],
                seed: int = 42, steps: int = 16, solver: str = "heun", samples: int = 1) -> torch.Tensor:
    """Target-free d12 sampling with cached independent-record body context."""
    if model.training or samples < 1 or len(keys) != len(inputs["anchor"]):
        raise ValueError("Sampling requires eval mode, positive samples, and matching keys")
    anchor = inputs["anchor"]
    condition = model.encode_body_conditions(anchor, inputs["numeric"], inputs["sex"],
        inputs["field_mask"], inputs["age_topcoded"], inputs["context"],
        inputs["context_time_mask"], inputs["context_window_weights"], inputs["context_available"])
    total = torch.zeros(len(anchor), 11, anchor.shape[-1], device=anchor.device, dtype=anchor.dtype)
    for index in range(samples):
        noise = keyed_noise(keys, anchor.shape[-1], seed, index, anchor.device, anchor.dtype)
        total += integrate(model, condition, noise, steps, solver)
    return torch.cat((condition.anchor_prediction, total / samples), dim=1)
