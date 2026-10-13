"""B5 with masked, independent-record d6 context conditioning."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet, ConditionCache


@dataclass
class ContextConditionCache(ConditionCache):
    context_embedding: torch.Tensor
    context_available: torch.Tensor


class MachineContextEncoder(nn.Module):
    """Encode each history window, then pool windows using valid-sample weights."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(6, 32, 15, stride=10, padding=7), nn.SiLU(),
            nn.Conv1d(32, 64, 9, stride=5, padding=4), nn.SiLU(),
            nn.Conv1d(64, output_dim, 7, stride=5, padding=3), nn.SiLU(),
        )
        self.output_dim = output_dim

    def forward(self, context: torch.Tensor, time_mask: torch.Tensor,
                window_mask: torch.Tensor, window_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 4 or context.shape[2] != 6 or context.shape[-1] != 5000:
            raise ValueError("context must be [B,W,6,5000]")
        if time_mask.shape != context.shape or window_mask.shape != context.shape[:2]:
            raise ValueError("context masks disagree with context shape")
        if window_weights.shape != context.shape[:2]:
            raise ValueError("context window weights must be [B,W]")
        batch, windows = context.shape[:2]
        valid = time_mask.bool() & window_mask[:, :, None, None].bool()
        safe = torch.where(valid, context, torch.zeros_like(context))
        flat_valid = valid.any(dim=2).reshape(batch * windows, 1, 5000).to(context.dtype)
        feature_map = self.network(safe.reshape(batch * windows, 6, 5000))
        # Weight the downsampled feature map by real support, so gaps and tail
        # padding never contribute merely because they are represented by zero.
        support = F.adaptive_avg_pool1d(flat_valid, feature_map.shape[-1])
        features = (feature_map * support).sum(dim=-1) / support.sum(dim=-1).clamp_min(1.0)
        features = features.reshape(batch, windows, self.output_dim)
        weights = torch.where(window_mask.bool(), window_weights, torch.zeros_like(window_weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        available = valid.any(dim=(1, 2, 3))
        pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
        pooled = torch.where(available[:, None], pooled, torch.zeros_like(pooled))
        return pooled, available


class ContextD6UNet(B5UNet):
    """Preserve every B5 module and add zero-initialized multiscale context residuals."""

    def __init__(self, config: ModelConfig = ModelConfig(), context_dim: int = 128) -> None:
        super().__init__(config)
        self.context_encoder = MachineContextEncoder(context_dim)
        self.context_projection = nn.ModuleList(
            [nn.Linear(context_dim, channels * 2) for channels in config.channels]
        )
        for layer in self.context_projection:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _modulate(features: torch.Tensor, projection: nn.Linear,
                  embedding: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        gamma, beta = projection(embedding).chunk(2, dim=-1)
        gate = available.to(features.dtype).unsqueeze(-1)
        gamma, beta = gamma * gate, beta * gate
        return features * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

    def encode_conditions(self, anchor: torch.Tensor, numeric: torch.Tensor, sex: torch.Tensor,
                          field_mask: torch.Tensor, age_topcoded: torch.Tensor, *,
                          context: torch.Tensor | None = None,
                          context_time_mask: torch.Tensor | None = None,
                          context_window_mask: torch.Tensor | None = None,
                          context_window_weights: torch.Tensor | None = None) -> ContextConditionCache:
        base = super().encode_conditions(anchor, numeric, sex, field_mask, age_topcoded)
        if context is None:
            embedding = anchor.new_zeros(anchor.shape[0], self.context_encoder.output_dim)
            available = torch.zeros(anchor.shape[0], dtype=torch.bool, device=anchor.device)
        else:
            if context_time_mask is None or context_window_mask is None or context_window_weights is None:
                raise ValueError("context input requires time, window, and weight masks")
            embedding, available = self.context_encoder(
                context, context_time_mask, context_window_mask, context_window_weights)
        features = tuple(self._modulate(feature, projection, embedding, available)
                         for feature, projection in zip(base.anchor_features, self.context_projection))
        return ContextConditionCache(features, base.metadata, base.anchor_prediction, embedding, available)

    def forward(self, state: torch.Tensor, time: torch.Tensor, anchor: torch.Tensor,
                numeric: torch.Tensor, sex: torch.Tensor, field_mask: torch.Tensor,
                age_topcoded: torch.Tensor, *, context: torch.Tensor | None = None,
                context_time_mask: torch.Tensor | None = None,
                context_window_mask: torch.Tensor | None = None,
                context_window_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.encode_conditions(
            anchor, numeric, sex, field_mask, age_topcoded, context=context,
            context_time_mask=context_time_mask, context_window_mask=context_window_mask,
            context_window_weights=context_window_weights)
        return self.velocity(state, time, condition), condition.anchor_prediction


def load_b5_weights(model: ContextD6UNet, state: dict[str, torch.Tensor]) -> None:
    """Load B5 strictly for inherited modules; only new context modules may be absent."""
    incompatible = model.load_state_dict(state, strict=False)
    expected = {name for name in model.state_dict()
                if name.startswith("context_encoder.") or name.startswith("context_projection.")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError(
            f"B5 initialization mismatch; missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}")

