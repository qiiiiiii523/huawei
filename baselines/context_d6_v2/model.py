"""B5 plus multiscale context encoding and mask-aware attentive pooling."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet, ConditionCache


@dataclass
class ContextV2ConditionCache(ConditionCache):
    context_embedding: torch.Tensor
    context_available: torch.Tensor
    context_attention: torch.Tensor


class MultiScaleWindowEncoder(nn.Module):
    """Encode local ECG morphology at three temporal receptive-field scales."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        branch_dim = 32
        self.stem = nn.Sequential(
            nn.Conv1d(6, branch_dim, 15, stride=10, padding=7),
            nn.GroupNorm(8, branch_dim),
            nn.SiLU(),
        )
        self.branches = nn.ModuleList([
            nn.Conv1d(branch_dim, branch_dim, 5, padding=2, dilation=1),
            nn.Conv1d(branch_dim, branch_dim, 9, padding=8, dilation=2),
            nn.Conv1d(branch_dim, branch_dim, 15, padding=56, dilation=8),
        ])
        self.projection = nn.Sequential(
            nn.GroupNorm(8, branch_dim * len(self.branches)),
            nn.SiLU(),
            nn.Conv1d(branch_dim * len(self.branches), output_dim, 1),
            nn.SiLU(),
        )

    def forward(self, signal: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        features = self.stem(signal)
        features = torch.cat([F.silu(branch(features)) for branch in self.branches], dim=1)
        features = self.projection(features)
        downsampled_support = F.adaptive_avg_pool1d(support, features.shape[-1])
        return (features * downsampled_support).sum(dim=-1) / downsampled_support.sum(dim=-1).clamp_min(1.0)


class AttentiveContextEncoder(nn.Module):
    """Pool a masked set of history windows using attention plus robust statistics."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.window_encoder = MultiScaleWindowEncoder(output_dim)
        hidden = max(32, output_dim // 2)
        self.attention_value = nn.Linear(output_dim, hidden)
        self.attention_gate = nn.Linear(output_dim, hidden)
        self.attention_score = nn.Linear(hidden, 1)
        nn.init.zeros_(self.attention_score.weight)
        nn.init.zeros_(self.attention_score.bias)
        self.fusion = nn.Sequential(
            nn.LayerNorm(output_dim * 4),
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, context: torch.Tensor, time_mask: torch.Tensor,
                window_mask: torch.Tensor, window_weights: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.ndim != 4 or context.shape[2:] != (6, 5000):
            raise ValueError("context must be [B,W,6,5000]")
        if time_mask.shape != context.shape or window_mask.shape != context.shape[:2]:
            raise ValueError("context masks disagree with context shape")
        if window_weights.shape != context.shape[:2]:
            raise ValueError("context window weights must be [B,W]")
        batch, windows = context.shape[:2]
        valid = time_mask.bool() & window_mask[:, :, None, None].bool()
        safe = torch.where(valid, context, torch.zeros_like(context))
        support = valid.any(dim=2).reshape(batch * windows, 1, 5000).to(context.dtype)
        tokens = self.window_encoder(safe.reshape(batch * windows, 6, 5000), support)
        tokens = tokens.reshape(batch, windows, self.output_dim)

        valid_windows = window_mask.bool() & valid.any(dim=(2, 3)) & (window_weights > 0)
        available = valid_windows.any(dim=1)
        scores = self.attention_score(
            torch.tanh(self.attention_value(tokens)) * torch.sigmoid(self.attention_gate(tokens))
        ).squeeze(-1)
        scores = scores + window_weights.clamp_min(1e-8).log()
        scores = scores.masked_fill(~valid_windows, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        attention = torch.where(valid_windows, attention, torch.zeros_like(attention))
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attended = (tokens * attention.unsqueeze(-1)).sum(dim=1)

        weights = torch.where(valid_windows, window_weights, torch.zeros_like(window_weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mean = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        variance = ((tokens - mean[:, None]) ** 2 * weights.unsqueeze(-1)).sum(dim=1)
        std = (variance.clamp_min(0) + 1e-6).sqrt()
        maximum = tokens.masked_fill(~valid_windows.unsqueeze(-1),
                                     torch.finfo(tokens.dtype).min).amax(dim=1)
        maximum = torch.where(available[:, None], maximum, torch.zeros_like(maximum))
        pooled = self.fusion(torch.cat((attended, mean, std, maximum), dim=-1))
        pooled = torch.where(available[:, None], pooled, torch.zeros_like(pooled))
        return pooled, available, attention


class ContextD6V2UNet(B5UNet):
    """Keep B5 intact and inject the V2 record-context embedding through zero-init FiLM."""

    def __init__(self, config: ModelConfig = ModelConfig(), context_dim: int = 128,
                 injection_layers: tuple[int, ...] | None = None) -> None:
        super().__init__(config)
        self.context_encoder = AttentiveContextEncoder(context_dim)
        self.context_injection_layers = tuple(range(len(config.channels))) if injection_layers is None else tuple(injection_layers)
        if any(layer < 0 or layer >= len(config.channels) for layer in self.context_injection_layers):
            raise ValueError("context injection layer index is out of range")
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
        return features * (1 + (gamma * gate).unsqueeze(-1)) + (beta * gate).unsqueeze(-1)

    def encode_conditions(self, anchor: torch.Tensor, numeric: torch.Tensor, sex: torch.Tensor,
                          field_mask: torch.Tensor, age_topcoded: torch.Tensor, *,
                          context: torch.Tensor | None = None,
                          context_time_mask: torch.Tensor | None = None,
                          context_window_mask: torch.Tensor | None = None,
                          context_window_weights: torch.Tensor | None = None) -> ContextV2ConditionCache:
        base = super().encode_conditions(anchor, numeric, sex, field_mask, age_topcoded)
        if context is None:
            embedding = anchor.new_zeros(anchor.shape[0], self.context_encoder.output_dim)
            available = torch.zeros(anchor.shape[0], dtype=torch.bool, device=anchor.device)
            attention = anchor.new_zeros(anchor.shape[0], 0)
        else:
            if context_time_mask is None or context_window_mask is None or context_window_weights is None:
                raise ValueError("context input requires time, window, and weight masks")
            embedding, available, attention = self.context_encoder(
                context, context_time_mask, context_window_mask, context_window_weights)
        features = tuple(
            self._modulate(feature, projection, embedding, available)
            if index in self.context_injection_layers else feature
            for index, (feature, projection) in enumerate(
                zip(base.anchor_features, self.context_projection)))
        return ContextV2ConditionCache(features, base.metadata, base.anchor_prediction,
                                       embedding, available, attention)

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


def load_b5_weights(model: ContextD6V2UNet, state: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    expected = {name for name in model.state_dict() if name.startswith("context_")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError(f"B5 initialization mismatch; missing={incompatible.missing_keys}, "
                         f"unexpected={incompatible.unexpected_keys}")

