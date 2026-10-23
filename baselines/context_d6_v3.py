"""V3: V2 context tokens selected by a query derived from the current I lead."""
from __future__ import annotations

import math

import torch
from torch import nn

from baselines.context_d6_v2.model import (
    AttentiveContextEncoder,
    ContextD6V2UNet,
    ContextV2ConditionCache,
)


class AnchorQueryContextEncoder(AttentiveContextEncoder):
    """Add query-key matching while retaining V2 gated attention and statistics."""

    def __init__(self, output_dim: int = 128, query_dim: int = 256) -> None:
        super().__init__(output_dim)
        hidden = max(32, output_dim // 2)
        self.token_key = nn.Linear(output_dim, hidden)
        self.query_key = nn.Linear(query_dim, hidden)
        self.query_scale = math.sqrt(float(hidden))

    def forward(self, context: torch.Tensor, time_mask: torch.Tensor,
                window_mask: torch.Tensor, window_weights: torch.Tensor,
                query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.ndim != 4 or context.shape[2:] != (6, 5000):
            raise ValueError("context must be [B,W,6,5000]")
        if time_mask.shape != context.shape or window_mask.shape != context.shape[:2]:
            raise ValueError("context masks disagree with context shape")
        if window_weights.shape != context.shape[:2]:
            raise ValueError("context window weights must be [B,W]")
        if query.ndim != 2 or query.shape[0] != context.shape[0]:
            raise ValueError("query must be [B,D]")
        batch, windows = context.shape[:2]
        valid = time_mask.bool() & window_mask[:, :, None, None].bool()
        safe = torch.where(valid, context, torch.zeros_like(context))
        support = valid.any(dim=2).reshape(batch * windows, 1, 5000).to(context.dtype)
        tokens = self.window_encoder(safe.reshape(batch * windows, 6, 5000), support)
        tokens = tokens.reshape(batch, windows, self.output_dim)
        valid_windows = window_mask.bool() & valid.any(dim=(2, 3)) & (window_weights > 0)
        available = valid_windows.any(dim=1)
        gated = torch.tanh(self.attention_value(tokens)) * torch.sigmoid(self.attention_gate(tokens))
        scores = self.attention_score(gated).squeeze(-1)
        token_key = torch.tanh(self.token_key(tokens))
        query_key = torch.tanh(self.query_key(query)).unsqueeze(1)
        scores = scores + (token_key * query_key).sum(dim=-1) / self.query_scale
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
        maximum = tokens.masked_fill(~valid_windows.unsqueeze(-1), torch.finfo(tokens.dtype).min).amax(dim=1)
        maximum = torch.where(available[:, None], maximum, torch.zeros_like(maximum))
        pooled = self.fusion(torch.cat((attended, mean, std, maximum), dim=-1))
        pooled = torch.where(available[:, None], pooled, torch.zeros_like(pooled))
        return pooled, available, attention


class ContextD6V3UNet(ContextD6V2UNet):
    """V2 with anchor-query selection of historical context windows."""

    def __init__(self, config=None, context_dim: int = 128,
                 injection_layers: tuple[int, ...] | None = None) -> None:
        super().__init__(config, context_dim, injection_layers=injection_layers)
        self.context_encoder = AnchorQueryContextEncoder(context_dim, context_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(config.channels[-1] * 2, context_dim),
            nn.LayerNorm(context_dim),
            nn.SiLU(),
        )

    def encode_conditions(self, anchor: torch.Tensor, numeric: torch.Tensor, sex: torch.Tensor,
                          field_mask: torch.Tensor, age_topcoded: torch.Tensor, *,
                          context: torch.Tensor | None = None,
                          context_time_mask: torch.Tensor | None = None,
                          context_window_mask: torch.Tensor | None = None,
                          context_window_weights: torch.Tensor | None = None) -> ContextV2ConditionCache:
        base = super(ContextD6V2UNet, self).encode_conditions(
            anchor, numeric, sex, field_mask, age_topcoded)
        if context is None:
            embedding = anchor.new_zeros(anchor.shape[0], self.context_encoder.output_dim)
            available = torch.zeros(anchor.shape[0], dtype=torch.bool, device=anchor.device)
            attention = anchor.new_zeros(anchor.shape[0], 0)
        else:
            if context_time_mask is None or context_window_mask is None or context_window_weights is None:
                raise ValueError("context input requires time, window, and weight masks")
            anchor_summary = base.anchor_features[-1]
            query = self.query_projection(torch.cat((anchor_summary.mean(dim=-1),
                                                      anchor_summary.std(dim=-1)), dim=-1))
            embedding, available, attention = self.context_encoder(
                context, context_time_mask, context_window_mask, context_window_weights, query)
        features = tuple(
            self._modulate(feature, projection, embedding, available)
            if index in self.context_injection_layers else feature
            for index, (feature, projection) in enumerate(
                zip(base.anchor_features, self.context_projection)))
        return ContextV2ConditionCache(features, base.metadata, base.anchor_prediction,
                                       embedding, available, attention)


def load_b5_weights(model: ContextD6V3UNet, state: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    expected = {name for name in model.state_dict()
                if name.startswith("context_") or name.startswith("query_projection.")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError(f"B5 initialization mismatch; missing={incompatible.missing_keys}, "
                         f"unexpected={incompatible.unexpected_keys}")

