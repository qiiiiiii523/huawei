"""V5: V4 FiLM plus zero-gated bottleneck cross-attention over context windows."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from baselines.B5.model import B5UNet
from baselines.context_d6_v2.model import ContextV2ConditionCache
from baselines.context_d6_v4 import ContextD6V4UNet, DualPoolContextEncoder


@dataclass
class CrossAttentionConditionCache(ContextV2ConditionCache):
    context_tokens: torch.Tensor
    context_token_mask: torch.Tensor


class TokenDualPoolContextEncoder(DualPoolContextEncoder):
    """Return the V4 pooled embedding and the masked per-window tokens."""

    def forward(self, context: torch.Tensor, time_mask: torch.Tensor,
                window_mask: torch.Tensor, window_weights: torch.Tensor,
                query: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
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
        base_scores = self.attention_score(gated).squeeze(-1)
        query_key = torch.tanh(self.query_key(query)).unsqueeze(1)
        token_key = torch.tanh(self.token_key(tokens))
        query_scores = base_scores + (token_key * query_key).sum(dim=-1) / self.query_scale
        log_weights = window_weights.clamp_min(1e-8).log()
        base_scores = (base_scores + log_weights).masked_fill(
            ~valid_windows, torch.finfo(base_scores.dtype).min)
        query_scores = (query_scores + log_weights).masked_fill(
            ~valid_windows, torch.finfo(query_scores.dtype).min)
        base_attention = torch.softmax(base_scores, dim=1)
        query_attention = torch.softmax(query_scores, dim=1)
        base_attention = torch.where(valid_windows, base_attention, torch.zeros_like(base_attention))
        query_attention = torch.where(valid_windows, query_attention, torch.zeros_like(query_attention))
        base_attention = base_attention / base_attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        query_attention = query_attention / query_attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attended_base = (tokens * base_attention.unsqueeze(-1)).sum(dim=1)
        attended_query = (tokens * query_attention.unsqueeze(-1)).sum(dim=1)
        query_fraction = torch.sigmoid(self.mix_gate(query)).to(tokens.dtype)
        attended = (1.0 - query_fraction) * attended_base + query_fraction * attended_query

        weights = torch.where(valid_windows, window_weights, torch.zeros_like(window_weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mean = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        variance = ((tokens - mean[:, None]) ** 2 * weights.unsqueeze(-1)).sum(dim=1)
        std = (variance.clamp_min(0) + 1e-6).sqrt()
        maximum = tokens.masked_fill(
            ~valid_windows.unsqueeze(-1), torch.finfo(tokens.dtype).min).amax(dim=1)
        maximum = torch.where(available[:, None], maximum, torch.zeros_like(maximum))
        pooled = self.fusion(torch.cat((attended, mean, std, maximum), dim=-1))
        pooled = torch.where(available[:, None], pooled, torch.zeros_like(pooled))
        attention = (1.0 - query_fraction) * base_attention + query_fraction * query_attention
        tokens = torch.where(valid_windows.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return pooled, available, attention, tokens, valid_windows


class ContextD6V5UNet(ContextD6V4UNet):
    """Add a zero-gated context cross-attention residual at the B5 bottleneck."""

    def __init__(self, config=None, context_dim: int = 128,
                 initial_query_fraction: float = 0.2,
                 injection_layers: tuple[int, ...] | None = None,
                 cross_attention_heads: int = 4,
                 cross_attention_dropout: float = 0.1) -> None:
        super().__init__(config, context_dim, initial_query_fraction, injection_layers)
        self.context_encoder = TokenDualPoolContextEncoder(
            context_dim, context_dim, initial_query_fraction)
        bottleneck_dim = config.channels[-1]
        self.context_cross_norm = nn.LayerNorm(bottleneck_dim)
        self.context_cross_query = nn.Linear(bottleneck_dim, context_dim)
        self.context_cross_attention = nn.MultiheadAttention(
            context_dim, cross_attention_heads, dropout=cross_attention_dropout,
            batch_first=True)
        self.context_cross_output = nn.Linear(context_dim, bottleneck_dim)
        self.context_cross_gate = nn.Parameter(torch.zeros(()))

    def encode_conditions(self, anchor: torch.Tensor, numeric: torch.Tensor, sex: torch.Tensor,
                          field_mask: torch.Tensor, age_topcoded: torch.Tensor, *,
                          context: torch.Tensor | None = None,
                          context_time_mask: torch.Tensor | None = None,
                          context_window_mask: torch.Tensor | None = None,
                          context_window_weights: torch.Tensor | None = None
                          ) -> CrossAttentionConditionCache:
        base = B5UNet.encode_conditions(
            self, anchor, numeric, sex, field_mask, age_topcoded)
        if context is None:
            embedding = anchor.new_zeros(anchor.shape[0], self.context_encoder.output_dim)
            available = torch.zeros(anchor.shape[0], dtype=torch.bool, device=anchor.device)
            attention = anchor.new_zeros(anchor.shape[0], 0)
            tokens = anchor.new_zeros(anchor.shape[0], 1, self.context_encoder.output_dim)
            token_mask = torch.zeros(anchor.shape[0], 1, dtype=torch.bool, device=anchor.device)
        else:
            if context_time_mask is None or context_window_mask is None or context_window_weights is None:
                raise ValueError("context input requires time, window, and weight masks")
            anchor_summary = base.anchor_features[-1]
            query = self.query_projection(torch.cat((anchor_summary.mean(dim=-1),
                                                      anchor_summary.std(dim=-1)), dim=-1))
            embedding, available, attention, tokens, token_mask = self.context_encoder(
                context, context_time_mask, context_window_mask, context_window_weights, query)
        features = tuple(
            self._modulate(feature, projection, embedding, available)
            if index in self.context_injection_layers else feature
            for index, (feature, projection) in enumerate(
                zip(base.anchor_features, self.context_projection)))
        return CrossAttentionConditionCache(
            features, base.metadata, base.anchor_prediction, embedding, available,
            attention, tokens, token_mask)

    def _cross_attend(self, features: torch.Tensor,
                      condition: CrossAttentionConditionCache) -> torch.Tensor:
        tokens = condition.context_tokens
        valid = condition.context_token_mask
        safe_valid = valid.clone()
        safe_tokens = tokens.clone()
        unavailable = ~safe_valid.any(dim=1)
        safe_valid[unavailable, 0] = True
        safe_tokens[unavailable, 0] = 0
        queries = self.context_cross_query(
            self.context_cross_norm(features.transpose(1, 2)))
        attended, _ = self.context_cross_attention(
            queries, safe_tokens, safe_tokens,
            key_padding_mask=~safe_valid, need_weights=False)
        residual = self.context_cross_output(attended).transpose(1, 2)
        residual = residual * condition.context_available.to(residual.dtype)[:, None, None]
        return features + torch.tanh(self.context_cross_gate) * residual

    def velocity(self, state: torch.Tensor, time: torch.Tensor,
                 condition: CrossAttentionConditionCache) -> torch.Tensor:
        f0, f1, f2 = condition.anchor_features
        if state.shape != (f0.shape[0], 11, f0.shape[-1]) or time.shape != (f0.shape[0],):
            raise ValueError("FM state/time shape mismatch")
        if (not torch.isfinite(state).all() or not torch.isfinite(time).all()
                or torch.any((time < 0) | (time > 1))):
            raise ValueError("FM state must be finite, time in [0,1]")
        time_embedding = self.time_embedding(time)
        features = self.stem(state) + self.condition_enc[0](f0)
        skip0 = self._blocks(self.encode0, features, time_embedding, condition.metadata)
        features = self.down1(skip0) + self.condition_enc[1](f1)
        skip1 = self._blocks(self.encode1, features, time_embedding, condition.metadata)
        features = self.down2(skip1) + self.condition_enc[2](f2)
        features = self._blocks(self.bottleneck, features, time_embedding, condition.metadata)
        features = self._cross_attend(features, condition)
        features = self.up1(torch.nn.functional.interpolate(
            features, size=skip1.shape[-1], mode="nearest"))
        features = self.merge1(torch.cat((features, skip1), dim=1)) + self.condition_dec[1](f1)
        features = self._blocks(self.decode1, features, time_embedding, condition.metadata)
        features = self.up0(torch.nn.functional.interpolate(
            features, size=skip0.shape[-1], mode="nearest"))
        features = self.merge0(torch.cat((features, skip0), dim=1)) + self.condition_dec[0](f0)
        return self.output(self._blocks(self.decode0, features, time_embedding, condition.metadata))


def load_b5_weights(model: ContextD6V5UNet, state: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    expected = {
        name for name in model.state_dict()
        if (name.startswith("context_") or name.startswith("query_projection."))
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError(f"B5 initialization mismatch; missing={incompatible.missing_keys}, "
                         f"unexpected={incompatible.unexpected_keys}")
