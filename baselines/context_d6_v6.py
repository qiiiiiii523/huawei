"""V6: V5 bottleneck cross-attention with learned per-window reliability."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from baselines.B5.model import B5UNet
from baselines.context_d6_v5 import (
    ContextD6V5UNet,
    CrossAttentionConditionCache,
    TokenDualPoolContextEncoder,
)


@dataclass
class ReliabilityConditionCache(CrossAttentionConditionCache):
    context_reliability_logits: torch.Tensor
    context_effective_log_weights: torch.Tensor


def effective_window_log_weights(window_weights: torch.Tensor,
                                 reliability_logits: torch.Tensor,
                                 logit_clamp: float = 2.0) -> torch.Tensor:
    """Combine the fixed window prior and bounded learned reliability term."""
    if window_weights.shape != reliability_logits.shape:
        raise ValueError("window weights and reliability logits must have matching shapes")
    if logit_clamp <= 0:
        raise ValueError("logit_clamp must be positive")
    prior = window_weights.clamp_min(1e-8).log()
    return prior + reliability_logits.clamp(-logit_clamp, logit_clamp)


class ReliabilityTokenDualPoolContextEncoder(TokenDualPoolContextEncoder):
    """V5 token encoder with a zero-initialized reliability-logit head."""

    def __init__(self, output_dim: int = 128, query_dim: int = 128,
                 initial_query_fraction: float = 0.2,
                 reliability_hidden: int = 64,
                 reliability_logit_clamp: float = 2.0) -> None:
        super().__init__(output_dim, query_dim, initial_query_fraction)
        if reliability_hidden < 1:
            raise ValueError("reliability_hidden must be positive")
        if reliability_logit_clamp <= 0:
            raise ValueError("reliability_logit_clamp must be positive")
        self.reliability_logit_clamp = float(reliability_logit_clamp)
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(output_dim + 2),
            nn.Linear(output_dim + 2, reliability_hidden),
            nn.SiLU(),
            nn.Linear(reliability_hidden, 1),
        )
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)

    def forward(self, context: torch.Tensor, time_mask: torch.Tensor,
                window_mask: torch.Tensor, window_weights: torch.Tensor,
                query: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Coverage is the fraction of time points with at least one observed context lead.
        temporal_coverage = valid.any(dim=2).to(tokens.dtype).mean(dim=-1)
        log_prior = window_weights.clamp_min(1e-8).log()
        reliability_input = torch.cat((
            tokens,
            temporal_coverage.unsqueeze(-1),
            log_prior.to(tokens.dtype).unsqueeze(-1),
        ), dim=-1)
        reliability_logits = self.reliability_head(reliability_input).squeeze(-1)
        reliability_logits = reliability_logits.clamp(
            -self.reliability_logit_clamp, self.reliability_logit_clamp)
        effective_log_weights = effective_window_log_weights(
            window_weights, reliability_logits, self.reliability_logit_clamp)

        gated = torch.tanh(self.attention_value(tokens)) * torch.sigmoid(self.attention_gate(tokens))
        base_scores = self.attention_score(gated).squeeze(-1)
        query_key = torch.tanh(self.query_key(query)).unsqueeze(1)
        token_key = torch.tanh(self.token_key(tokens))
        query_scores = base_scores + (token_key * query_key).sum(dim=-1) / self.query_scale
        base_scores = (base_scores + effective_log_weights).masked_fill(
            ~valid_windows, torch.finfo(base_scores.dtype).min)
        query_scores = (query_scores + effective_log_weights).masked_fill(
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

        weights = torch.where(valid_windows, effective_log_weights, torch.zeros_like(effective_log_weights))
        weights = torch.softmax(weights.masked_fill(
            ~valid_windows, torch.finfo(weights.dtype).min), dim=1)
        weights = torch.where(valid_windows, weights, torch.zeros_like(weights))
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
        reliability_logits = torch.where(valid_windows, reliability_logits,
                                         torch.zeros_like(reliability_logits))
        return (pooled, available, attention, tokens, valid_windows,
                reliability_logits, effective_log_weights)


class ContextD6V6UNet(ContextD6V5UNet):
    """V5 model with per-window reliability used only in context pooling weights."""

    def __init__(self, config=None, context_dim: int = 128,
                 initial_query_fraction: float = 0.2,
                 injection_layers: tuple[int, ...] | None = None,
                 cross_attention_heads: int = 4,
                 cross_attention_dropout: float = 0.1,
                 reliability_hidden: int = 64,
                 reliability_logit_clamp: float = 2.0) -> None:
        super().__init__(config, context_dim, initial_query_fraction, injection_layers,
                         cross_attention_heads, cross_attention_dropout)
        self.context_encoder = ReliabilityTokenDualPoolContextEncoder(
            context_dim, context_dim, initial_query_fraction, reliability_hidden,
            reliability_logit_clamp)

    def encode_conditions(self, anchor: torch.Tensor, numeric: torch.Tensor, sex: torch.Tensor,
                          field_mask: torch.Tensor, age_topcoded: torch.Tensor, *,
                          context: torch.Tensor | None = None,
                          context_time_mask: torch.Tensor | None = None,
                          context_window_mask: torch.Tensor | None = None,
                          context_window_weights: torch.Tensor | None = None
                          ) -> ReliabilityConditionCache:
        base = B5UNet.encode_conditions(self, anchor, numeric, sex, field_mask, age_topcoded)
        if context is None:
            embedding = anchor.new_zeros(anchor.shape[0], self.context_encoder.output_dim)
            available = torch.zeros(anchor.shape[0], dtype=torch.bool, device=anchor.device)
            attention = anchor.new_zeros(anchor.shape[0], 0)
            tokens = anchor.new_zeros(anchor.shape[0], 1, self.context_encoder.output_dim)
            token_mask = torch.zeros(anchor.shape[0], 1, dtype=torch.bool, device=anchor.device)
            reliability_logits = anchor.new_zeros(anchor.shape[0], 1)
            effective_log_weights = anchor.new_zeros(anchor.shape[0], 1)
        else:
            if context_time_mask is None or context_window_mask is None or context_window_weights is None:
                raise ValueError("context input requires time, window, and weight masks")
            anchor_summary = base.anchor_features[-1]
            query = self.query_projection(torch.cat((anchor_summary.mean(dim=-1),
                                                      anchor_summary.std(dim=-1)), dim=-1))
            (embedding, available, attention, tokens, token_mask,
             reliability_logits, effective_log_weights) = self.context_encoder(
                context, context_time_mask, context_window_mask, context_window_weights, query)
        features = tuple(
            self._modulate(feature, projection, embedding, available)
            if index in self.context_injection_layers else feature
            for index, (feature, projection) in enumerate(
                zip(base.anchor_features, self.context_projection)))
        return ReliabilityConditionCache(
            features, base.metadata, base.anchor_prediction, embedding, available,
            attention, tokens, token_mask, reliability_logits, effective_log_weights)


def load_b5_weights(model: ContextD6V6UNet, state: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    expected = {
        name for name in model.state_dict()
        if name.startswith("context_") or name.startswith("query_projection.")
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError(f"B5 initialization mismatch; missing={incompatible.missing_keys}, "
                         f"unexpected={incompatible.unexpected_keys}")
