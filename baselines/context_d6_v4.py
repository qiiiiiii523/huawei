"""V4: blend stable V2 pooling with V3 anchor-query pooling."""
from __future__ import annotations

import math

import torch
from torch import nn

from baselines.context_d6_v3 import AnchorQueryContextEncoder, ContextD6V3UNet


class DualPoolContextEncoder(AnchorQueryContextEncoder):
    """Mix unguided and anchor-query attention over the same context tokens."""

    def __init__(self, output_dim: int = 128, query_dim: int = 256,
                 initial_query_fraction: float = 0.2) -> None:
        super().__init__(output_dim, query_dim)
        if not 0.0 < initial_query_fraction < 1.0:
            raise ValueError("initial_query_fraction must be between 0 and 1")
        self.mix_gate = nn.Linear(query_dim, 1)
        nn.init.zeros_(self.mix_gate.weight)
        bias = math.log(initial_query_fraction / (1.0 - initial_query_fraction))
        nn.init.constant_(self.mix_gate.bias, bias)

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
        maximum = tokens.masked_fill(~valid_windows.unsqueeze(-1), torch.finfo(tokens.dtype).min).amax(dim=1)
        maximum = torch.where(available[:, None], maximum, torch.zeros_like(maximum))
        pooled = self.fusion(torch.cat((attended, mean, std, maximum), dim=-1))
        pooled = torch.where(available[:, None], pooled, torch.zeros_like(pooled))
        attention = (1.0 - query_fraction) * base_attention + query_fraction * query_attention
        return pooled, available, attention


class ContextD6V4UNet(ContextD6V3UNet):
    """V3 with a learnable, V2-biased blend of two attention pools."""

    def __init__(self, config=None, context_dim: int = 128,
                 initial_query_fraction: float | None = None,
                 injection_layers: tuple[int, ...] | None = None) -> None:
        if initial_query_fraction is None:
            initial_query_fraction = 0.2
            if isinstance(config, dict):
                initial_query_fraction = float(config.get("initial_query_fraction", 0.2))
        super().__init__(config, context_dim, injection_layers=injection_layers)
        self.context_encoder = DualPoolContextEncoder(
            context_dim, context_dim, initial_query_fraction=initial_query_fraction)


def load_b5_weights(model: ContextD6V4UNet, state: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    expected = {name for name in model.state_dict()
                if name.startswith("context_") or name.startswith("query_projection.")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError(f"B5 initialization mismatch; missing={incompatible.missing_keys}, "
                         f"unexpected={incompatible.unexpected_keys}")
