"""Low-resolution temporal residual branch; no hidden targets or cross-window context."""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn


def resize_linear_1d(signal: torch.Tensor, length: int) -> torch.Tensor:
    """Half-pixel linear resize using deterministic-capable index_select backward."""
    if signal.ndim != 3 or signal.shape[-1] < 1 or not isinstance(length, int) or length < 1:
        raise ValueError('Expected [B,C,T] and a positive output length')
    if not signal.is_floating_point():
        raise ValueError('Resize requires floating-point features')
    source_length = signal.shape[-1]
    if source_length == length:
        return signal
    positions = (torch.arange(length, device=signal.device, dtype=torch.float64)+.5)*(source_length/length)-.5
    left = positions.floor().long()
    right_weight = positions-left
    values = signal.float() if signal.dtype in (torch.float16, torch.bfloat16) else signal
    weight = right_weight.to(values.dtype)[None, None]
    result = (values.index_select(-1, left.clamp(0, source_length-1))*(1-weight)
              + values.index_select(-1, (left+1).clamp(0, source_length-1))*weight)
    return result.to(signal.dtype)


class TemporalBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(nn.Linear(dimension, ffn), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn, dimension), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        q = self.norm_attention(tokens)
        if q.is_cuda and torch.are_deterministic_algorithms_enabled():
            from torch.nn.attention import SDPBackend, sdpa_kernel
            context = sdpa_kernel(SDPBackend.MATH)
        else:
            context = nullcontext()
        with context:
            attended = self.attention(q, q, q, need_weights=False, is_causal=False)[0]
        tokens = tokens+self.dropout(attended)
        return tokens+self.ffn(self.norm_ffn(tokens))


class TemporalResidual(nn.Module):
    def __init__(self, channels: int, tokens: int, dimension: int, heads: int,
                 layers: int, ffn: int, dropout: float):
        super().__init__()
        self.token_count = tokens
        self.input_projection = nn.Conv1d(channels, dimension, 1)
        self.position = nn.Parameter(torch.empty(1, tokens, dimension))
        nn.init.normal_(self.position, std=.02)
        self.blocks = nn.ModuleList([TemporalBlock(dimension, heads, ffn, dropout) for _ in range(layers)])
        self.norm_output = nn.LayerNorm(dimension)
        self.output_projection = nn.Conv1d(dimension, channels, 1)
        self.reset_output_projection()

    def reset_output_projection(self):
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features):
        length = features.shape[-1]
        count = min(self.token_count, length)
        if length % count == 0:
            pooled = F.avg_pool1d(features, length//count, stride=length//count)
        else:
            pooled = resize_linear_1d(features, count)
        tokens = self.input_projection(pooled).transpose(1, 2)
        position = self.position if count == self.token_count else resize_linear_1d(self.position.transpose(1, 2), count).transpose(1, 2)
        tokens = tokens+position.to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        delta = self.output_projection(self.norm_output(tokens).transpose(1, 2))
        return features+resize_linear_1d(delta, length)
