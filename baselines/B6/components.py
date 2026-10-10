"""Adapt M1-main-v3's CNN, axial attention, and per-lead FPN to a velocity network.

Source: ecg12gen/m1_axial.py on baseline/M1-main-v3. Lengths/channels are
parameterized, and the decoder supports the 11 missing velocity channels.
No M1 regressor or historical-context weights are imported.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .resampling import resize_linear_1d


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, channels), channels)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 5, stride: int = 1, dilation: int = 1):
        padding = (kernel - 1) * dilation // 2
        super().__init__(nn.Conv1d(cin, cout, kernel, stride, padding, dilation=dilation),
                         group_norm(cout), nn.GELU())


class LocalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.first = ConvNormAct(channels, channels, 5, dilation=dilation)
        self.second = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                                    group_norm(channels), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.second(self.first(x)))


class MultiScaleCNNEncoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int], dropout: float):
        super().__init__()
        c0, c1, c2 = channels
        self.stem = ConvNormAct(1, c0, 7)
        self.full = nn.Sequential(LocalResidualBlock(c0, 1, dropout), LocalResidualBlock(c0, 2, dropout))
        self.down1 = ConvNormAct(c0, c1, 7, stride=4)
        self.quarter = nn.Sequential(LocalResidualBlock(c1, 1, dropout), LocalResidualBlock(c1, 2, dropout))
        self.down2 = ConvNormAct(c1, c2, 5, stride=5)
        self.coarse = nn.Sequential(LocalResidualBlock(c2, 1, dropout), LocalResidualBlock(c2, 2, dropout))

    def forward(self, anchor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f0 = self.full(self.stem(anchor))
        f1 = self.quarter(self.down1(f0))
        f2 = self.coarse(self.down2(f1))
        return f0, f1, f2


class FeedForward(nn.Sequential):
    def __init__(self, d: int, hidden: int, dropout: float):
        super().__init__(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, d), nn.Dropout(dropout))


class AxialLeadTimeBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float, axes: str):
        super().__init__()
        self.axes = axes
        self.time_norm = nn.LayerNorm(d)
        self.lead_norm = nn.LayerNorm(d)
        self.time_attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) if axes != 'lead_only' else None
        self.lead_attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) if axes != 'time_only' else None
        self.time_ffn_norm, self.lead_ffn_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.time_ffn, self.lead_ffn = FeedForward(d, ffn, dropout), FeedForward(d, ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        batch, leads, length, d = grid.shape
        if leads != 12:
            raise ValueError('B6 axial attention requires I plus 11 missing lead slots')
        x = grid.reshape(batch * leads, length, d)
        if self.time_attention is not None:
            q = self.time_norm(x)
            x = x + self.dropout(self.time_attention(q, q, q, need_weights=False, is_causal=False)[0])
        x = x + self.time_ffn(self.time_ffn_norm(x))
        x = x.reshape(batch, leads, length, d).permute(0, 2, 1, 3).reshape(batch * length, leads, d)
        if self.lead_attention is not None:
            q = self.lead_norm(x)
            x = x + self.dropout(self.lead_attention(q, q, q, need_weights=False, is_causal=False)[0])
        x = x + self.lead_ffn(self.lead_ffn_norm(x))
        return x.reshape(batch, length, leads, d).permute(0, 2, 1, 3)


class LeadConditionedSkip(nn.Module):
    def __init__(self, cin: int, cout: int, d: int):
        super().__init__()
        self.signal, self.lead = nn.Conv1d(cin, cout, 1), nn.Linear(d, cout)

    def forward(self, features: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        return self.signal(features)[:, None] + self.lead(embeddings)[None, :, :, None]


class LeadTimeDecoder(nn.Module):
    def __init__(self, d: int, cnn_channels: tuple[int, ...], decoder_channels: tuple[int, ...], dropout: float):
        super().__init__()
        coarse, fine = decoder_channels
        self.coarse = nn.Linear(d, coarse)
        self.refine_quarter = nn.Sequential(nn.Conv1d(coarse, coarse, 5, padding=2), group_norm(coarse), nn.GELU(), nn.Dropout(dropout))
        self.skip_quarter = LeadConditionedSkip(cnn_channels[1], coarse, d)
        self.fuse_quarter = ConvNormAct(coarse * 2, coarse, 3)
        self.refine_full = nn.Sequential(nn.Conv1d(coarse, fine, 5, padding=2), group_norm(fine), nn.GELU(), nn.Dropout(dropout))
        self.skip_full = LeadConditionedSkip(cnn_channels[0], fine, d)
        self.fuse_full = ConvNormAct(fine * 2, fine, 3)
        self.head, self.lead_bias = nn.Conv1d(fine, 1, 3, padding=1), nn.Linear(d, 1)

    @staticmethod
    def each(features: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        batch, leads, channels, length = features.shape
        return layer(features.reshape(batch * leads, channels, length)).reshape(batch, leads, -1, length)

    @staticmethod
    def resize(features: torch.Tensor, length: int) -> torch.Tensor:
        batch, leads, channels, _ = features.shape
        x = resize_linear_1d(features.reshape(batch * leads, channels, -1), length)
        return x.reshape(batch, leads, channels, length)

    def forward(self, tokens: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.coarse(tokens).permute(0, 1, 3, 2)
        x = self.each(self.resize(x, f1.shape[-1]), self.refine_quarter)
        x = self.each(torch.cat((x, self.skip_quarter(f1, embeddings)), dim=2), self.fuse_quarter)
        x = self.each(self.resize(x, f0.shape[-1]), self.refine_full)
        x = self.each(torch.cat((x, self.skip_full(f0, embeddings)), dim=2), self.fuse_full)
        return self.each(x, self.head).squeeze(2) + self.lead_bias(embeddings)[None, :, :]
