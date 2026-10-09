"""Conditional three-scale 1D U-Net predicting a continuous velocity field."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .conditions import FiLM, MetadataEncoder, TimeEmbedding
from .config import ModelConfig


def norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(channels, 8), channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, metadata_dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm1, self.norm2 = norm(channels), norm(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.time = nn.Linear(time_dim, channels)
        self.film = FiLM(metadata_dim, channels)

    def forward(self, x: torch.Tensor, time: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x))) + self.time(time).unsqueeze(-1)
        return x + self.conv2(F.silu(self.film(self.norm2(h), metadata)))


@dataclass
class ConditionCache:
    anchor_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    metadata: torch.Tensor
    anchor_prediction: torch.Tensor


class B5UNet(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        c0, c1, c2 = config.channels
        self.time_embedding = TimeEmbedding(config.time_dim)
        self.metadata_encoder = MetadataEncoder(config.metadata_dim, config.metadata_enabled, config.metadata_dropout,
                                                config.metadata_field_dropout)
        self.anchor_full = nn.Conv1d(1, c0, 7, padding=3)
        self.anchor_half = nn.Conv1d(c0, c1, 4, stride=2, padding=1)
        self.anchor_quarter = nn.Conv1d(c1, c2, 4, stride=2, padding=1)
        self.anchor_head = nn.Sequential(nn.Conv1d(c0, c0, 3, padding=1), nn.SiLU(), nn.Conv1d(c0, 1, 1))
        self.stem = nn.Conv1d(11, c0, 7, padding=3)
        block = lambda c, d=1: ResidualBlock(c, config.time_dim, config.metadata_dim, d)
        self.encode0 = nn.ModuleList([block(c0), block(c0)])
        self.down1 = nn.Conv1d(c0, c1, 4, stride=2, padding=1)
        self.encode1 = nn.ModuleList([block(c1), block(c1)])
        self.down2 = nn.Conv1d(c1, c2, 4, stride=2, padding=1)
        self.bottleneck = nn.ModuleList([block(c2, d) for d in config.dilations])
        self.up1 = nn.Conv1d(c2, c1, 3, padding=1)
        self.merge1 = nn.Conv1d(c1 * 2, c1, 1)
        self.decode1 = nn.ModuleList([block(c1), block(c1)])
        self.up0 = nn.Conv1d(c1, c0, 3, padding=1)
        self.merge0 = nn.Conv1d(c0 * 2, c0, 1)
        self.decode0 = nn.ModuleList([block(c0), block(c0)])
        self.condition_enc = nn.ModuleList([nn.Conv1d(c, c, 1) for c in config.channels])
        self.condition_dec = nn.ModuleList([nn.Conv1d(c, c, 1) for c in config.channels[:2]])
        self.output = nn.Conv1d(c0, 11, 1)

    def encode_conditions(self, anchor: torch.Tensor, numeric: torch.Tensor, sex: torch.Tensor,
                          field_mask: torch.Tensor, age_topcoded: torch.Tensor) -> ConditionCache:
        if anchor.ndim != 3 or anchor.shape[1] != 1 or anchor.shape[-1] < 8 or not torch.isfinite(anchor).all():
            raise ValueError("anchor must be finite [B,1,T], T>=8")
        f0 = self.anchor_full(anchor)
        f1 = self.anchor_half(F.silu(f0))
        f2 = self.anchor_quarter(F.silu(f1))
        embedding = self.metadata_encoder(numeric, sex, field_mask, age_topcoded)
        if embedding.shape[0] != anchor.shape[0]:
            raise ValueError("Condition batch mismatch")
        return ConditionCache((f0, f1, f2), embedding, self.anchor_head(f0))

    @staticmethod
    def _blocks(blocks: nn.ModuleList, features: torch.Tensor, time: torch.Tensor,
                metadata: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            features = block(features, time, metadata)
        return features

    def velocity(self, state: torch.Tensor, time: torch.Tensor, condition: ConditionCache) -> torch.Tensor:
        f0, f1, f2 = condition.anchor_features
        if state.shape != (f0.shape[0], 11, f0.shape[-1]) or time.shape != (f0.shape[0],):
            raise ValueError("FM state/time shape mismatch")
        if not torch.isfinite(state).all() or not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("FM state must be finite, time in [0,1]")
        t = self.time_embedding(time)
        h = self.stem(state) + self.condition_enc[0](f0)
        skip0 = self._blocks(self.encode0, h, t, condition.metadata)
        h = self.down1(skip0) + self.condition_enc[1](f1)
        skip1 = self._blocks(self.encode1, h, t, condition.metadata)
        h = self.down2(skip1) + self.condition_enc[2](f2)
        h = self._blocks(self.bottleneck, h, t, condition.metadata)
        h = self.up1(F.interpolate(h, size=skip1.shape[-1], mode="nearest"))
        h = self.merge1(torch.cat((h, skip1), dim=1)) + self.condition_dec[1](f1)
        h = self._blocks(self.decode1, h, t, condition.metadata)
        h = self.up0(F.interpolate(h, size=skip0.shape[-1], mode="nearest"))
        h = self.merge0(torch.cat((h, skip0), dim=1)) + self.condition_dec[0](f0)
        return self.output(self._blocks(self.decode0, h, t, condition.metadata))

    def forward(self, state: torch.Tensor, time: torch.Tensor, anchor: torch.Tensor,
                numeric: torch.Tensor, sex: torch.Tensor, field_mask: torch.Tensor,
                age_topcoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.encode_conditions(anchor, numeric, sex, field_mask, age_topcoded)
        return self.velocity(state, time, condition), condition.anchor_prediction
