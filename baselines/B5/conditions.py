"""Continuous-time and population-level conditioning for B5-U."""
from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        frequencies = torch.exp(-math.log(10000.) * torch.arange(dimension // 2) / max(1, dimension // 2 - 1))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension * 4), nn.SiLU(), nn.Linear(dimension * 4, dimension))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        phases = time.float().reshape(-1, 1) * 1000. * self.frequencies.reshape(1, -1)
        return self.mlp(torch.cat((phases.sin(), phases.cos()), dim=-1))


class MetadataEncoder(nn.Module):
    def __init__(self, dimension: int, enabled: bool, dropout: float, field_dropout: float = 0.) -> None:
        super().__init__()
        self.enabled, self.dropout = enabled, dropout
        self.field_dropout = field_dropout
        self.sex_embedding = nn.Embedding(3, 8)
        self.mlp = nn.Sequential(nn.Linear(16, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        self.dimension = dimension

    def forward(self, numeric: torch.Tensor, sex: torch.Tensor, field_mask: torch.Tensor,
                age_topcoded: torch.Tensor) -> torch.Tensor:
        batch = numeric.shape[0]
        if numeric.shape != (batch, 3) or field_mask.shape != (batch, 4) or age_topcoded.shape != (batch, 1):
            raise ValueError("Expected demographics [B,3], field mask [B,4], topcoded [B,1]")
        if sex.shape != (batch,) or torch.any((sex < 0) | (sex > 2)):
            raise ValueError("sex must contain IDs 0=male,1=female,2=unknown")
        if not self.enabled:
            return numeric.new_zeros(batch, self.dimension)
        valid = field_mask.bool().clone()
        if self.training and self.dropout:
            valid &= (torch.rand(batch, 1, device=numeric.device) >= self.dropout)
        if self.training and self.field_dropout:
            valid &= (torch.rand(batch, 4, device=numeric.device) >= self.field_dropout)
        numeric_valid = valid[:, [0, 2, 3]]
        values = torch.where(numeric_valid, numeric, torch.zeros_like(numeric))
        if not torch.isfinite(values).all():
            raise ValueError("Observed demographics must be finite")
        sex = torch.where(valid[:, 1], sex.long(), torch.full_like(sex.long(), 2))
        topcoded = torch.where(valid[:, :1], age_topcoded, torch.zeros_like(age_topcoded))
        if not torch.isfinite(topcoded).all():
            raise ValueError("Invalid age flag")
        return self.mlp(torch.cat((values, self.sex_embedding(sex), valid.float(), topcoded), dim=-1))


class FiLM(nn.Module):
    def __init__(self, dimension: int, channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dimension, channels * 2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.projection(metadata).chunk(2, dim=-1)
        return features * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
