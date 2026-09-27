"""Independent one-dimensional conditional denoiser for B4."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE_VERSION = "B4-I-conditional-ddpm-1d-v2"
ARCHITECTURE_ID = "B4-I-conditional-ddpm-1d-v2"


def architecture_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(-scale * torch.arange(half, device=timesteps.device))
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        return F.pad(embedding, (0, self.dimension - embedding.shape[1]))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, signal: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(signal)
        hidden = self.norm1(hidden)
        hidden = F.silu(hidden + self.time_projection(time_embedding)[:, :, None])
        hidden = self.conv2(self.dropout(hidden))
        hidden = self.norm2(hidden)
        return F.silu(hidden + self.skip(signal))


class B4ConditionalUNet1D(nn.Module):
    """Predict a diffusion target for II--V6 conditioned on complete lead I."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        base = int(self.config.get("base_channels", 64))
        multipliers = tuple(int(value) for value in self.config.get("channel_multipliers", (1, 2, 4)))
        if len(multipliers) != 3:
            raise ValueError("B4 channel_multipliers must contain exactly three levels")
        time_dim = int(self.config.get("time_embedding_dim", 128))
        dropout = float(self.config.get("dropout", 0.1))
        channels = tuple(base * value for value in multipliers)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.noisy_projection = nn.Conv1d(11, channels[0], 3, padding=1)
        self.anchor_full = nn.Conv1d(1, channels[0], 7, padding=3)
        self.anchor_half = nn.Conv1d(channels[0], channels[1], 4, stride=2, padding=1)
        self.anchor_quarter = nn.Conv1d(channels[1], channels[2], 4, stride=2, padding=1)
        self.down1 = ResidualBlock(channels[0], channels[0], time_dim, dropout)
        self.downsample1 = nn.Conv1d(channels[0], channels[1], 4, stride=2, padding=1)
        self.down2 = ResidualBlock(channels[1], channels[1], time_dim, dropout)
        self.downsample2 = nn.Conv1d(channels[1], channels[2], 4, stride=2, padding=1)
        self.middle1 = ResidualBlock(channels[2], channels[2], time_dim, dropout, dilation=2)
        self.middle2 = ResidualBlock(channels[2], channels[2], time_dim, dropout, dilation=4)
        self.upsample2 = nn.ConvTranspose1d(channels[2], channels[1], 4, stride=2, padding=1)
        self.up2 = ResidualBlock(channels[1] * 2, channels[1], time_dim, dropout)
        self.upsample1 = nn.ConvTranspose1d(channels[1], channels[0], 4, stride=2, padding=1)
        self.up1 = ResidualBlock(channels[0] * 2, channels[0], time_dim, dropout)
        self.output = nn.Conv1d(channels[0], 11, 3, padding=1)

    @property
    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture_version": ARCHITECTURE_VERSION,
            "architecture_id": ARCHITECTURE_ID,
            "architecture_config": self.config,
            "architecture_config_hash": architecture_config_hash(self.config),
            "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            "diffused_lead_indices": list(range(1, 12)),
        }

    def forward(self, noisy_missing: torch.Tensor, timesteps: torch.Tensor, anchor_i: torch.Tensor) -> torch.Tensor:
        if noisy_missing.ndim != 3 or noisy_missing.shape[1] != 11:
            raise ValueError("noisy_missing must have shape [B,11,T]")
        if anchor_i.shape != (noisy_missing.shape[0], 1, noisy_missing.shape[2]):
            raise ValueError("anchor_i must have shape [B,1,T]")
        if timesteps.shape != (noisy_missing.shape[0],):
            raise ValueError("timesteps must have shape [B]")
        time_embedding = self.time_embedding(timesteps)
        condition_full = self.anchor_full(anchor_i)
        condition_half = self.anchor_half(F.silu(condition_full))
        condition_quarter = self.anchor_quarter(F.silu(condition_half))
        hidden = self.noisy_projection(noisy_missing) + condition_full
        skip1 = self.down1(hidden, time_embedding)
        skip2 = self.down2(self.downsample1(skip1) + condition_half, time_embedding)
        hidden = self.downsample2(skip2) + condition_quarter
        hidden = self.middle2(self.middle1(hidden, time_embedding), time_embedding)
        hidden = self.upsample2(hidden)
        if hidden.shape[-1] != skip2.shape[-1]:
            hidden = F.interpolate(hidden, size=skip2.shape[-1], mode="linear", align_corners=False)
        hidden = self.up2(torch.cat((hidden, skip2), dim=1), time_embedding)
        hidden = self.upsample1(hidden)
        if hidden.shape[-1] != skip1.shape[-1]:
            hidden = F.interpolate(hidden, size=skip1.shape[-1], mode="linear", align_corners=False)
        return self.output(self.up1(torch.cat((hidden, skip1), dim=1), time_embedding))
