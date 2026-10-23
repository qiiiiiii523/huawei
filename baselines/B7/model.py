"""Raw-I concatenation U-Net. Only seven unknown channels have a velocity."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class TimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(dimension // 2) / max(1, dimension // 2 - 1))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension * 4), nn.SiLU(), nn.Linear(dimension * 4, dimension))

    def forward(self, time):
        phase = time[:, None].float() * 1000 * self.frequencies[None]
        return self.mlp(torch.cat((phase.sin(), phase.cos()), dim=-1))


class MetadataEncoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.sex_embedding = nn.Embedding(3, 8)
        self.mlp = nn.Sequential(nn.Linear(16, dimension), nn.SiLU(), nn.Linear(dimension, dimension))

    def forward(self, reference, metadata=None):
        b = len(reference)
        if not metadata:
            numeric = reference.new_zeros(b, 3)
            valid = torch.zeros(b, 4, dtype=torch.bool, device=reference.device)
            sex = torch.full((b,), 2, dtype=torch.long, device=reference.device)
            topcoded = reference.new_zeros(b, 1)
        else:
            numeric = metadata["numeric"].to(device=reference.device, dtype=reference.dtype)
            sex = metadata["sex"].to(device=reference.device)
            valid = metadata["field_mask"].to(device=reference.device, dtype=torch.bool)
            topcoded = metadata["age_topcoded"].to(device=reference.device, dtype=reference.dtype)
        if numeric.shape != (b, 3) or valid.shape != (b, 4) or sex.shape != (b,) or topcoded.shape != (b, 1):
            raise ValueError("Metadata shape mismatch")
        numeric = torch.where(valid[:, [0, 2, 3]], numeric, torch.zeros_like(numeric))
        sex = torch.where(valid[:, 1], sex.long(), torch.full_like(sex.long(), 2))
        topcoded = torch.where(valid[:, :1], topcoded, torch.zeros_like(topcoded))
        if not torch.isfinite(numeric).all() or not torch.isfinite(topcoded).all() or torch.any((sex < 0) | (sex > 2)):
            raise ValueError("Invalid observed demographics")
        return self.mlp(torch.cat((numeric, self.sex_embedding(sex), valid.to(numeric.dtype), topcoded), dim=-1))


class ResidualBlock(nn.Module):
    def __init__(self, channels, time_dim, metadata_dim, dilation=1):
        super().__init__()
        groups = math.gcd(channels, 8)
        self.norm1, self.norm2 = nn.GroupNorm(groups, channels), nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.time = nn.Linear(time_dim, channels)
        self.film = nn.Linear(metadata_dim, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h, time, metadata):
        x = self.conv1(F.silu(self.norm1(h))) + self.time(time)[:, :, None]
        gamma, beta = self.film(metadata).chunk(2, dim=-1)
        x = self.norm2(x) * (1 + gamma[:, :, None]) + beta[:, :, None]
        return h + self.conv2(F.silu(x))


class B7UNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c0, c1, c2 = self.config.channels
        self.time_embedding = TimeEmbedding(self.config.time_dim)
        self.metadata_encoder = MetadataEncoder(self.config.metadata_dim)
        block = lambda c, d=1: ResidualBlock(c, self.config.time_dim, self.config.metadata_dim, d)
        # No separate I encoder/head. The original samples are channel zero.
        self.stem = nn.Conv1d(8, c0, 7, padding=3)
        self.encode0 = nn.ModuleList([block(c0), block(c0)])
        self.down1 = nn.Conv1d(c0, c1, 4, stride=2, padding=1)
        self.encode1 = nn.ModuleList([block(c1), block(c1)])
        self.down2 = nn.Conv1d(c1, c2, 4, stride=2, padding=1)
        self.bottleneck = nn.ModuleList([block(c2, d) for d in self.config.dilations])
        self.up1 = nn.Conv1d(c2, c1, 3, padding=1)
        self.merge1 = nn.Conv1d(c1 * 2, c1, 1)
        self.decode1 = nn.ModuleList([block(c1), block(c1)])
        self.up0 = nn.Conv1d(c1, c0, 3, padding=1)
        self.merge0 = nn.Conv1d(c0 * 2, c0, 1)
        self.decode0 = nn.ModuleList([block(c0), block(c0)])
        self.output = nn.Conv1d(c0, 7, 1)

    @staticmethod
    def _blocks(blocks, h, time, metadata):
        for block in blocks:
            h = block(h, time, metadata)
        return h

    def forward(self, state, time, anchor, metadata=None):
        if state.ndim != 3 or state.shape[1] != 7 or state.shape[-1] < 8:
            raise ValueError("State must be [B,7,T], T>=8")
        if anchor.shape != (len(state), 1, state.shape[-1]) or time.shape != (len(state),):
            raise ValueError("I must be [B,1,T] and time [B]")
        if not torch.isfinite(state).all() or not torch.isfinite(anchor).all() or not torch.isfinite(time).all():
            raise ValueError("Nonfinite inputs")
        if torch.any((time < 0) | (time > 1)):
            raise ValueError("Flow time must be in [0,1]")
        te = self.time_embedding(time)
        me = self.metadata_encoder(anchor, metadata)
        h = self.stem(torch.cat((anchor, state), dim=1))
        s0 = self._blocks(self.encode0, h, te, me)
        s1 = self._blocks(self.encode1, self.down1(s0), te, me)
        h = self._blocks(self.bottleneck, self.down2(s1), te, me)
        h = self.up1(F.interpolate(h, size=s1.shape[-1], mode="nearest"))
        h = self._blocks(self.decode1, self.merge1(torch.cat((h, s1), dim=1)), te, me)
        h = self.up0(F.interpolate(h, size=s0.shape[-1], mode="nearest"))
        h = self._blocks(self.decode0, self.merge0(torch.cat((h, s0), dim=1)), te, me)
        return self.output(h)
