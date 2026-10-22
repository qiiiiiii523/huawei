"""Two-stage deterministic regression with exact limb-lead vector operations."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from baselines.B5.conditions import MetadataEncoder
from baselines.B5.config import ModelConfig


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(channels, 8), channels, affine=False)


class ConditionalResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm1, self.norm2 = _norm(channels), _norm(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.affine1 = nn.Linear(condition_dim, channels * 2)
        self.affine2 = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.affine1.weight); nn.init.zeros_(self.affine1.bias)
        nn.init.zeros_(self.affine2.weight); nn.init.zeros_(self.affine2.bias)

    @staticmethod
    def _condition(x: torch.Tensor, projection: nn.Linear, condition: torch.Tensor) -> torch.Tensor:
        gain, bias = projection(condition).chunk(2, dim=-1)
        return x * (1 + gain.unsqueeze(-1)) + bias.unsqueeze(-1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self._condition(self.norm1(x), self.affine1, condition)))
        h = self.conv2(F.silu(self._condition(self.norm2(h), self.affine2, condition)))
        return x + h


class LeadConditionalUNet(nn.Module):
    """Shared generator whose AdaIN condition identifies the desired target lead."""
    def __init__(self, in_channels: int, config: ModelConfig) -> None:
        super().__init__()
        c0, c1, c2 = config.channels
        d = config.metadata_dim
        self.target_embedding = nn.Embedding(12, d)
        self.condition = nn.Sequential(nn.Linear(d * 2, d), nn.SiLU(), nn.Linear(d, d))
        self.stem = nn.Conv1d(in_channels, c0, 7, padding=3)
        block = lambda c, dilation=1: ConditionalResidualBlock(c, d, dilation)
        self.enc0 = nn.ModuleList([block(c0), block(c0)])
        self.down1 = nn.Conv1d(c0, c1, 4, stride=2, padding=1)
        self.enc1 = nn.ModuleList([block(c1), block(c1)])
        self.down2 = nn.Conv1d(c1, c2, 4, stride=2, padding=1)
        self.middle = nn.ModuleList([block(c2, dilation) for dilation in config.dilations])
        self.up1, self.merge1 = nn.Conv1d(c2, c1, 3, padding=1), nn.Conv1d(c1 * 2, c1, 1)
        self.dec1 = nn.ModuleList([block(c1), block(c1)])
        self.up0, self.merge0 = nn.Conv1d(c1, c0, 3, padding=1), nn.Conv1d(c0 * 2, c0, 1)
        self.dec0 = nn.ModuleList([block(c0), block(c0)])
        self.output = nn.Conv1d(c0, 1, 1)

    @staticmethod
    def _blocks(blocks: nn.ModuleList, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            x = block(x, condition)
        return x

    def _target_condition(self, metadata: torch.Tensor, target_id: int) -> torch.Tensor:
        ids = torch.full((len(metadata),), int(target_id), dtype=torch.long, device=metadata.device)
        return self.condition(torch.cat((metadata, self.target_embedding(ids)), dim=-1))

    def forward(self, source: torch.Tensor, target_id: int, metadata: torch.Tensor,
                *, representation_only: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        condition = self._target_condition(metadata, target_id)
        skip0 = self._blocks(self.enc0, self.stem(source), condition)
        skip1 = self._blocks(self.enc1, self.down1(skip0), condition)
        h = self._blocks(self.middle, self.down2(skip1), condition)
        representation = h.mean(dim=-1)
        if representation_only:
            return None, representation
        h = self.merge1(torch.cat((self.up1(F.interpolate(h, size=skip1.shape[-1], mode="nearest")), skip1), 1))
        h = self._blocks(self.dec1, h, condition)
        h = self.merge0(torch.cat((self.up0(F.interpolate(h, size=skip0.shape[-1], mode="nearest")), skip0), 1))
        h = self._blocks(self.dec0, h, condition)
        return self.output(h), representation

    def generate(self, source: torch.Tensor, target_ids: tuple[int, ...],
                 metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, representations = [], []
        for target_id in target_ids:
            output, representation = self(source, target_id, metadata)
            outputs.append(output); representations.append(representation)
        return torch.cat(outputs, dim=1), torch.stack(representations, dim=1)

    def representations(self, source: torch.Tensor, target_ids: tuple[int, ...],
                        metadata: torch.Tensor) -> torch.Tensor:
        values = [self(source, target_id, metadata, representation_only=True)[1] for target_id in target_ids]
        return torch.stack(values, dim=1)


def derive_limb_leads(anchor_i: torch.Tensor, lead_ii: torch.Tensor,
                      d12_scale: torch.Tensor) -> torch.Tensor:
    """Return normalized I/II/III/aVR/aVL/aVF; identities are enforced in raw uV."""
    scale = torch.as_tensor(d12_scale, dtype=anchor_i.dtype, device=anchor_i.device).reshape(-1)
    if scale.shape != (12,) or not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("d12_scale must contain 12 finite positive lead scales")
    lead_i_uv, lead_ii_uv = anchor_i * scale[0], lead_ii * scale[1]
    lead_iii_uv = lead_ii_uv - lead_i_uv
    avr_uv = -(lead_i_uv + lead_ii_uv) / 2
    avl_uv = (lead_i_uv - lead_iii_uv) / 2
    avf_uv = (lead_ii_uv + lead_iii_uv) / 2
    raw = torch.cat((lead_i_uv, lead_ii_uv, lead_iii_uv, avr_uv, avl_uv, avf_uv), dim=1)
    return raw / scale[:6].reshape(1, 6, 1)


@dataclass
class AuroraOutput:
    full12: torch.Tensor
    lead_ii: torch.Tensor
    chest6: torch.Tensor
    cycles: dict[str, torch.Tensor]
    representations: dict[str, torch.Tensor]


class AuroraRegression(nn.Module):
    """I -> II -> exact limb system -> V1..V6, all deterministic in one inference graph."""
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.metadata_encoder = MetadataEncoder(config.metadata_dim, config.metadata_enabled,
                                                config.metadata_dropout, config.metadata_field_dropout)
        self.limb = LeadConditionalUNet(1, config)
        self.chest = LeadConditionalUNet(6, config)

    def forward(self, anchor: torch.Tensor, numeric: torch.Tensor, sex: torch.Tensor,
                field_mask: torch.Tensor, age_topcoded: torch.Tensor, d12_scale: torch.Tensor,
                *, compute_cycles: bool = False) -> AuroraOutput:
        metadata = self.metadata_encoder(numeric, sex, field_mask, age_topcoded)
        lead_ii, limb_forward_rep = self.limb(anchor, 1, metadata)
        limb6 = derive_limb_leads(anchor, lead_ii, d12_scale)
        chest6, chest_forward_rep = self.chest.generate(limb6, tuple(range(6, 12)), metadata)
        cycles: dict[str, torch.Tensor] = {}
        representations = {"limb_forward": limb_forward_rep, "chest_forward": chest_forward_rep}
        if compute_cycles:
            source_i, limb_reverse_rep = self.limb(lead_ii, 0, metadata)
            limb_cycle, chest_reverse_rep = self.chest.generate(chest6, tuple(range(6)), metadata)
            cycles = {"source_i": source_i, "limb6": limb_cycle}
            representations.update({
                "limb_generated": self.limb.representations(lead_ii, (1,), metadata).squeeze(1),
                "limb_reverse": limb_reverse_rep,
                "limb_source": self.limb.representations(anchor, (0,), metadata).squeeze(1),
                "chest_generated": self.chest.representations(chest6, tuple(range(6, 12)), metadata),
                "chest_reverse": chest_reverse_rep,
                "chest_source": self.chest.representations(limb6, tuple(range(6)), metadata),
            })
        return AuroraOutput(torch.cat((limb6, chest6), dim=1), lead_ii, chest6, cycles, representations)
