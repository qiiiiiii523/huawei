"""Zero-initialized residual fusion of the three existing encoder scales.

This module uses features of the current FM state and visible conditions only.
It has no attention, target-dependent calibration, or cross-window context.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class WeightedScaleResidual(nn.Module):
    def __init__(self, channels: tuple[int, int, int], dimension: int, output_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv1d(c, dimension, 1) for c in channels])
        # One set of three global learned scale weights per decoder stage.
        self.scale_logits = nn.Parameter(torch.zeros(3))
        self.norm = nn.GroupNorm(math.gcd(dimension, 8), dimension)
        self.output_projection = nn.Conv1d(dimension, output_channels, 1)
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor], length: int) -> torch.Tensor:
        if len(features) != 3 or length < 1:
            raise ValueError("Fusion requires three scales and a positive output length")
        # Same nearest-neighbor alignment as the original decoder. Gradients
        # remain compatible with strict deterministic CUDA settings.
        weights = self.scale_logits.float().softmax(dim=0)
        fused = None
        for feature, projection, weight in zip(features, self.projections, weights):
            aligned = F.interpolate(feature, size=length, mode="nearest")
            projected = projection(aligned)
            contribution = projected * weight.to(projected.dtype)
            fused = contribution if fused is None else fused + contribution
        return self.output_projection(F.silu(self.norm(fused)))


class DecoderFusion(nn.Module):
    def __init__(self, channels: tuple[int, int, int], dimension: int) -> None:
        super().__init__()
        self.at_half = WeightedScaleResidual(channels, dimension, channels[1])
        self.at_full = WeightedScaleResidual(channels, dimension, channels[0])

    def reset_output_projections(self) -> None:
        self.at_half.reset_output_projection()
        self.at_full.reset_output_projection()
