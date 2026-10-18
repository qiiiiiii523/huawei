"""An optional per-record attenuation gate around the unchanged body II CNN."""
from __future__ import annotations

import torch
from torch import nn

from .body_model import BodyIIAdapter
from .body_gate_features import FEATURE_COUNT


class RecordGatedIIAdapter(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("C", "G"):
            raise ValueError("Use C=fixed alpha1, G=learned record alpha")
        self.mode = mode
        self.body = BodyIIAdapter()
        self.gate = nn.Sequential(nn.Linear(FEATURE_COUNT, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        if mode == "C":
            self.gate.requires_grad_(False)

    def alpha(self, features):
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise ValueError("Record gate requires [B,48] visible features")
        if self.mode == "C":
            return features.new_ones((len(features), 1))
        # Zero output => alpha exactly1, matching C/source at initialization.
        # Clamp has the boundary subgradient needed to learn attenuation from1.
        return (2 * torch.sigmoid(self.gate(features))).clamp(max=1.)

    def forward(self, lead_i, base, beat, stats, enabled, record_features):
        corrected = self.body(lead_i, base, beat, stats, enabled)
        alpha = self.alpha(record_features)
        return (1 - alpha) * base + alpha * corrected
