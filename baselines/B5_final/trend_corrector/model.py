"""Small dilated TCN used to correct slow residual trends after B5-final."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SAMPLES = 60_000
BIN = 125                 # 0.25 s at 500 Hz
STEPS = SAMPLES // BIN    # 480
N_LEADS = 12
INPUT_INDICES = (0, 6, 7, 8, 9, 10, 11)   # I + V1..V6
OUTPUT_INDICES = (1, 6, 7, 8, 9, 10, 11)  # II + V1..V6


def make_features(prediction: np.ndarray, scales: np.ndarray) -> torch.Tensor:
    """Return target-free I/V1..V6 features from a finite B5-final output."""
    p = np.asarray(prediction, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    if p.shape != (N_LEADS, SAMPLES) or scales.shape != (N_LEADS,):
        raise ValueError("Expected B5-final [12,60000] prediction and 12 lead scales")
    if not np.isfinite(p).all() or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Prediction/scales must be finite and scales positive")
    # Median bins suppress beat-scale mismatch; all quantities are available at inference.
    visible = p[list(INPUT_INDICES)]
    x = torch.from_numpy(np.median(visible.reshape(len(INPUT_INDICES), STEPS, BIN), axis=-1)
                         / scales[list(INPUT_INDICES), None])
    x = x[None]
    slow_1s = F.avg_pool1d(F.pad(x, (2, 1), mode="replicate"), 4, 1)
    slow_5s = F.avg_pool1d(F.pad(x, (10, 9), mode="replicate"), 20, 1)
    return torch.cat((x, slow_1s, slow_5s, x - slow_1s, slow_1s - slow_5s), dim=1)[0].contiguous()


class _Block(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Conv1d(width, width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) * 0.3


class TrendCorrector(nn.Module):
    """Predict II/V1..V6 corrections from I/V1..V6, never correcting lead I.

    The remaining four limb leads are always reconstructed from final I and II:
    III=II-I, aVR=-(I+II)/2, aVL=I-II/2, aVF=II-I/2. Both the 4 Hz
    controls and interpolated 500 Hz curve have zero per-record mean.
    """
    def __init__(self, scales_uV: np.ndarray | torch.Tensor, width: int = 64):
        super().__init__()
        scales = torch.as_tensor(scales_uV, dtype=torch.float32)
        if scales.shape != (N_LEADS,):
            raise ValueError("Expected 12 lead scales")
        self.register_buffer("scales", scales[list(OUTPUT_INDICES)])
        self.input = nn.Conv1d(len(INPUT_INDICES) * 5, width, 1)
        self.blocks = nn.Sequential(*(_Block(width, d) for d in (1, 2, 4, 8, 16, 32, 64)))
        self.global_head = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.GELU(), nn.Conv1d(width, len(OUTPUT_INDICES), 1))
        self.local_head = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.GELU(), nn.Conv1d(width, len(OUTPUT_INDICES), 1))
        nn.init.zeros_(self.global_head[-1].weight); nn.init.zeros_(self.global_head[-1].bias)
        nn.init.zeros_(self.local_head[-1].weight); nn.init.zeros_(self.local_head[-1].bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (len(INPUT_INDICES) * 5, STEPS):
            raise ValueError("Expected [B,35,480] I/V1..V6 features")
        h = self.blocks(self.input(features))
        global_curve = F.interpolate(F.avg_pool1d(h, 16), size=STEPS, mode="linear", align_corners=False)
        controls = self.global_head(global_curve) + self.local_head(h)
        controls = controls - controls.mean(dim=-1, keepdim=True)
        full = F.interpolate(controls, size=SAMPLES, mode="linear", align_corners=False)
        full = full - full.mean(dim=-1, keepdim=True)
        return full * self.scales[None, :, None], controls


def compose_d12(anchor_i: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    """Build ordered 12-lead output from I and network-output II/V1..V6."""
    if anchor_i.ndim != 3 or anchor_i.shape[1] != 1 or generated.ndim != 3 or generated.shape[1] != len(OUTPUT_INDICES):
        raise ValueError("Expected I [B,1,T] and II/V1..V6 [B,7,T]")
    if anchor_i.shape[0] != generated.shape[0] or anchor_i.shape[-1] != generated.shape[-1]:
        raise ValueError("I and generated leads must share batch/time dimensions")
    i, ii = anchor_i[:, 0], generated[:, 0]
    return torch.stack((i, ii, ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2,
                        generated[:, 1], generated[:, 2], generated[:, 3], generated[:, 4], generated[:, 5], generated[:, 6]), dim=1)
