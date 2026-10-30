"""120-second correction models adapted from the B7 fine/mid architecture.

Only observed I and seven CNN predictions enter model features. Ground truth
is never used by inference. Output order is II, V1..V6 until final assembly.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LEADS7 = ("II", "V1", "V2", "V3", "V4", "V5", "V6")
SAMPLES = 60000


def compose_d12(anchor: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    """Keep I fixed and derive III/aVR/aVL/aVF from corrected II."""
    if anchor.ndim != 3 or anchor.shape[1:] != (1, SAMPLES):
        raise ValueError("anchor must be [batch,1,60000]")
    if generated.shape != (anchor.shape[0], 7, SAMPLES):
        raise ValueError("generated must be [batch,7,60000]")
    ii = generated[:, :1]
    return torch.cat((anchor, ii, ii - anchor, -(anchor + ii) / 2,
                      anchor - ii / 2, ii - anchor / 2, generated[:, 1:]), dim=1)


def make_features(anchor: np.ndarray, seven: np.ndarray, scale8: np.ndarray,
                  input_seconds: float) -> torch.Tensor:
    """B7 median/1s/5s feature stack from I and seven generated leads."""
    if anchor.shape != (1, SAMPLES) or seven.shape != (7, SAMPLES):
        raise ValueError("expected raw-uV I [1,60000] and seven [7,60000]")
    if scale8.shape != (8,) or np.any(scale8 <= 0):
        raise ValueError("scale8 must be positive [8]")
    bins = round(120 / input_seconds)
    samples = round(500 * input_seconds)
    if bins * samples != SAMPLES:
        raise ValueError("input_seconds must partition 120 s exactly")
    raw = np.concatenate((anchor, seven), axis=0).reshape(8, bins, samples)
    med = np.median(raw, axis=-1).astype(np.float32)
    med /= scale8[:, None]
    x = torch.from_numpy(med[None])
    width1, width5 = round(1 / input_seconds), round(5 / input_seconds)
    slow1 = F.avg_pool1d(F.pad(x, (width1 // 2, (width1 - 1) // 2),
                                mode="replicate"), width1, 1)
    slow5 = F.avg_pool1d(F.pad(x, (width5 // 2, (width5 - 1) // 2),
                                mode="replicate"), width5, 1)
    return torch.cat((x, slow1, slow5, x - slow1, slow1 - slow5), dim=1)[0].contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, 3, padding=dilation,
                                           dilation=dilation), nn.GELU(),
                                 nn.Conv1d(width, width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.3 * self.net(x)


class FineCorrector(nn.Module):
    """B7-style 0.1-second seven-lead additive curve, trained from scratch."""
    def __init__(self, scale8: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("scales", torch.as_tensor(scale8[1:], dtype=torch.float32))
        self.input = nn.Conv1d(40, 64, 1)
        self.blocks = nn.Sequential(*(ResidualBlock(64, dilation) for dilation in
                                      (2, 5, 10, 20, 40, 80, 160)))
        self.global_head = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1),
                                         nn.GELU(), nn.Conv1d(64, 7, 1))
        self.local_head = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1),
                                        nn.GELU(), nn.Conv1d(64, 7, 1))
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (40, 1200):
            raise ValueError("fine features must be [batch,40,1200]")
        h = self.blocks(self.input(features))
        global_curve = F.interpolate(F.avg_pool1d(h, 40), size=1200,
                                     mode="linear", align_corners=False)
        controls = self.global_head(global_curve) + self.local_head(h)
        full = F.interpolate(controls, size=SAMPLES, mode="linear", align_corners=False)
        return full * self.scales[None, :, None], controls


def gaussian_kernel(sigma_bins: float) -> torch.Tensor:
    radius = round(3 * sigma_bins)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma_bins) ** 2)
    return (kernel / kernel.sum())[None, None]


def gaussian_torch(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    radius = (kernel.shape[-1] - 1) // 2
    return F.conv1d(F.pad(x, (radius, radius), mode="replicate"),
                    kernel.repeat(x.shape[1], 1, 1), groups=x.shape[1])


class ChestMidCorrector(nn.Module):
    """B7-style 4-Hz mid-scale residual for V1..V6 only."""
    def __init__(self, scale8: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("scales", torch.as_tensor(scale8[2:], dtype=torch.float32))
        self.register_buffer("mid_kernel", gaussian_kernel(10 / 0.25))
        self.input = nn.Conv1d(112, 64, 1)
        self.blocks = nn.Sequential(*(ResidualBlock(64, d) for d in
                                      (1, 2, 4, 8, 16, 32, 64, 128)))
        self.mid_head = nn.Conv1d(64, 6, 1)
        nn.init.zeros_(self.mid_head.weight)
        nn.init.zeros_(self.mid_head.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (112, 480):
            raise ValueError("mid features must be [batch,112,480]")
        raw = self.mid_head(self.blocks(self.input(features)))
        controls = raw - gaussian_torch(raw, self.mid_kernel)
        curve = F.interpolate(controls, size=SAMPLES, mode="linear", align_corners=False)
        return curve * self.scales[None, :, None], controls


def mid_features(anchor: np.ndarray, base7: np.ndarray, fine7: np.ndarray,
                 scale8: np.ndarray) -> torch.Tensor:
    """B7 original+fine 112-channel features at 4 Hz."""
    original = make_features(anchor, base7, scale8, 0.25)
    fixed = make_features(anchor, fine7, scale8, 0.25)
    x = fixed[0:8][None]

    def average(seconds: int) -> torch.Tensor:
        width = round(seconds / 0.25)
        return F.avg_pool1d(F.pad(x, (width // 2, (width - 1) // 2),
                                  mode="replicate"), width, 1)[0]

    slow5, slow15, slow30 = fixed[16:24], average(15), average(30)
    return torch.cat((original, fixed, slow15, slow30,
                      slow5 - slow15, slow15 - slow30), dim=0).contiguous()
