"""B7's own serializable configuration; no B5 weights or configuration imports."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    channels: tuple[int, ...] = (64, 128, 256)
    time_dim: int = 128
    metadata_dim: int = 128
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    output_head: str = "linear"
    head_channels: int = 64
    coarse_interval_seconds: float = 0.5

    def __post_init__(self):
        if not 3 <= len(self.channels) <= 8 or any(type(c) is not int or c < 4 for c in self.channels):
            raise ValueError("Three to eight scale widths, integer channels >=4 required")
        if self.time_dim < 4 or self.time_dim % 2 or self.metadata_dim < 4:
            raise ValueError("Invalid condition dimensions")
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError("Invalid dilations")
        if self.output_head not in ("linear", "split", "multiscale"):
            raise ValueError("output_head must be linear, split or multiscale")
        if type(self.head_channels) is not int or self.head_channels < 1:
            raise ValueError("head_channels must be a positive integer")
        if not math.isfinite(self.coarse_interval_seconds) or self.coarse_interval_seconds < 1 / 500:
            raise ValueError("coarse_interval_seconds must be at least one 500Hz sample")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("channels", "dilations"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)


@dataclass(frozen=True)
class LossConfig:
    mae_weight: float = 1.0
    pcc_weight: float = 0.1
    ii_weight: float = 5.0
    target_variance_floor: float = 1e-8
    pcc_leads: int = 7
    low_frequency_weight: float = 0.0
    low_frequency_cutoff_hz: float = 0.5

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Loss settings must be finite and nonnegative")
        if self.ii_weight <= 0 or self.target_variance_floor <= 0:
            raise ValueError("II weight and variance floor must be positive")
        if type(self.pcc_leads) is not int or self.pcc_leads not in (7, 11):
            raise ValueError("pcc_leads must be 7 or 11")
        if not 0 < self.low_frequency_cutoff_hz < 250:
            raise ValueError("Low frequency cutoff must lie between 0 and the 250Hz Nyquist frequency")


@dataclass(frozen=True)
class SamplingConfig:
    sample_rate: int = 500
    window_seconds: int = 40
    core_seconds: int = 10
    solver: str = "heun"
    steps: int = 16
    samples: int = 16
    seed: int = 42
    sample_batch_size: int = 1
    validation_batch_size: int = 1

    def __post_init__(self):
        if self.sample_rate != 500 or self.window_seconds not in (10, 40) or self.core_seconds != 10:
            raise ValueError("B7 uses 500Hz, 10s/40s windows and 10s cores")
        if self.solver not in ("heun", "euler") or self.steps < 1 or self.samples < 1:
            raise ValueError("Invalid sampling settings")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed")
        if any(type(v) is not int or v < 1 for v in (self.sample_batch_size, self.validation_batch_size)):
            raise ValueError("Sampling/validation batch sizes must be positive integers")


def read_json(path: str | Path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)
