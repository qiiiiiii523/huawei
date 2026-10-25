"""B7's own serializable configuration; no B5 weights or configuration imports."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int = 10
    embed_dims: tuple[int, int, int] = (128, 256, 384)
    heads: tuple[int, int, int] = (4, 8, 12)
    depths: tuple[int, int, int] = (2, 2, 4)
    window_size: int = 64
    mlp_ratio: int = 4
    time_dim: int = 128
    metadata_dim: int = 128

    def __post_init__(self):
        dimensions = (*self.embed_dims, *self.heads, *self.depths, self.patch_size,
                      self.window_size, self.mlp_ratio, self.time_dim, self.metadata_dim)
        if any(type(v) is not int or v < 1 for v in dimensions):
            raise ValueError("Transformer dimensions must be positive integers")
        if len(self.embed_dims) != 3 or len(self.heads) != 3 or len(self.depths) != 3:
            raise ValueError("Three Transformer scales required")
        if any(d < 4 or d % 2 or d % h for d, h in zip(self.embed_dims, self.heads)):
            raise ValueError("Even embedding widths divisible by attention heads required")
        if self.window_size < 2 or any(d < 2 or d % 2 for d in self.depths[:2]):
            raise ValueError("Local stages need pairs of regular/shifted windows")
        if self.time_dim < 4 or self.time_dim % 2 or self.metadata_dim < 4:
            raise ValueError("Invalid condition dimensions")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("embed_dims", "heads", "depths"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)


@dataclass(frozen=True)
class LossConfig:
    mae_weight: float = 1.0
    pcc_weight: float = 0.1
    ii_weight: float = 5.0
    target_variance_floor: float = 1e-8
    pcc_core_seconds: int = 10
    mae_core_seconds: int = 0  # Zero retains full-window MAE, including legacy runs.

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Loss settings must be finite and nonnegative")
        if self.ii_weight <= 0 or self.target_variance_floor <= 0:
            raise ValueError("II weight and variance floor must be positive")
        if type(self.pcc_core_seconds) is not int or self.pcc_core_seconds < 1:
            raise ValueError("PCC core seconds must be a positive integer")
        if type(self.mae_core_seconds) is not int or self.mae_core_seconds < 0:
            raise ValueError("MAE core seconds must be a nonnegative integer")


@dataclass(frozen=True)
class SamplingConfig:
    sample_rate: int = 500
    window_seconds: int = 40
    core_seconds: int = 10
    solver: str = "heun"
    steps: int = 16
    samples: int = 16
    seed: int = 42

    def __post_init__(self):
        if (self.sample_rate != 500 or type(self.window_seconds) is not int or type(self.core_seconds) is not int
                or (self.window_seconds, self.core_seconds) not in ((10, 10), (40, 10), (40, 20), (75, 75), (120, 120))):
            raise ValueError("B7 uses 500Hz and window/core pairs 10/10, 40/10, 40/20, 75/75 or 120/120")
        if self.solver not in ("heun", "euler") or self.steps < 1 or self.samples < 1:
            raise ValueError("Invalid sampling settings")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed")


def read_json(path: str | Path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)
