"""Zero-initialized 120-second CNN residual controls on top of frozen Final."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines.B5_best.slow import Block, DILATIONS, SameConv

from . import FIELDS, GROUPS


def visible_inputs(anchor_uV, scale_i, demographics):
    """Only observed I and demographics; 120 seconds at the deployed 50 Hz."""
    raw = np.asarray(anchor_uV, np.float32).reshape(-1)
    if raw.shape != (60000,) or not np.isfinite(raw).all() or not np.isfinite(scale_i) or scale_i <= 0:
        raise ValueError("Need finite 120s observed I at 500Hz and a positive fixed I scale")
    return {"anchor50": (raw.reshape(6000, 10).mean(-1) / scale_i)[None].astype(np.float32),
            **{key: np.asarray(demographics[key]).copy() for key in FIELDS}}


class ResidualSlowCNN(nn.Module):
    """Frozen S50 feature trunk plus a trainable, zero-output 120s CNN branch."""
    def __init__(self, group, channels=48, **ignored):
        super().__init__()
        if group not in GROUPS or channels != 48:
            raise ValueError("Use the matched 48-channel CNN120 residual controls")
        self.group = group
        self.sex = nn.Embedding(3, 4)
        self.metadata = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32))
        self.input = SameConv(1, channels)
        self.blocks = nn.ModuleList(Block(channels, dilation) for dilation in DILATIONS)
        self.output = nn.Conv1d(channels, 11, 1)  # Loaded for the exact S50 state contract; unused.
        self.extra = nn.ModuleList(Block(channels, dilation) for dilation in
                                   ((1, 1) if group == "L120" else (256, 256)))
        self.correction = nn.Conv1d(channels, 11, 1)

    def initialize_from_s50(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        expected = {f"extra.{block}.{part}" for block in range(2) for part in (
            "norm.weight", "norm.bias", "one.conv.weight", "one.conv.bias",
            "two.conv.weight", "two.conv.bias", "film.weight", "film.bias")}
        expected |= {"correction.weight", "correction.bias"}
        if set(missing) != expected or unexpected:
            raise ValueError("S50 state is not compatible with CNN120 residual")
        # Identity-initializing only the added blocks is insufficient because
        # Final stitches S50 windows.  A separate zero head guarantees that the
        # complete residual branch is an exact no-op at epoch 0.
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        for block in self.extra:
            nn.init.zeros_(block.two.conv.weight)
            nn.init.zeros_(block.two.conv.bias)

    def forward(self, anchor50, numeric, sex, field_mask, age_topcoded):
        if anchor50.shape[1:] != (1, 6000):
            raise ValueError("CNN120 residual needs full normalized I")
        vector = torch.cat((numeric, self.sex(sex), field_mask.float(), age_topcoded), dim=-1)
        metadata = self.metadata(vector)
        value = self.input(anchor50)
        for block in (*self.blocks, *self.extra):
            value = block(value, metadata)
        return self.correction(F.silu(value))


def inputs(batch):
    return {key: batch[key] for key in ("anchor50", *FIELDS)}


def correct_base(base_uV, delta_native, scales):
    from baselines.B5_best.combine import upsample_native
    delta = upsample_native(np.asarray(delta_native, np.float64) * np.asarray(scales, np.float64)[1:, None])
    result = np.asarray(base_uV).copy()
    result[1:] = (result[1:].astype(np.float64) + .75 * delta).astype(np.float32)
    return result
