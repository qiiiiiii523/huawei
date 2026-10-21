"""Full-record SlowCNN controls: matched capacity, only long dilation differs."""
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


class FullSlowCNN(nn.Module):
    """P-S50 plus two identity-initialized blocks, evaluated once over 120 seconds."""
    def __init__(self, group, channels=48, **ignored):
        super().__init__()
        if group not in GROUPS or channels != 48:
            raise ValueError("Use the matched 48-channel CNN120 controls")
        self.group = group
        self.sex = nn.Embedding(3, 4)
        self.metadata = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32))
        self.input = SameConv(1, channels)
        self.blocks = nn.ModuleList(Block(channels, dilation) for dilation in DILATIONS)
        self.extra = nn.ModuleList(Block(channels, dilation) for dilation in
                                   ((1, 1) if group == "L120" else (256, 256)))
        self.output = nn.Conv1d(channels, 11, 1)

    def initialize_from_s50(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        expected = {f"extra.{block}.{part}" for block in range(2) for part in (
            "norm.weight", "norm.bias", "one.conv.weight", "one.conv.bias",
            "two.conv.weight", "two.conv.bias", "film.weight", "film.bias")}
        if set(missing) != expected or unexpected:
            raise ValueError("S50 state is not compatible with CNN120")
        # Each residual extension starts as the exact identity regardless of dilation.
        for block in self.extra:
            nn.init.zeros_(block.two.conv.weight)
            nn.init.zeros_(block.two.conv.bias)

    def forward(self, anchor50, numeric, sex, field_mask, age_topcoded, base_slow):
        if anchor50.shape[1:] != (1, 6000) or base_slow.shape[1:] != (11, 6000):
            raise ValueError("CNN120 needs full normalized I and original P native slow curves")
        vector = torch.cat((numeric, self.sex(sex), field_mask.float(), age_topcoded), dim=-1)
        metadata = self.metadata(vector)
        value = self.input(anchor50)
        for block in (*self.blocks, *self.extra):
            value = block(value, metadata)
        # Return a delta so the established Final combination remains unchanged.
        return self.output(F.silu(value)) - base_slow


def inputs(batch):
    return {key: batch[key] for key in ("anchor50", "base_slow", *FIELDS)}


def correct_base(base_uV, delta_native, scales):
    from baselines.B5_best.combine import upsample_native
    delta = upsample_native(np.asarray(delta_native, np.float64) * np.asarray(scales, np.float64)[1:, None])
    result = np.asarray(base_uV).copy()
    result[1:] = (result[1:].astype(np.float64) + .75 * delta).astype(np.float32)
    return result
