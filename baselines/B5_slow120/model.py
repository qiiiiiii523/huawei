"""Identical weights/parameter count; only visible temporal context differs."""
import math
import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn

from . import FIELDS, GROUPS


def visible_inputs(anchor_uV, scale_i, demographics):
    """RAW observed I only. Filter each local window separately to prevent leakage."""
    raw = np.asarray(anchor_uV, np.float64).reshape(-1)
    if raw.shape != (60000,) or not np.isfinite(raw).all() or not np.isfinite(scale_i) or scale_i <= 0:
        raise ValueError("Need finite 120s observed I at 500Hz and a positive fixed I scale")
    raw = raw / scale_i
    def down(x):
        # scipy polyphase FIR: default Kaiser beta5, constant edge extension.
        return resample_poly(x, 1, 100, padtype="line").astype(np.float32)
    # Edge REPEATS are applied before choosing a 50s context. Never filter the
    # entire record first and then crop: that would leak samples outside G50.
    padded = np.pad(raw, (10000, 10000), mode="edge")
    local = np.stack([down(padded[s:s + 25000]) for s in range(0, 60000, 5000)])[:, None]
    return {"global_i": down(raw)[None], "local_i": local,
            **{k: np.asarray(demographics[k]).copy() for k in FIELDS}}


def tokens_to_native(value, local=False):
    """Linear interpolation at the original P block centers, deterministic backward."""
    left = value[..., 100:150] if local else value
    right = value[..., 101:151] if local else torch.cat((value[..., 1:], value[..., -1:]), -1)
    weight = (torch.arange(10, device=value.device, dtype=value.dtype) * 10 + 4.5) / 100
    return (left[..., None] * (1 - weight) + right[..., None] * weight).flatten(-2)


class AttentionBlock(nn.Module):
    def __init__(self, channels=32, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(nn.Linear(channels, 2 * channels), nn.SiLU(), nn.Linear(2 * channels, channels))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attention = (q @ k.transpose(-2, -1) / math.sqrt(c // self.heads)).softmax(-1)
        mixed = (attention @ v).transpose(1, 2).reshape(b, t, c)
        x = x + self.projection(mixed)
        return x + self.ff(self.norm2(x))


class SlowCorrection(nn.Module):
    """No original CNN/FM parameters inside this module. 5Hz -> 50Hz correction."""
    def __init__(self, group, channels=32, layers=2, heads=4):
        super().__init__()
        if group not in GROUPS or channels % heads or channels % 2 or layers < 1:
            raise ValueError("Invalid matched correction configuration")
        self.group, self.channels = group, channels
        self.sex = nn.Embedding(3, 4)
        self.metadata = nn.Sequential(nn.Linear(12, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.input = nn.Linear(1, channels)
        self.blocks = nn.ModuleList([AttentionBlock(channels, heads) for _ in range(layers)])
        self.output = nn.Linear(channels, 11)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, global_i, local_i, numeric, sex, field_mask, age_topcoded):
        b = len(numeric)
        if global_i.shape != (b, 1, 600) or local_i.shape != (b, 12, 1, 250):
            raise ValueError("Need full120 [B,1,600] and local50 [B,12,1,250] visible I")
        meta = torch.cat((numeric, self.sex(sex), field_mask.float(), age_topcoded), -1)
        meta = self.metadata(meta)
        signal = global_i if self.group == "G120" else local_i.reshape(b * 12, 1, 250)
        if self.group == "G50":
            meta = meta.repeat_interleave(12, 0)
        x = self.input(signal.transpose(1, 2)) + meta[:, None]
        position = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)[:, None]
        frequency = torch.exp(torch.arange(0, self.channels, 2, dtype=x.dtype, device=x.device)
                              * (-math.log(10000.) / self.channels))
        angle = position * frequency
        pe = torch.stack((angle.sin(), angle.cos()), -1).reshape(x.shape[1], self.channels)
        x = x + pe[None]
        for block in self.blocks:
            x = block(x)
        tokens = self.output(x).transpose(1, 2)
        # scipy resample output token i is at raw index 100*i. The original P
        # native output uses 10-point block centers: raw index 10*i+4.5.
        delta = tokens_to_native(tokens, local=self.group == "G50")
        if self.group == "G50":
            delta = delta.reshape(b, 12, 11, 500).permute(0, 2, 1, 3).reshape(b, 11, 6000)
        return delta


def inputs(batch):
    return {k: batch[k] for k in ("global_i", "local_i", *FIELDS)}


def upsample_native(value):
    """Differentiable exact block-center interpolation, matching B5-best."""
    if value.shape[-1] != 6000:
        raise ValueError("Expected complete native-50Hz curve")
    weight = (torch.arange(10, device=value.device, dtype=value.dtype) + .5) / 10
    middle = value[..., :-1, None] * (1 - weight) + value[..., 1:, None] * weight
    return torch.cat((value[..., :1].expand(*value.shape[:-1], 5), middle.flatten(-2),
                      value[..., -1:].expand(*value.shape[:-1], 5)), -1)


def correct_base(base_uV, delta_native, scales):
    from baselines.B5_best.combine import upsample_native as numpy_upsample
    delta = numpy_upsample(np.asarray(delta_native, np.float64) * np.asarray(scales, np.float64)[1:, None])
    result = np.asarray(base_uV).copy()
    result[1:] = (result[1:].astype(np.float64) + .75 * delta).astype(np.float32)
    return result
