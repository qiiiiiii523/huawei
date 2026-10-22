"""Keep G120's parameterization and physical time coordinates across rates."""
import math
import numpy as np
from scipy.signal import resample_poly
import torch

from baselines.B5_slow120 import FIELDS
from baselines.B5_slow120.model import SlowCorrection
from . import RATES


def visible_inputs(anchor_uV, scale_i, demographics, group):
    raw = np.asarray(anchor_uV, np.float64).reshape(-1)
    if raw.shape != (60000,) or not np.isfinite(raw).all() or not np.isfinite(scale_i) or scale_i <= 0:
        raise ValueError("Need finite raw 120s observed I at 500Hz and positive fixed I scale")
    # Each arm starts from RAW I, never from another arm's downsampled signal.
    factor = int(500 / RATES[group])
    signal = resample_poly(raw / scale_i, 1, factor, padtype="line").astype(np.float32)
    return {"global_i": signal[None], **{k: np.asarray(demographics[k]).copy() for k in FIELDS}}


def positions(length, hz, *, dtype, device):
    # Units are original G120 0.2-second ticks. Thus equal physical times
    # receive equal positions; 5Hz exactly retains the historical encoding.
    return torch.arange(length, dtype=dtype, device=device)[:, None] * (5. / hz)


def tokens_to_native(value, hz):
    factor = int(50 / hz)
    if value.shape[-1] != int(120 * hz):
        raise ValueError("Expected complete 120-second tokens")
    right = torch.cat((value[..., 1:], value[..., -1:]), -1)
    # P native bin j is centered at raw-500Hz index 10*j+4.5.
    weight = (torch.arange(factor, device=value.device, dtype=value.dtype) * 10 + 4.5) / (500. / hz)
    return (value[..., None] * (1 - weight) + right[..., None] * weight).flatten(-2)


class RateCorrection(SlowCorrection):
    def __init__(self, group, channels=32, layers=2, heads=4):
        if group not in RATES:
            raise ValueError("Unknown rate arm")
        super().__init__("G120", channels, layers, heads)
        self.group, self.hz = group, RATES[group]

    def forward(self, global_i, numeric, sex, field_mask, age_topcoded):
        if global_i.shape != (len(numeric), 1, int(120 * self.hz)):
            raise ValueError("Need full120 observed I at this arm's sampling rate")
        meta = self.metadata(torch.cat((numeric, self.sex(sex), field_mask.float(), age_topcoded), -1))
        x = self.input(global_i.transpose(1, 2)) + meta[:, None]
        position = positions(x.shape[1], self.hz, dtype=x.dtype, device=x.device)
        frequency = torch.exp(torch.arange(0, self.channels, 2, dtype=x.dtype, device=x.device)
                              * (-math.log(10000.) / self.channels))
        angle = position * frequency
        pe = torch.stack((angle.sin(), angle.cos()), -1).reshape(x.shape[1], self.channels)
        x = x + pe[None]
        for block in self.blocks:
            x = block(x)
        return tokens_to_native(self.output(x).transpose(1, 2), self.hz)


def inputs(batch):
    return {k: batch[k] for k in ("global_i", *FIELDS)}


def array_batch(data, device):
    return {k: torch.from_numpy(np.asarray(data[k])[None].copy()).to(device) for k in ("global_i", *FIELDS)}
