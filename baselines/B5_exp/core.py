from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import PROTOCOL

LEADS = ('II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
FS, FACTOR, WIDTH, POINTS, WINDOW = 500, 10, 501, 60000, 500
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def json_write(path, value):
    def safe(x):
        if isinstance(x, dict):
            return {str(k): safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [safe(v) for v in x]
        if isinstance(x, np.ndarray):
            return safe(x.tolist())
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return float(x) if math.isfinite(x) else None
        return x
    Path(path).write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slow(x):
    """Same full-record, centered 501-point edge-replicated filter as diagnostics."""
    x = np.asarray(x, dtype=np.float64)
    pad = np.pad(x, ((0, 0), (WIDTH // 2, WIDTH // 2)), mode='edge')
    acc = np.concatenate((np.zeros((len(x), 1)), np.cumsum(pad, axis=1)), axis=1)
    return (acc[:, WIDTH:] - acc[:, :-WIDTH]) / WIDTH


def downsample(x):
    """Fixed boxcar anti-alias prefilter + decimation; same in both groups."""
    return np.asarray(x).reshape(*x.shape[:-1], -1, FACTOR).mean(axis=-1)


def upsample(x):
    # Restore block-center coordinates across the WHOLE record, not each window.
    centers = np.arange(x.shape[-1]) * FACTOR + (FACTOR - 1) / 2
    return np.stack([np.interp(np.arange(POINTS), centers, row) for row in x])


def correlations(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    a, b = p - p.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.divide((a * b).sum(axis=-1), den, out=np.full(den.shape, np.nan), where=den > 0)


def macro(values, mask=None):
    values = np.asarray(values, float)
    if mask is not None:
        values = np.where(mask, values, np.nan)
    means = [float(values[:, i][np.isfinite(values[:, i])].mean())
             for i in range(values.shape[1]) if np.isfinite(values[:, i]).any()]
    return float(np.mean(means)) if means else float('nan')


def context_radius(seconds):
    if seconds not in (10, 30, 50):
        raise ValueError('Expected context seconds 10, 30 or 50')
    return (seconds // 10 - 1) // 2 * WINDOW


def contexts(anchor, seconds):
    """Input I only: [B,1,6000] -> [B*12,1,500/1500/2500] at 50Hz."""
    radius = context_radius(seconds)
    if anchor.ndim != 3 or anchor.shape[1:] != (1, POINTS // FACTOR):
        raise ValueError('Expected a complete 120-second input I [B,1,6000]')
    if radius:
        anchor = torch.cat((anchor[..., :1].expand(-1, -1, radius), anchor,
                            anchor[..., -1:].expand(-1, -1, radius)), dim=-1)
    value = anchor.unfold(-1, WINDOW + 2 * radius, WINDOW)
    return value.permute(0, 2, 1, 3).reshape(-1, 1, WINDOW + 2 * radius)


class SameConv(nn.Module):
    def __init__(self, incoming, outgoing, dilation=1):
        super().__init__()
        self.radius = 2 * dilation
        self.conv = nn.Conv1d(incoming, outgoing, 5, dilation=dilation)

    def forward(self, x):
        # Explicit repeats have deterministic backward; no reflection across patients.
        r = self.radius
        return self.conv(torch.cat((x[..., :1].expand(-1, -1, r), x,
                                    x[..., -1:].expand(-1, -1, r)), dim=-1))


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(channels)  # Channel-only, not per-window time centering.
        self.one, self.two = SameConv(channels, channels, dilation), SameConv(channels, channels, dilation)
        self.film = nn.Linear(32, 2 * channels)

    def forward(self, x, metadata):
        a, b = self.film(metadata).chunk(2, dim=-1)
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = h * (1 + a[..., None]) + b[..., None]
        return x + self.two(F.silu(self.one(F.silu(h))))


class SlowCNN(nn.Module):
    """Same 48-channel network for all contexts; nominal RF 40.9s at 50Hz."""
    def __init__(self, channels=48):
        super().__init__()
        self.sex = nn.Embedding(3, 4)
        self.metadata = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32))
        self.input = SameConv(1, channels)
        self.blocks = nn.ModuleList(Block(channels, d) for d in DILATIONS)
        self.output = nn.Conv1d(channels, 11, 1)

    def forward(self, anchor, numeric, sex, field_mask, age_topcoded):
        if anchor.ndim != 3 or anchor.shape[1] != 1:
            raise ValueError('Model accepts only I plus demographics, never target curves')
        v = torch.cat((numeric, self.sex(sex), field_mask.float(), age_topcoded), dim=-1)
        meta = self.metadata(v)
        x = self.input(anchor)
        for block in self.blocks:
            x = block(x, meta)
        return self.output(F.silu(x))


def generate(model, batch, seconds):
    # Same number of center-window outputs and forward calls for either context size.
    inputs = {k: batch[k].repeat_interleave(12, dim=0)
              for k in ('numeric', 'sex', 'field_mask', 'age_topcoded')}
    output = model(contexts(batch['anchor'], seconds), **inputs)
    start = context_radius(seconds)
    center = output[..., start:start + WINDOW]
    return center.reshape(len(batch['anchor']), 12, 11, WINDOW).permute(0, 2, 1, 3).reshape(-1, 11, 12 * WINDOW)


def objective(p, y, mask):
    """Huber keeps absolute voltage; PCC uses whole 120s, never window means."""
    valid = mask[..., None]
    y = torch.where(valid, y, torch.zeros_like(y))
    p_safe = torch.where(valid, p, torch.zeros_like(p))
    huber = F.huber_loss(p_safe, y, reduction='none').sum() / (mask.sum().clamp_min(1) * y.shape[-1])
    a, b = p_safe - p_safe.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
    aa, bb = a.square().sum(-1), b.square().sum(-1)
    r = (a * b).sum(-1) / torch.sqrt(aa * bb + 1e-8)
    eligible = mask & (bb > 1e-8)
    pcc = torch.where(eligible, 1 - r, 0.).sum() / eligible.sum().clamp_min(1)
    total = huber + .1 * pcc
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite probe loss')
    return total, {'huber': float(huber.detach()), 'pcc': float(pcc.detach())}


class Records(torch.utils.data.Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {k: archive[k].copy() for k in archive.files}

    def __len__(self):
        return len(self.arrays['anchor'])

    def __getitem__(self, i):
        return {key: torch.from_numpy(np.asarray(value[i]).copy()) for key, value in self.arrays.items()}


def device_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def load_prepared(path):
    path = Path(path).resolve()
    info = json.loads((path / 'prepared.json').read_text(encoding='utf-8'))
    if info['protocol'] != PROTOCOL or not info['completed']:
        raise ValueError('Incomplete or incompatible prepared data')
    for name, expected in info['files'].items():
        if sha(path / name) != expected:
            raise ValueError(f'Prepared file changed: {name}')
    return path, info
