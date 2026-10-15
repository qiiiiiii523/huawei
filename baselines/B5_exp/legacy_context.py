"""Frozen original C worktree slicing/cropping, used ONLY by compatibility checks."""
import torch

from .core import POINTS, FACTOR, WINDOW


def contexts_s30_reference(anchor, seconds):
    if seconds not in (10, 30) or anchor.shape[1:] != (1, POINTS // FACTOR):
        raise ValueError('Expected original S10/S30 input')
    radius = WINDOW if seconds == 30 else 0
    if radius:
        anchor = torch.cat((anchor[..., :1].expand(-1, -1, radius), anchor,
                            anchor[..., -1:].expand(-1, -1, radius)), dim=-1)
    value = anchor.unfold(-1, WINDOW + 2 * radius, WINDOW)
    return value.permute(0, 2, 1, 3).reshape(-1, 1, WINDOW + 2 * radius)


def generate_s30_reference(model, batch, seconds=30):
    inputs = {k: batch[k].repeat_interleave(12, dim=0)
              for k in ('numeric', 'sex', 'field_mask', 'age_topcoded')}
    output = model(contexts_s30_reference(batch['anchor'], seconds), **inputs)
    start = WINDOW if seconds == 30 else 0
    center = output[..., start:start + WINDOW]
    return center.reshape(len(batch['anchor']), 12, 11, WINDOW).permute(0, 2, 1, 3).reshape(-1, 11, 12 * WINDOW)
