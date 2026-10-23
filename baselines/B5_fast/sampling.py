"""Frozen 8 antithetic pairs; the original B5 I condition is never noised."""
from __future__ import annotations

import torch

from baselines.B5.flow import integrate, keyed_noise


def member_noise(keys, length, member_index, device, dtype=torch.float32):
    if type(member_index) is not int or not 0 <= member_index < 16:
        raise ValueError("K16 member index must be in [0,15]")
    # Keep the old B5 seed/key formula for each pair's positive base sample.
    z = keyed_noise(keys, length, 42, member_index // 2, device, dtype)
    return z if member_index % 2 == 0 else -z


@torch.no_grad()
def sample_fast_fm(model, inputs, keys):
    if model.training:
        raise ValueError("Frozen sampling requires eval mode")
    anchor = inputs["anchor"]
    if anchor.shape != (len(keys), 1, 5000) or not torch.isfinite(anchor).all():
        raise ValueError("Original B5 sampling requires clean [B,1,5000] I")
    condition = model.encode_conditions(anchor, inputs["numeric"], inputs["sex"],
                                        inputs["field_mask"], inputs["age_topcoded"])
    total = torch.zeros(len(keys), 11, 5000, device=anchor.device, dtype=anchor.dtype)
    for member in range(16):
        z = member_noise(keys, 5000, member, anchor.device, anchor.dtype)
        total += integrate(model, condition, z, steps=16, solver="heun")
    return torch.cat((condition.anchor_prediction, total / 16), dim=1)
