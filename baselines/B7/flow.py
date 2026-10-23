"""Seven-dimensional flow path and ODE; I never enters the random state."""
from __future__ import annotations

import hashlib
import torch
from .config import SamplingConfig


def linear_path(target, valid, noise, time):
    if target.ndim != 3 or target.shape[1] != 7 or noise.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Seven-channel target/noise/point mask required")
    if time.shape != (len(target),) or not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("Invalid flow time")
    if not torch.isfinite(noise).all():
        raise ValueError("Nonfinite noise")
    safe = torch.where(valid, target, torch.zeros_like(target))
    if not torch.isfinite(safe).all():
        raise ValueError("Nonfinite supervised targets")
    t = time[:, None, None]
    state = torch.where(valid, (1 - t) * noise + t * safe, noise)
    return state, torch.where(valid, safe - noise, torch.zeros_like(noise))


def keyed_noise(keys, length, seed, sample_index, device, dtype=torch.float32):
    if not keys or length < 8:
        raise ValueError("Nonempty keys and length>=8 required")
    result = []
    for key in keys:
        raw = f"B7-U7-v1|{seed}|{key}|{sample_index}".encode("utf-8")
        derived = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)
        gen = torch.Generator(device="cpu").manual_seed(derived)
        result.append(torch.randn(7, length, generator=gen))
    return torch.stack(result).to(device=device, dtype=dtype)


@torch.no_grad()
def integrate(model, anchor, noise, metadata=None, *, steps=16, solver="heun"):
    if model.training or steps < 1 or solver not in ("heun", "euler"):
        raise ValueError("ODE requires eval mode and valid settings")
    if noise.ndim != 3 or noise.shape[1] != 7 or anchor.shape != (len(noise), 1, noise.shape[-1]):
        raise ValueError("ODE state has seven channels, I has one")
    state = noise.clone()
    for index in range(steps):
        t = state.new_full((len(state),), index / steps)
        first = model(state, t, anchor, metadata)
        if first.shape != state.shape:
            raise ValueError("Model must return seven velocity channels")
        if solver == "heun":
            second = model(state + first / steps, t.new_full(t.shape, (index + 1) / steps), anchor, metadata)
            if second.shape != state.shape:
                raise ValueError("Model must return seven velocity channels")
            state = state + (first + second) / (2 * steps)
        else:
            state = state + first / steps
        if not torch.isfinite(state).all():
            raise FloatingPointError("Nonfinite ODE output")
    return state


@torch.no_grad()
def sample(model, anchor, keys, metadata=None, config=None):
    config = config or SamplingConfig()
    if len(keys) != len(anchor):
        raise ValueError("Keys and batch disagree")
    total = anchor.new_zeros(len(anchor), 7, anchor.shape[-1])
    for k in range(config.samples):
        noise = keyed_noise(keys, anchor.shape[-1], config.seed, k, anchor.device, anchor.dtype)
        total += integrate(model, anchor, noise, metadata, steps=config.steps, solver=config.solver)
    return total / config.samples
