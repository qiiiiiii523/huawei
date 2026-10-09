"""Independent noise/data linear paths and target-free ODE sampling."""
from __future__ import annotations

import hashlib

import torch

from .model import B5UNet, ConditionCache


def linear_path(target: torch.Tensor, quality_mask: torch.Tensor, noise: torch.Tensor,
                time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim != 3 or target.shape[1] != 11 or noise.shape != target.shape:
        raise ValueError("target and noise must have matching [B,11,T] shapes")
    if quality_mask.shape != target.shape[:2] or time.shape != (target.shape[0],):
        raise ValueError("Invalid path mask/time shape")
    if not torch.isfinite(noise).all() or not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("Noise must be finite and time in [0,1]")
    valid = quality_mask.bool().unsqueeze(-1)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    if not torch.isfinite(safe_target).all():
        raise ValueError("Supervised target values must be finite")
    t = time.reshape(-1, 1, 1)
    state = torch.where(valid, (1 - t) * noise + t * safe_target, noise)
    velocity = torch.where(valid, safe_target - noise, torch.zeros_like(noise))
    return state, velocity


def noise_seed(seed: int, record_key: str, sample_index: int) -> int:
    value = f"B5-v1|{seed}|{record_key}|{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def keyed_noise(keys: list[str], length: int, seed: int, sample_index: int,
                device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if not keys or length < 8:
        raise ValueError("Noise requires nonempty keys and at least 8 samples")
    # CPU generation is stable across CUDA device count, batching and order.
    tensors = []
    for key in keys:
        generator = torch.Generator(device="cpu").manual_seed(noise_seed(seed, key, sample_index))
        tensors.append(torch.randn(11, length, generator=generator, dtype=torch.float32))
    return torch.stack(tensors).to(device=device, dtype=dtype)


@torch.no_grad()
def integrate(model: B5UNet, condition: ConditionCache, noise: torch.Tensor,
              steps: int = 16, solver: str = "heun") -> torch.Tensor:
    if model.training:
        raise ValueError("ODE inference requires model.eval() to disable condition dropout")
    if steps < 1 or solver not in {"heun", "euler"}:
        raise ValueError("Invalid ODE settings")
    state = noise.clone()
    dt = 1.0 / steps
    for index in range(steps):
        time = state.new_full((len(state),), index / steps)
        first = model.velocity(state, time, condition)
        if solver == "euler":
            state = state + dt * first
        else:
            end_time = state.new_full((len(state),), (index + 1) / steps)
            second = model.velocity(state + dt * first, end_time, condition)
            state = state + (dt / 2) * (first + second)
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"Nonfinite ODE state at step {index + 1}; no clipping is applied")
    return state


@torch.no_grad()
def sample(model: B5UNet, inputs: dict[str, torch.Tensor], keys: list[str], seed: int = 42,
           steps: int = 16, solver: str = "heun", samples: int = 1) -> torch.Tensor:
    """Return raw-scaled full d12, without targets or I replacement."""
    if model.training or samples < 1:
        raise ValueError("Sampling requires eval mode and positive sample count")
    anchor = inputs["anchor"]
    if len(keys) != len(anchor):
        raise ValueError("Record keys and input batch disagree")
    condition = model.encode_conditions(anchor, inputs["numeric"], inputs["sex"],
                                        inputs["field_mask"], inputs["age_topcoded"])
    total = torch.zeros(len(anchor), 11, anchor.shape[-1], device=anchor.device, dtype=anchor.dtype)
    for index in range(samples):
        noise = keyed_noise(keys, anchor.shape[-1], seed, index, anchor.device, anchor.dtype)
        total += integrate(model, condition, noise, steps, solver)
    return torch.cat((condition.anchor_prediction, total / samples), dim=1)
