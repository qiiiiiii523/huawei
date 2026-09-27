"""Diffusion schedule and missing-11-only training/sampling for B4."""
from __future__ import annotations

import math

import torch
from torch import nn


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((points / steps) + offset) / (1 + offset) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class B4Diffusion(nn.Module):
    """Diffuse only the eleven unknown leads; lead I is immutable condition."""

    def __init__(self, denoiser: nn.Module, training_steps: int = 200) -> None:
        super().__init__()
        if training_steps < 2:
            raise ValueError("training_steps must be at least two")
        self.denoiser = denoiser
        self.training_steps = training_steps
        betas = cosine_beta_schedule(training_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return values.gather(0, timesteps).reshape(-1, 1, 1).to(dtype=target.dtype)

    def q_sample(self, clean_missing: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alpha_bars, timesteps, clean_missing) * clean_missing
            + self._extract(self.sqrt_one_minus_alpha_bars, timesteps, clean_missing) * noise
        )

    def training_loss(
        self,
        clean_missing: torch.Tensor,
        anchor_i: torch.Tensor,
        target_quality_mask: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = clean_missing.shape[0]
        if clean_missing.ndim != 3 or clean_missing.shape[1] != 11:
            raise ValueError("clean_missing must have shape [B,11,T]")
        if target_quality_mask.shape != clean_missing.shape[:2]:
            raise ValueError("target_quality_mask must have shape [B,11]")
        noise = torch.randn_like(clean_missing) if noise is None else noise
        timesteps = torch.randint(self.training_steps, (batch,), device=clean_missing.device) if timesteps is None else timesteps
        prediction = self.denoiser(self.q_sample(clean_missing, timesteps, noise), timesteps, anchor_i)
        squared_error = (prediction - noise).square().mean(dim=2)
        weights = target_quality_mask.to(dtype=squared_error.dtype)
        if not torch.any(weights):
            raise ValueError("B4 batch has no quality-eligible missing leads")
        return (squared_error * weights).sum() / weights.sum()

    @torch.no_grad()
    def sample(
        self,
        anchor_i: torch.Tensor,
        *,
        sampling_steps: int = 50,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if anchor_i.ndim != 3 or anchor_i.shape[1] != 1:
            raise ValueError("anchor_i must have shape [B,1,T]")
        if not 1 <= sampling_steps <= self.training_steps:
            raise ValueError("sampling_steps must be within the training schedule")
        batch, _, length = anchor_i.shape
        current = torch.randn((batch, 11, length), device=anchor_i.device, dtype=anchor_i.dtype, generator=generator)
        sequence = torch.linspace(self.training_steps - 1, 0, sampling_steps, device=anchor_i.device).round().long()
        sequence = torch.unique_consecutive(sequence)
        for index, timestep in enumerate(sequence):
            timesteps = torch.full((batch,), int(timestep), device=anchor_i.device, dtype=torch.long)
            alpha_bar = self.alpha_bars[timestep].to(dtype=current.dtype)
            predicted_noise = self.denoiser(current, timesteps, anchor_i)
            predicted_clean = (current - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            if index == len(sequence) - 1:
                current = predicted_clean
                continue
            previous = sequence[index + 1]
            previous_alpha_bar = self.alpha_bars[previous].to(dtype=current.dtype)
            sigma = eta * (((1 - previous_alpha_bar) / (1 - alpha_bar)) * (1 - alpha_bar / previous_alpha_bar)).clamp_min(0).sqrt()
            direction = (1 - previous_alpha_bar - sigma.square()).clamp_min(0).sqrt() * predicted_noise
            random_noise = torch.randn(current.shape, device=current.device, dtype=current.dtype, generator=generator)
            current = previous_alpha_bar.sqrt() * predicted_clean + direction + sigma * random_noise
        return torch.cat((anchor_i, current), dim=1)
