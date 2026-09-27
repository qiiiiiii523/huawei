"""Diffusion schedule and missing-11-only training/sampling for B4."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((points / steps) + offset) / (1 + offset) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class B4Diffusion(nn.Module):
    """Diffuse only the eleven unknown leads; lead I is immutable condition."""

    def __init__(
        self,
        denoiser: nn.Module,
        training_steps: int = 200,
        prediction_type: str = "v",
        clip_denoised: float | None = 6.0,
        dynamic_threshold_quantile: float = 0.995,
        x0_loss_weight: float = 0.1,
        correlation_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if training_steps < 2:
            raise ValueError("training_steps must be at least two")
        if prediction_type not in {"epsilon", "v"}:
            raise ValueError("prediction_type must be 'epsilon' or 'v'")
        if clip_denoised is not None and clip_denoised <= 0:
            raise ValueError("clip_denoised must be positive")
        if not 0 < dynamic_threshold_quantile <= 1:
            raise ValueError("dynamic_threshold_quantile must be in (0,1]")
        self.denoiser = denoiser
        self.training_steps = training_steps
        self.prediction_type = prediction_type
        self.clip_denoised = clip_denoised
        self.dynamic_threshold_quantile = dynamic_threshold_quantile
        self.x0_loss_weight = x0_loss_weight
        self.correlation_loss_weight = correlation_loss_weight
        betas = cosine_beta_schedule(training_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    @property
    def diffusion_config(self) -> dict[str, float | int | str | None]:
        return {
            "training_steps": self.training_steps,
            "prediction_type": self.prediction_type,
            "clip_denoised": self.clip_denoised,
            "dynamic_threshold_quantile": self.dynamic_threshold_quantile,
            "x0_loss_weight": self.x0_loss_weight,
            "correlation_loss_weight": self.correlation_loss_weight,
        }

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return values.gather(0, timesteps).reshape(-1, 1, 1).to(dtype=target.dtype)

    def q_sample(self, clean_missing: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alpha_bars, timesteps, clean_missing) * clean_missing
            + self._extract(self.sqrt_one_minus_alpha_bars, timesteps, clean_missing) * noise
        )

    def _training_target(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return noise
        return (
            self._extract(self.sqrt_alpha_bars, timesteps, clean) * noise
            - self._extract(self.sqrt_one_minus_alpha_bars, timesteps, clean) * clean
        )

    def _model_output_to_clean_noise(
        self,
        noisy: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, timesteps, noisy)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, noisy)
        if self.prediction_type == "v":
            predicted_clean = sqrt_alpha_bar * noisy - sqrt_one_minus * model_output
            predicted_noise = sqrt_one_minus * noisy + sqrt_alpha_bar * model_output
        else:
            predicted_noise = model_output
            predicted_clean = (noisy - sqrt_one_minus * predicted_noise) / sqrt_alpha_bar.clamp_min(1e-8)
        return predicted_clean, predicted_noise

    def _limit_clean(self, predicted_clean: torch.Tensor) -> torch.Tensor:
        if self.clip_denoised is None:
            return predicted_clean
        absolute = predicted_clean.detach().abs().flatten(1)
        quantile = torch.quantile(
            absolute.float(),
            self.dynamic_threshold_quantile,
            dim=1,
        ).to(dtype=predicted_clean.dtype).reshape(-1, 1, 1)
        threshold = quantile.clamp_min(self.clip_denoised)
        scale = threshold / self.clip_denoised
        return predicted_clean.clamp(-threshold, threshold) / scale

    @staticmethod
    def _masked_mean(per_lead_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=per_lead_loss.dtype)
        if not torch.any(weights):
            raise ValueError("B4 batch has no quality-eligible missing leads")
        return (per_lead_loss * weights).sum() / weights.sum()

    @staticmethod
    def _correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction - prediction.mean(dim=2, keepdim=True)
        target = target - target.mean(dim=2, keepdim=True)
        numerator = (prediction * target).sum(dim=2)
        denominator = prediction.square().sum(dim=2).sqrt() * target.square().sum(dim=2).sqrt()
        return 1.0 - numerator / denominator.clamp_min(1e-6)

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
        noisy = self.q_sample(clean_missing, timesteps, noise)
        model_output = self.denoiser(noisy, timesteps, anchor_i)
        target = self._training_target(clean_missing, noise, timesteps)
        diffusion_loss = self._masked_mean((model_output - target).square().mean(dim=2), target_quality_mask)
        if self.x0_loss_weight == 0 and self.correlation_loss_weight == 0:
            return diffusion_loss
        predicted_clean, _ = self._model_output_to_clean_noise(noisy, model_output, timesteps)
        x0_loss = self._masked_mean(
            F.smooth_l1_loss(predicted_clean, clean_missing, reduction="none").mean(dim=2),
            target_quality_mask,
        )
        correlation_loss = self._masked_mean(
            self._correlation_loss(predicted_clean, clean_missing),
            target_quality_mask,
        )
        return (
            diffusion_loss
            + self.x0_loss_weight * x0_loss
            + self.correlation_loss_weight * correlation_loss
        )

    @torch.no_grad()
    def sample(
        self,
        anchor_i: torch.Tensor,
        *,
        sampling_steps: int = 100,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchor_i.ndim != 3 or anchor_i.shape[1] != 1:
            raise ValueError("anchor_i must have shape [B,1,T]")
        if not 1 <= sampling_steps <= self.training_steps:
            raise ValueError("sampling_steps must be within the training schedule")
        batch, _, length = anchor_i.shape
        expected_shape = (batch, 11, length)
        if initial_noise is None:
            current = torch.randn(expected_shape, device=anchor_i.device, dtype=anchor_i.dtype, generator=generator)
        else:
            if initial_noise.shape != expected_shape:
                raise ValueError("initial_noise must have shape [B,11,T]")
            current = initial_noise.to(device=anchor_i.device, dtype=anchor_i.dtype)
        sequence = torch.linspace(self.training_steps - 1, 0, sampling_steps, device=anchor_i.device).round().long()
        sequence = torch.unique_consecutive(sequence)
        for index, timestep in enumerate(sequence):
            timesteps = torch.full((batch,), int(timestep), device=anchor_i.device, dtype=torch.long)
            model_output = self.denoiser(current, timesteps, anchor_i)
            predicted_clean, predicted_noise = self._model_output_to_clean_noise(current, model_output, timesteps)
            predicted_clean = self._limit_clean(predicted_clean)
            sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, timesteps, current)
            sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, current)
            predicted_noise = (current - sqrt_alpha_bar * predicted_clean) / sqrt_one_minus.clamp_min(1e-8)
            if index == len(sequence) - 1:
                current = predicted_clean
                continue
            previous = sequence[index + 1]
            alpha_bar = self.alpha_bars[timestep].to(dtype=current.dtype)
            previous_alpha_bar = self.alpha_bars[previous].to(dtype=current.dtype)
            sigma = eta * (
                ((1 - previous_alpha_bar) / (1 - alpha_bar))
                * (1 - alpha_bar / previous_alpha_bar)
            ).clamp_min(0).sqrt()
            direction = (1 - previous_alpha_bar - sigma.square()).clamp_min(0).sqrt() * predicted_noise
            if eta:
                random_noise = torch.randn(
                    current.shape,
                    device=current.device,
                    dtype=current.dtype,
                    generator=generator,
                )
                stochastic = sigma * random_noise
            else:
                stochastic = 0.0
            current = previous_alpha_bar.sqrt() * predicted_clean + direction + stochastic
        return torch.cat((anchor_i, current), dim=1)
