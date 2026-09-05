"""Torch loss primitives governed by configs/losses.yaml; no training loop lives here."""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def _lead_weight(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("mask must have [batch, lead] shape for [batch, lead, time] values")
    return mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)


def masked_huber_loss(prediction: torch.Tensor, target: torch.Tensor, lead_mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber reconstruction loss only on selected output leads."""
    weight = _lead_weight(lead_mask, prediction)
    loss = F.huber_loss(prediction, target, delta=delta, reduction="none") * weight
    return loss.sum() / weight.sum().clamp_min(1.0) / prediction.shape[-1]


def masked_pcc_loss(prediction: torch.Tensor, target: torch.Tensor, lead_mask: torch.Tensor) -> torch.Tensor:
    """One minus mean per-window Pearson correlation on selected leads."""
    weight = _lead_weight(lead_mask, prediction).squeeze(-1)
    p = prediction - prediction.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    correlation = (p * t).sum(dim=-1) / (torch.sqrt((p.square().sum(dim=-1) * t.square().sum(dim=-1)).clamp_min(EPS)))
    return 1.0 - (correlation * weight).sum() / weight.sum().clamp_min(1.0)


def observed_consistency_loss(prediction: torch.Tensor, canonical_observed_input: torch.Tensor, lead_mask: torch.Tensor) -> torch.Tensor:
    """Low-weight data consistency on visible leads in common model space."""
    return masked_huber_loss(prediction, canonical_observed_input, lead_mask)


def spectral_stat_loss(prediction: torch.Tensor, strict_train_d12_reference: torch.Tensor) -> torch.Tensor:
    """Compare batch-level spectral mean/std to an independent strict train d12 bank.

    This deliberately accepts a reference bank rather than a row-aligned weak
    target, so raw weak-pair training never performs a per-pair target loss.
    """
    if prediction.ndim != 3 or strict_train_d12_reference.ndim != 3 or prediction.shape[1:] != strict_train_d12_reference.shape[1:]:
        raise ValueError("prediction and strict_train_d12_reference must be [N, 12, T] with matching lead/time dimensions")
    pred_power = torch.log1p(torch.fft.rfft(prediction, dim=-1).abs().square())
    ref_power = torch.log1p(torch.fft.rfft(strict_train_d12_reference, dim=-1).abs().square())
    mean_term = F.smooth_l1_loss(pred_power.mean(dim=0), ref_power.mean(dim=0))
    std_term = F.smooth_l1_loss(pred_power.std(dim=0, unbiased=False), ref_power.std(dim=0, unbiased=False))
    return mean_term + std_term


def pair_invariant_stat_loss(prediction: torch.Tensor, paired_d12_reference: torch.Tensor) -> torch.Tensor:
    """Compare per-window phase-invariant spectrum and amplitude statistics.

    The paired d12 reference supplies subject/window-specific statistics only.
    Fourier magnitudes discard phase, and no time-domain pointwise operation is
    used, so this remains a weak-pair loss rather than target reconstruction.
    """
    if prediction.ndim != 3 or paired_d12_reference.ndim != 3 or prediction.shape != paired_d12_reference.shape:
        raise ValueError("prediction and paired_d12_reference must have matching [batch, 12, time] shapes")
    pred_centered = prediction - prediction.mean(dim=-1, keepdim=True)
    ref_centered = paired_d12_reference - paired_d12_reference.mean(dim=-1, keepdim=True)

    pred_std = pred_centered.std(dim=-1, unbiased=False)
    ref_std = ref_centered.std(dim=-1, unbiased=False)
    pred_abs_mean = pred_centered.abs().mean(dim=-1)
    ref_abs_mean = ref_centered.abs().mean(dim=-1)
    amplitude_term = F.smooth_l1_loss(torch.log1p(pred_std), torch.log1p(ref_std))
    amplitude_term = amplitude_term + F.smooth_l1_loss(torch.log1p(pred_abs_mean), torch.log1p(ref_abs_mean))

    pred_power = torch.log1p(torch.fft.rfft(pred_centered, dim=-1).abs().square())
    ref_power = torch.log1p(torch.fft.rfft(ref_centered, dim=-1).abs().square())
    pred_shape = pred_power / pred_power.mean(dim=-1, keepdim=True).clamp_min(EPS)
    ref_shape = ref_power / ref_power.mean(dim=-1, keepdim=True).clamp_min(EPS)
    spectral_term = F.smooth_l1_loss(pred_shape, ref_shape)
    return amplitude_term + spectral_term


def physiology_constraint_loss(prediction: torch.Tensor) -> torch.Tensor:
    """Apply limb-lead algebraic constraints to a full 12-lead prediction."""
    if prediction.ndim != 3 or prediction.shape[1] != 12:
        raise ValueError("prediction must have shape [batch, 12, time]")
    i, ii, iii, avr, avl, avf = (prediction[:, index] for index in range(6))
    residuals = torch.stack((iii - (ii - i), avr + (i + ii) / 2, avl - (i - ii / 2), avf - (ii - i / 2)), dim=1)
    return residuals.square().mean()