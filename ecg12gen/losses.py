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

def replace_output_i_with_anchor(prediction: torch.Tensor, anchor_i: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 3 or prediction.shape[1] != 12 or anchor_i.shape != prediction[:, :1].shape:
        raise ValueError('prediction must be [batch,12,time] and anchor_i [batch,1,time]')
    output = prediction.clone()
    output[:, :1] = anchor_i
    return output

def anchored_weighted_limb_physiology_loss(
    prediction: torch.Tensor,
    anchor_i: torch.Tensor,
    d12_scale_uV: torch.Tensor,
    relation_weights: tuple[float, float, float, float] = (2.0, 0.5, 1.0, 1.0),
) -> torch.Tensor:
    """Soft limb-lead algebra loss conditioned on the supplied machine-I anchor."""
    scale = torch.as_tensor(d12_scale_uV, dtype=prediction.dtype, device=prediction.device)
    weights = torch.as_tensor(relation_weights, dtype=prediction.dtype, device=prediction.device)
    if prediction.ndim != 3 or prediction.shape[1] != 12 or anchor_i.shape != prediction[:, :1].shape:
        raise ValueError('prediction must be [B,12,T] and anchor_i must be [B,1,T]')
    if scale.ndim != 1 or scale.shape[0] != 12 or not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError('d12_scale_uV must be finite [12] with positive values')
    if weights.shape != (4,) or not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError('relation_weights must contain four finite positive values')
    constrained_prediction = prediction.clone()
    constrained_prediction[:, :1] = anchor_i
    morphology = constrained_prediction * scale.view(1, 12, 1)
    i, ii, iii, avr, avl, avf = (morphology[:, index] for index in range(6))
    residuals = torch.stack((iii - (ii - i), avr + (i + ii) / 2, avl - (i - ii / 2), avf - (ii - i / 2)), dim=1)
    residuals = residuals - residuals.median(dim=-1, keepdim=True).values
    residual_scale = scale[torch.tensor([2, 3, 4, 5], device=prediction.device)].view(1, 4, 1)
    per_relation = (residuals / residual_scale).square().mean(dim=(0, 2))
    return (per_relation * weights).sum() / weights.sum()

def strict_anchor_pretrain_loss(prediction: torch.Tensor, target: torch.Tensor, anchor_i: torch.Tensor, *, huber_weight: float = 1.0, pcc_weight: float = 0.1, physiology_weight: float = 0.10, observed_weight: float = 0.02, d12_scale_uV: torch.Tensor | None = None) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 12:
        raise ValueError('prediction and target must be matching [batch,12,time]')
    if anchor_i.shape != prediction[:, :1].shape:
        raise ValueError('anchor_i must be [batch,1,time]')
    all_leads = torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device)
    anchor_mask = torch.zeros_like(all_leads); anchor_mask[:, 0] = True
    physiology = (anchored_weighted_limb_physiology_loss(prediction, anchor_i, d12_scale_uV)
                   if physiology_weight and d12_scale_uV is not None else prediction.new_zeros(()))
    return (huber_weight * masked_huber_loss(prediction, target, all_leads) + pcc_weight * masked_pcc_loss(prediction, target, all_leads) + physiology_weight * physiology + observed_weight * observed_consistency_loss(prediction, anchor_i.expand_as(prediction), anchor_mask))

def joint_anchor_sync_loss(prediction: torch.Tensor, target: torch.Tensor, anchor_i: torch.Tensor, *, huber_weight: float = 1.0, pcc_weight: float = 0.1, physiology_weight: float = 0.10, observed_weight: float = 0.02, d12_scale_uV: torch.Tensor | None = None) -> torch.Tensor:
    return strict_anchor_pretrain_loss(prediction, target, anchor_i, huber_weight=huber_weight, pcc_weight=pcc_weight, physiology_weight=physiology_weight, observed_weight=observed_weight, d12_scale_uV=d12_scale_uV)
