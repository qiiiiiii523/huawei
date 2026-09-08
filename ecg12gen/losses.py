"""Torch loss primitives governed by configs/losses.yaml; no training loop lives here."""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def replace_output_i_with_anchor(prediction: torch.Tensor, anchor_i: torch.Tensor) -> torch.Tensor:
    """Return a d12 prediction whose observed I is exactly the test-time anchor."""
    if prediction.ndim != 3 or prediction.shape[1] != 12 or anchor_i.shape != prediction[:, :1].shape:
        raise ValueError("prediction must be [batch,12,time] and anchor_i [batch,1,time]")
    output = prediction.clone()
    output[:, :1] = anchor_i
    return output


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


def strict_anchor_pretrain_loss(prediction: torch.Tensor, target: torch.Tensor,
                                anchor_i: torch.Tensor, *, huber_weight: float = 1.0,
                                pcc_weight: float = 0.1, physiology_weight: float = 0.05,
                                observed_weight: float = 0.02,
                                d12_scale_uV: torch.Tensor | None = None) -> torch.Tensor:
    """Strict same-window machine-I -> d12 loss; all twelve leads are trained."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 12:
        raise ValueError("prediction and target must be matching [batch,12,time]")
    if anchor_i.shape != prediction[:, :1].shape:
        raise ValueError("anchor_i must be [batch,1,time]")
    all_leads = torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device)
    anchor_mask = torch.zeros_like(all_leads); anchor_mask[:, 0] = True
    physiology = (physiology_constraint_loss(prediction, d12_scale_uV)
                  if physiology_weight else prediction.new_zeros(()))
    return (huber_weight * masked_huber_loss(prediction, target, all_leads) +
            pcc_weight * masked_pcc_loss(prediction, target, all_leads) +
            physiology_weight * physiology +
            observed_weight * observed_consistency_loss(prediction, anchor_i.expand_as(prediction), anchor_mask))


def joint_anchor_sync_loss(prediction: torch.Tensor, target: torch.Tensor,
                           anchor_i: torch.Tensor, *, huber_weight: float = 1.0,
                           pcc_weight: float = 0.1, physiology_weight: float = 0.05,
                           observed_weight: float = 0.02,
                           d12_scale_uV: torch.Tensor | None = None) -> torch.Tensor:
    """Joint-anchor loss; the model predicts and is supervised on all d12 leads.

    Full-d12 pointwise supervision is legal because the input includes the
    same-window target-time I anchor.  The cross-time context is conditioning
    only and is never compared pointwise with the target.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 12:
        raise ValueError("prediction and target must be matching [batch,12,time]")
    if anchor_i.shape != prediction[:, :1].shape:
        raise ValueError("anchor_i must be [batch,1,time]")
    all_leads = torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device)
    anchor_mask = torch.zeros_like(all_leads); anchor_mask[:, 0] = True
    physiology = (physiology_constraint_loss(prediction, d12_scale_uV)
                  if physiology_weight else prediction.new_zeros(()))
    return (huber_weight * masked_huber_loss(prediction, target, all_leads) +
            pcc_weight * masked_pcc_loss(prediction, target, all_leads) +
            physiology_weight * physiology +
            observed_weight * observed_consistency_loss(prediction, anchor_i.expand_as(prediction), anchor_mask))


def physiology_constraint_loss(prediction: torch.Tensor,
                               d12_scale_uV: torch.Tensor | None) -> torch.Tensor:
    """Apply scale-aware, baseline-invariant limb-lead constraints.

    ``prediction`` is the centered/scaled d12 model view.  The limb-lead
    equations are defined in microvolts, so each lead is first restored with
    its frozen d12 scale.  Because preprocessing removes an independent
    median from every lead, each algebraic residual may contain a constant
    window offset; removing that residual median keeps the constraint focused
    on morphology rather than an unavailable baseline.
    """
    if prediction.ndim != 3 or prediction.shape[1] != 12:
        raise ValueError("prediction must have shape [batch, 12, time]")
    if d12_scale_uV is None:
        raise ValueError("d12_scale_uV is required for the physiology constraint")
    scale = torch.as_tensor(d12_scale_uV, dtype=prediction.dtype, device=prediction.device)
    if scale.ndim != 1 or scale.shape[0] != 12 or not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("d12_scale_uV must be finite and have shape [12]")
    morphology_uV = prediction * scale.view(1, 12, 1)
    i, ii, iii, avr, avl, avf = (morphology_uV[:, index] for index in range(6))
    residuals = torch.stack((iii - (ii - i), avr + (i + ii) / 2, avl - (i - ii / 2), avf - (ii - i / 2)), dim=1)
    residuals = residuals - residuals.median(dim=-1, keepdim=True).values
    residual_scale = scale[torch.tensor([2, 3, 4, 5], device=prediction.device)].view(1, 4, 1)
    return (residuals / residual_scale).square().mean()
