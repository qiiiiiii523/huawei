"""Direct-regression losses obeying B7's QC masks and II emphasis."""
from __future__ import annotations
import torch

def regression_loss(prediction, target, lead_mask, time_mask, *, mae_weight=1., pcc_weight=.3, ii_weight=5., variance_floor=1e-8):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 7:
        raise ValueError("B8 prediction and target must be [B,7,5000]")
    valid = lead_mask.bool()[:, :, None] & time_mask.bool()[:, None, :]
    weights = prediction.new_ones(1, 7, 1); weights[:, 0] = ii_weight
    delta = torch.where(valid, prediction - target, torch.zeros_like(prediction))
    mae = (delta.abs() * weights).sum() / (valid * weights).sum().clamp_min(1)
    # B7-compatible seven-lead reconstruction loss. Full missing11 selection is
    # computed after physical d12 composition in evaluate.py.
    p, y, m = prediction, target, valid
    n = m.sum(-1).clamp_min(1); p = torch.where(m, p, 0.); y = torch.where(m, y, 0.)
    p = torch.where(m, p - p.sum(-1, keepdim=True) / n[..., None], 0.)
    y = torch.where(m, y - y.sum(-1, keepdim=True) / n[..., None], 0.)
    yy = y.square().sum(-1); eligible = (m.sum(-1) >= 2) & (yy / n > variance_floor)
    corr = (p * y).sum(-1) / torch.sqrt((p.square().sum(-1) * yy).clamp_min(1e-12))
    pcc = ((1 - corr.clamp(-1, 1)) * eligible).sum() / eligible.sum().clamp_min(1)
    total = mae_weight * mae + pcc_weight * pcc
    return total, {"mae": mae, "pcc_missing11": pcc, "pcc_valid_leads": eligible.sum().to(prediction.dtype)}
