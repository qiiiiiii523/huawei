"""Preserve legacy S50-F losses; add one full-120s, 500Hz hybrid PCC term."""
import torch
from torch.nn import functional as F
from .core import upsample


def smooth(x, width=51):
    if width < 3 or width % 2 != 1 or width > x.shape[-1]:
        raise ValueError("Need an odd smoothing width in [3, record length]")
    radius = width // 2
    padded = torch.cat((x[..., :1].expand(*x.shape[:-1], radius), x,
                        x[..., -1:].expand(*x.shape[:-1], radius)), dim=-1)
    return F.avg_pool1d(padded, width, stride=1)


def pcc_loss(prediction, target, mask):
    valid = mask.bool()[..., None]
    p, y = torch.where(valid, prediction, 0.), torch.where(valid, target, 0.)
    a, b = p - p.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
    aa, bb = a.square().sum(-1), b.square().sum(-1)
    r = (a * b).sum(-1) / torch.sqrt(aa * bb + 1e-8)
    eligible = mask.bool() & (bb > 1e-8)
    return torch.where(eligible, 1 - r, 0.).sum() / eligible.sum().clamp_min(1)


def old_f_objective(prediction, target, mask, residual_weight, width=51):
    """Matches B5_exp.core.objective + group-F auxiliary_terms, including masking."""
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError("Expected [B,11,6000] and [B,11] mask")
    valid = mask.bool()[..., None]
    p, y = torch.where(valid, prediction, 0.), torch.where(valid, target, 0.)
    denominator = mask.sum().clamp_min(1) * target.shape[-1]
    huber = F.huber_loss(p, y, reduction="sum") / denominator
    pcc = pcc_loss(prediction, target, mask)
    residual = F.huber_loss(p - smooth(p, width), y - smooth(y, width), reduction="sum") / denominator
    return huber + .1 * pcc + residual_weight * residual, {
        "slow_huber": huber, "slow_pcc": pcc, "F_residual": residual}


def objective(prediction, batch, scales, residual_weight, combo_weight, width=51, beta=1.):
    mask = batch["mask"][:, 1:].bool()
    base, parts = old_f_objective(prediction, batch["slow_target"], mask, residual_weight, width)
    fm = batch["fm_uV"][:, 1:]
    if fm.requires_grad:
        raise ValueError("FM must be fixed; use the shared inference-only cache")
    # Cached FM is float32 raw uV; full-record mean uses float64 exactly as deployment.
    fast = fm.double() - smooth(fm.double(), 501)
    learned = upsample(prediction.double() * scales[None, 1:, None].double())
    hybrid = fast + beta * learned
    combo = pcc_loss(hybrid, batch["target_uV"][:, 1:].double(), mask)
    total = base + combo_weight * combo
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite SlowPCC objective")
    return total, {**parts, "original_F_loss": base, "final_hybrid_pcc": combo,
                   "weighted_hybrid_pcc": combo_weight * combo}
