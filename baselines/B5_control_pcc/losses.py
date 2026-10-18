"""Preserve the original objective; C alone adds full-record multiscale shape loss."""
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


def scale_bands(x):
    """Approximate <5s, 5..15s and >15s changes, not ideal frequency bands.

    Input is the complete native-50Hz/120s slow curve, never individual 10s chunks.
    No mean is removed from deployed predictions or absolute-voltage labels.
    """
    five, fifteen = smooth(x, 251), smooth(x, 751)
    return (x - five, five - fifteen, fifteen)


def multiscale_shape_loss(prediction, target, mask, config):
    """Target-derived normalization and train-only floors are used only in the loss.

    Each eligible record/lead/band compares direction and relative amplitude.
    Ignore near-flat labels using frozen training-set thresholds, not validation QC.
    """
    if prediction.shape != target.shape or prediction.shape[1:] != (11, 6000):
        raise ValueError('Multiscale supervision requires [B,11,6000] full-120s slow curves')
    if mask.shape != prediction.shape[:2]:
        raise ValueError('Need per-record, per-lead target QC')
    floors = torch.as_tensor(config['std_floor'], device=prediction.device, dtype=prediction.dtype)
    if floors.shape != (3, 11) or not torch.isfinite(floors).all() or (floors <= 0).any():
        raise ValueError('Need positive frozen TRAIN-only [3,11] standard-deviation floors')
    amplitude_weight = float(config['shape_huber_weight'])
    if amplitude_weight < 0 or not torch.isfinite(torch.tensor(amplitude_weight)):
        raise ValueError('Shape Huber coefficient must be finite and nonnegative')
    valid = mask.bool()
    p = torch.where(valid[..., None], prediction, 0.)
    y = torch.where(valid[..., None], target, 0.)
    losses, metrics = [], {}
    names = ('short_lt5s', 'middle_5to15s', 'long_gt15s')
    for index, (name, pb, yb) in enumerate(zip(names, scale_bands(p), scale_bands(y))):
        # Whole-record centering is ONLY part of the added shape loss.
        pc = pb - pb.mean(-1, keepdim=True)
        yc = yb - yb.mean(-1, keepdim=True)
        std = yc.square().mean(-1).sqrt().detach()
        eligible = valid & (std >= floors[index][None])
        normalizer = torch.maximum(std, floors[index][None])[..., None]
        a, b = pc / normalizer, yc / normalizer
        r = (a * b).mean(-1) / torch.sqrt(a.square().mean(-1) * b.square().mean(-1) + 1e-8)
        shape = F.huber_loss(a, b, reduction='none').mean(-1)
        count = eligible.sum().clamp_min(1)
        pcc = torch.where(eligible, 1 - r, 0.).sum() / count
        huber = torch.where(eligible, shape, 0.).sum() / count
        losses.append(pcc + amplitude_weight * huber)
        metrics[name + '_pcc'] = pcc
        metrics[name + '_shape_huber'] = huber
        metrics[name + '_eligible_fraction'] = eligible.sum() / valid.sum().clamp_min(1)
    return torch.stack(losses).mean(), metrics


def objective(prediction, batch, scales, residual_weight, combo_weight, width=51, beta=1., multiscale=None):
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
    extra = prediction.sum() * 0.
    extra_parts = {}
    coefficient = float(multiscale['weight']) if multiscale is not None else 0.
    if coefficient < 0 or not torch.isfinite(torch.tensor(coefficient)):
        raise ValueError('Multiscale coefficient must be finite and nonnegative')
    if coefficient > 0:
        extra, extra_parts = multiscale_shape_loss(prediction, batch['slow_target'], mask, multiscale)
        total = total + coefficient * extra
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite SlowPCC objective")
    return total, {**parts, "original_F_loss": base, "final_hybrid_pcc": combo,
                   "weighted_hybrid_pcc": combo_weight * combo,
                   'multiscale_loss': extra, 'weighted_multiscale_loss': coefficient * extra, **extra_parts}
