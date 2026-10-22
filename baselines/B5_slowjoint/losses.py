"""Supervise the sum, not two copies of the complete slow target."""
import torch
from torch.nn import functional as F
from baselines.B5_slow120.losses import objective as original_objective, pcc_loss, smooth, F_WEIGHT


def objective(prediction, batch, scales, public=False):
    if not public:
        # The frozen base includes the ORIGINAL P. Replace it by the new total P+G.
        return original_objective(prediction - batch["base_slow"], batch, scales)
    mask = batch["mask"][:, 1:].bool()
    valid = mask[..., None]
    a = torch.where(valid, prediction, 0.)
    b = torch.where(valid, batch["slow_target"], 0.)
    denominator = mask.sum().clamp_min(1) * b.shape[-1]
    huber = F.huber_loss(a, b, reduction="sum") / denominator
    pcc = pcc_loss(a, b, mask)
    residual = F.huber_loss(a - smooth(a), b - smooth(b), reduction="sum") / denominator
    loss = huber + .1 * pcc + F_WEIGHT * residual
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite public joint loss")
    return loss, dict(slow_huber=huber, slow_pcc=pcc, F_residual=residual)
