"""The existing P SlowPCC objective, with no new multiscale loss."""
import torch
from torch.nn import functional as F


def upsample_native(value):
    weight = (torch.arange(10, device=value.device, dtype=value.dtype) + .5) / 10
    middle = value[..., :-1, None] * (1 - weight) + value[..., 1:, None] * weight
    return torch.cat((value[..., :1].expand(*value.shape[:-1], 5), middle.flatten(-2),
                      value[..., -1:].expand(*value.shape[:-1], 5)), -1)

F_WEIGHT = 2.5890171538829843
COMBO_WEIGHT = .1


def smooth(x, width=51):
    r = width // 2
    padded = torch.cat((x[..., :1].expand(*x.shape[:-1], r), x,
                        x[..., -1:].expand(*x.shape[:-1], r)), -1)
    return F.avg_pool1d(padded, width, stride=1)


def pcc_loss(p, y, mask):
    valid = mask.bool()[..., None]
    p, y = torch.where(valid, p, 0.), torch.where(valid, y, 0.)
    a, b = p - p.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
    aa, bb = a.square().sum(-1), b.square().sum(-1)
    r = (a * b).sum(-1) / torch.sqrt(aa * bb + 1e-8)
    eligible = mask.bool() & (bb > 1e-8)
    return torch.where(eligible, 1 - r, 0.).sum() / eligible.sum().clamp_min(1)


def objective(delta, batch, scales):
    mask = batch["mask"][:, 1:].bool()
    p = batch["base_slow"] + delta
    target = batch["slow_target"]
    valid = mask[..., None]
    a, b = torch.where(valid, p, 0.), torch.where(valid, target, 0.)
    denominator = mask.sum().clamp_min(1) * target.shape[-1]
    huber = F.huber_loss(a, b, reduction="sum") / denominator
    pcc = pcc_loss(p, target, mask)
    residual = F.huber_loss(a - smooth(a), b - smooth(b), reduction="sum") / denominator
    added = upsample_native(delta.double() * scales[None, 1:, None].double())
    hybrid = batch["base_uV"][:, 1:].double() + .75 * added
    combo = pcc_loss(hybrid, batch["target_uV"][:, 1:].double(), mask)
    loss = huber + .1 * pcc + F_WEIGHT * residual + COMBO_WEIGHT * combo
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite CNN120 objective")
    return loss, {"slow_huber": huber, "slow_pcc": pcc, "F_residual": residual, "hybrid_pcc": combo}
