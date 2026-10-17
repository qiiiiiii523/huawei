"""The deployed S50 path, with variable real record length only in public stage one."""
import numpy as np
import torch
from torch.nn import functional as F

FIELDS = ("anchor", "numeric", "sex", "field_mask", "age_topcoded")


def generate(model, inputs):
    anchor = inputs["anchor"]
    if anchor.ndim != 3 or anchor.shape[1] != 1 or anchor.shape[-1] % 500 or anchor.shape[-1] < 2500:
        raise ValueError("S50 needs real records >=50s, in 10s multiples, at 50Hz")
    if not torch.isfinite(anchor).all():
        raise ValueError("Nonfinite visible I")
    count = anchor.shape[-1] // 500
    padded = torch.cat((anchor[..., :1].expand(-1, -1, 1000), anchor,
                        anchor[..., -1:].expand(-1, -1, 1000)), -1)
    windows = padded.unfold(-1, 2500, 500).permute(0, 2, 1, 3).reshape(-1, 1, 2500)
    meta = {k: inputs[k].repeat_interleave(count, 0) for k in FIELDS[1:]}
    center = model(windows, **meta)[..., 1000:1500]
    return center.reshape(len(anchor), count, 11, 500).permute(0, 2, 1, 3).reshape(len(anchor), 11, -1)


def upsample(native):
    if native.shape[-1] != 6000:
        raise ValueError("Official hybrid PCC requires a real 120s Huawei record")
    weights = (torch.arange(10, device=native.device, dtype=native.dtype) + .5) / 10
    middle = native[..., :-1, None] * (1 - weights) + native[..., 1:, None] * weights
    return torch.cat((native[..., :1].expand(*native.shape[:-1], 5), middle.flatten(-2),
                      native[..., -1:].expand(*native.shape[:-1], 5)), -1)


def smooth(x, width=51):
    radius = width // 2
    if width % 2 != 1 or width > x.shape[-1]:
        raise ValueError("Invalid centered smoothing width")
    padded = torch.cat((x[..., :1].expand(*x.shape[:-1], radius), x,
                        x[..., -1:].expand(*x.shape[:-1], radius)), -1)
    return F.avg_pool1d(padded, width, stride=1)


def slow_numpy(x):
    x = np.asarray(x, np.float64)
    padded = np.pad(x, ((0, 0), (250, 250)), mode="edge")
    sums = np.concatenate((np.zeros((len(x), 1)), np.cumsum(padded, axis=-1)), -1)
    return (sums[:, 501:] - sums[:, :-501]) / 501


def pcc_loss(p, y, mask):
    valid = mask.bool()[..., None]
    p, y = torch.where(valid, p, 0.), torch.where(valid, y, 0.)
    a, b = p - p.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
    aa, bb = a.square().sum(-1), b.square().sum(-1)
    r = (a * b).sum(-1) / torch.sqrt(aa * bb + 1e-8)
    eligible = mask.bool() & (bb > 1e-8)
    return torch.where(eligible, 1 - r, 0.).sum() / eligible.sum().clamp_min(1)


def objective(prediction, batch, scales, residual_weight, combo_weight):
    mask = batch["mask"][:, 1:].bool()
    valid = mask[..., None]
    y = batch["slow_target"]
    p, y = torch.where(valid, prediction, 0.), torch.where(valid, y, 0.)
    denominator = mask.sum().clamp_min(1) * p.shape[-1]
    huber = F.huber_loss(p, y, reduction="sum") / denominator
    correlation = pcc_loss(p, y, mask)
    residual = F.huber_loss(p - smooth(p), y - smooth(y), reduction="sum") / denominator
    loss = huber + .1 * correlation + residual_weight * residual
    combo = prediction.sum() * 0
    if combo_weight:
        fm = batch["fm_uV"][:, 1:].double()
        if fm.requires_grad or fm.shape[-1] != 60000:
            raise ValueError("Only frozen Huawei 120s FM may enter combo training")
        fast = fm - smooth(fm, 501)
        slow = upsample(prediction.double() * scales[None, 1:, None].double())
        combo = pcc_loss(fast + .75 * slow, batch["target_uV"][:, 1:].double(), mask)
        loss = loss + combo_weight * combo
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    return loss, {"slow_huber": huber, "slow_pcc": correlation, "F": residual, "combo_pcc": combo}


def batch_record(record, device, labels=True):
    keys = FIELDS + (("mask", "slow_target", "target_uV", "fm_uV") if labels else ())
    return {k: torch.from_numpy(np.asarray(record[k]).copy()).unsqueeze(0).to(device)
            for k in keys if k in record}
