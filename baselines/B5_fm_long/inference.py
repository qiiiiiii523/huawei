"""Same physical per-10s noise for 10s and 40s ODE generation; no target arguments."""
import numpy as np
import torch

from baselines.B5.flow import keyed_noise, integrate
from baselines.B5_final.profile import SAMPLING
from baselines.B5_best.combine import combine, upsample_native
from baselines.B5_best.slow import predict_native
from . import FIELDS, WINDOW_SECONDS


def validate_visible(anchor, scale, record_key):
    raw = np.asarray(anchor, np.float32).reshape(1, -1)
    scale = np.asarray(scale, np.float32)
    if (raw.shape != (1, 60000) or not np.isfinite(raw).all() or scale.shape != (12,)
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not isinstance(record_key, str) or not record_key):
        raise ValueError("Need real finite120s I, positive frozen scales and stable record key")
    return raw, scale


def paired_noise(record_key, start, length, sample_index, device, seed=42):
    """Assemble the SAME old record:start noise keys before selecting a long window."""
    if start % 5000 or length % 5000 or start < 0 or start + length > 60000:
        raise ValueError("Noise segments must be aligned real10s blocks inside120s")
    keys = [record_key + ":" + str(s) for s in range(start, start + length, 5000)]
    blocks = keyed_noise(keys, 5000, seed, sample_index, device)
    return blocks.permute(1, 0, 2).reshape(11, length)


@torch.no_grad()
def generate_record(model, anchor_uV, meta, record_key, scales, device, window_seconds,
                    batch_size=1, sampling=None):
    raw, scale = validate_visible(anchor_uV, scales, record_key)
    if model.training or window_seconds not in WINDOW_SECONDS or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Use eval model, positive batch size, and10s or40s window")
    settings = dict(SAMPLING if sampling is None else sampling)
    if settings["solver"] not in ("heun", "euler") or settings["steps"] < 1 or settings["samples"] < 1:
        raise ValueError("Invalid ODE settings")
    width = window_seconds * 500
    starts = list(range(0, 60000, width))
    output = np.empty((12, 60000), np.float32)
    for offset in range(0, len(starts), batch_size):
        group = starts[offset:offset + batch_size]
        inputs = {key: torch.from_numpy(np.stack([np.asarray(meta[key]) for _ in group])).to(device)
                  for key in FIELDS}
        inputs["anchor"] = torch.from_numpy(np.stack([raw[:, s:s + width] / scale[0] for s in group])).to(device)
        condition = model.encode_conditions(**inputs)
        total = inputs["anchor"].new_zeros((len(group), 11, width))
        for index in range(settings["samples"]):
            noise = torch.stack([paired_noise(record_key, start, width, index, device, settings["seed"])
                                 for start in group])
            total += integrate(model, condition, noise, settings["steps"], settings["solver"])
        generated = torch.cat((condition.anchor_prediction, total / settings["samples"]), dim=1)
        restored = generated.float().cpu().numpy() * scale[None, :, None]
        for j, start in enumerate(group):
            output[:, start:start + width] = restored[j]
    return output


@torch.no_grad()
def frozen_slow(model, anchor_uV, meta, scales, device):
    raw, scale = validate_visible(anchor_uV, scales, "slow_visible_input")
    fields = {key: torch.from_numpy(np.asarray(meta[key])[None].copy()).to(device) for key in FIELDS}
    anchor50 = raw.reshape(1, 6000, 10).mean(-1) / scale[0]
    native = predict_native(model, {"anchor": torch.from_numpy(anchor50[None]).to(device), **fields})
    return upsample_native(native[0].cpu().numpy().astype(np.float64) * scale[1:, None].astype(np.float64))


def outputs(raw, slow):
    return {"raw": raw, "hybrid": combine(raw, slow, slow_beta=.75)}
