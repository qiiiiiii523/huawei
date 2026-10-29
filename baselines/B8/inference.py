"""Target-free, direct-regression B8 inference with full-record 10-second stitching."""
from __future__ import annotations
import numpy as np
import torch
from baselines.B7.leads import compose_d12

@torch.no_grad()
def predict_generated_records(model, anchors_uV, scales_uV):
    if model.training: raise ValueError("B8 prediction requires eval mode")
    if not anchors_uV: raise ValueError("at least one anchor is required")
    device = next(model.parameters()).device
    anchors = []
    for anchor in anchors_uV:
        value = torch.as_tensor(anchor, dtype=torch.float32, device=device)
        if value.ndim == 1: value = value[None]
        if value.ndim != 2 or value.shape[0] != 1 or value.shape[-1] < 5000 or value.shape[-1] % 5000 or not torch.isfinite(value).all():
            raise ValueError("visible I must be finite [1,T], T a positive 10-second multiple")
        anchors.append(value)
    if len({x.shape[-1] for x in anchors}) != 1: raise ValueError("batched record lengths must agree")
    raw = torch.stack(anchors); scales = torch.as_tensor(scales_uV, dtype=raw.dtype, device=device)
    if scales.shape != (12,) or not torch.isfinite(scales).all() or (scales <= 0).any(): raise ValueError("need twelve positive frozen scales")
    generated = raw.new_empty(len(raw), 7, raw.shape[-1])
    for start in range(0, raw.shape[-1], 5000):
        generated[:, :, start:start + 5000] = model(raw[:, :, start:start + 5000] / scales[:1, None]) * scales[[1, 6, 7, 8, 9, 10, 11]][None, :, None]
    if not torch.isfinite(generated).all(): raise FloatingPointError("nonfinite B8 generated prediction")
    return generated


@torch.no_grad()
def predict_records(model, anchors_uV, scales_uV):
    generated = predict_generated_records(model, anchors_uV, scales_uV)
    raw = []
    for anchor in anchors_uV:
        anchor = torch.as_tensor(anchor, dtype=generated.dtype, device=generated.device)
        raw.append(anchor[None] if anchor.ndim == 1 else anchor)
    raw = torch.stack(raw)
    output = compose_d12(raw, generated)
    if not torch.isfinite(output).all(): raise FloatingPointError("nonfinite B8 prediction")
    return output
