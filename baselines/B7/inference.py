"""Target-free inference: immutable I, K-sample mean, central-core stitching."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from . import GENERATED_INDICES
from .checkpoint import load_checkpoint, model_from_checkpoint
from .config import SamplingConfig
from .flow import sample
from .leads import compose_d12, validate_scales


@torch.no_grad()
def predict_record(model, anchor_uV, scales_uV, record_key, metadata=None, config=None):
    return predict_records(model, [anchor_uV], scales_uV, [record_key], [metadata], config)[0]


@torch.no_grad()
def predict_records(model, anchors_uV, scales_uV, record_keys, metadatas=None, config=None):
    config = config or SamplingConfig()
    if model.training:
        raise ValueError("Prediction requires eval mode")
    device = next(model.parameters(), torch.empty(0)).device
    if not anchors_uV or len(record_keys) != len(anchors_uV):
        raise ValueError("Matching nonempty anchors and keys required")
    tensors = []
    for a in anchors_uV:
        a = torch.as_tensor(a, dtype=torch.float32, device=device)
        if a.ndim == 1:
            a = a[None]
        if a.ndim != 2 or a.shape[0] != 1 or not torch.isfinite(a).all():
            raise ValueError("Visible I must be finite [1,T] in microvolts")
        tensors.append(a)
    if len({a.shape[-1] for a in tensors}) != 1:
        raise ValueError("Batched records must have equal lengths")
    anchor = torch.stack(tensors)
    scales = validate_scales(scales_uV).to(device)
    length = anchor.shape[-1]
    core, width = config.core_seconds * 500, config.window_seconds * 500
    margin = (width - core) // 2
    if length < core or length % core:
        raise ValueError("Record length must be a positive multiple of real 10s (5000 points)")
    metadatas = metadatas or [None] * len(anchor)
    if len(metadatas) != len(anchor):
        raise ValueError("Metadata count differs")
    unknown = {"numeric": [0., 0., 0.], "sex": 2, "field_mask": [False]*4, "age_topcoded": [0.]}
    metadata = {k: torch.stack([torch.as_tensor((m or unknown)[k], device=device) for m in metadatas])
                for k in unknown}
    result = anchor.new_empty(len(anchor), 7, length)
    for s in range(0, length, core):
        positions = torch.arange(s - margin, s - margin + width, device=device).clamp(0, length - 1)
        # Fixed full-resolution I is the first channel at every ODE evaluation.
        q = anchor[:, :, positions] / scales[0]
        predicted = sample(model, q, [f"{key}:core={s}" for key in record_keys], metadata, config)
        result[:, :, s:s + core] = predicted[:, :, margin:margin + core]
    result = result * scales[list(GENERATED_INDICES)][None, :, None]
    output = compose_d12(anchor, result)
    if not torch.isfinite(output).all():
        raise FloatingPointError("Nonfinite physical waveform")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor", required=True, help=".npy: current visible I only, raw microvolts at 500Hz")
    parser.add_argument("--record-key", required=True)
    parser.add_argument("--metadata", help="JSON of already encoded B7 metadata arrays")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Use a new prediction output")
    checkpoint = load_checkpoint(args.checkpoint)
    model = model_from_checkpoint(checkpoint, args.device)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8-sig")) if args.metadata else None
    config = SamplingConfig(**checkpoint["sampling"])
    prediction = predict_record(model, np.load(args.anchor, allow_pickle=False), checkpoint["scales_uV"],
                                args.record_key, metadata, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, prediction.cpu().numpy())
    path.with_suffix(path.suffix + ".json").write_text(json.dumps({"sampling": asdict(config),
        "shape": list(prediction.shape), "unit": "uV", "target_used": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
