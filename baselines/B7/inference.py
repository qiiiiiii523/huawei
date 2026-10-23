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
    config = config or SamplingConfig()
    if model.training:
        raise ValueError("Prediction requires eval mode")
    device = next(model.parameters(), torch.empty(0)).device
    anchor = torch.as_tensor(anchor_uV, dtype=torch.float32, device=device)
    if anchor.ndim == 1:
        anchor = anchor[None]
    if anchor.ndim != 2 or anchor.shape[0] != 1 or not torch.isfinite(anchor).all():
        raise ValueError("Visible I must be finite [1,T] in microvolts")
    scales = validate_scales(scales_uV).to(device)
    length = anchor.shape[-1]
    core, width = config.core_seconds * 500, config.window_seconds * 500
    margin = (width - core) // 2
    if length < core or length % core:
        raise ValueError("Record length must be a positive multiple of real 10s (5000 points)")
    if metadata:
        metadata = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in metadata.items()}
    result = anchor.new_empty(1, 7, length)
    for s in range(0, length, core):
        positions = torch.arange(s - margin, s - margin + width, device=device).clamp(0, length - 1)
        # Fixed full-resolution I is the first channel at every ODE evaluation.
        q = anchor[:, positions].unsqueeze(0) / scales[0]
        predicted = sample(model, q, [f"{record_key}:core={s}"], metadata, config)
        result[:, :, s:s + core] = predicted[:, :, margin:margin + core]
    result = result * scales[list(GENERATED_INDICES)][None, :, None]
    output = compose_d12(anchor.unsqueeze(0), result)[0]
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
