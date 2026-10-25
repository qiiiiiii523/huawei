"""Target-free inference: immutable I, K-sample mean, central-core stitching."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import torch

from . import GENERATED_INDICES
from .checkpoint import load_checkpoint, model_from_checkpoint
from .config import SamplingConfig
from .flow import sample
from .leads import compose_d12, validate_scales


def collate_metadata(values, device):
    if not any(values):
        return None
    defaults = {"numeric": torch.zeros(3), "sex": torch.tensor(2),
                "field_mask": torch.zeros(4, dtype=torch.bool), "age_topcoded": torch.zeros(1)}
    if any(value and set(value) != set(defaults) for value in values):
        raise ValueError("All four metadata arrays must be provided together")
    return {key: torch.stack([torch.as_tensor((value or defaults)[key], device=device)
                              for value in values]) for key in defaults}


@torch.no_grad()
def predict_record(model, anchor_uV, scales_uV, record_key, metadata=None, config=None, *, batch_size=1):
    return predict_records(model, [anchor_uV], scales_uV, [record_key], [metadata], config,
                           batch_size=batch_size)[0]


@torch.no_grad()
def predict_records(model, anchors_uV, scales_uV, record_keys, metadata=None, config=None, *, batch_size=1):
    """Batch windows across records; each retains its own noise key and core position."""
    config = config or SamplingConfig()
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive integer inference batch_size required")
    if not anchors_uV or len(anchors_uV) != len(record_keys):
        raise ValueError("Nonempty anchors and matching record keys required")
    metadata = [None] * len(anchors_uV) if metadata is None else metadata
    if len(metadata) != len(anchors_uV):
        raise ValueError("Metadata and records disagree")
    if model.training:
        raise ValueError("Prediction requires eval mode")
    device = next(model.parameters(), torch.empty(0)).device
    scales = validate_scales(scales_uV).to(device)
    core, width = config.core_seconds * 500, config.window_seconds * 500
    margin = (width - core) // 2
    anchors, results = [], []
    for value in anchors_uV:
        anchor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if anchor.ndim == 1:
            anchor = anchor[None]
        if anchor.ndim != 2 or anchor.shape[0] != 1 or not torch.isfinite(anchor).all():
            raise ValueError("Visible I must be finite [1,T] in microvolts")
        length = anchor.shape[-1]
        if length < core or length % core:
            raise ValueError("Record length must be a positive multiple of real 10s (5000 points)")
        anchors.append(anchor)
        results.append(anchor.new_empty(1, 7, length))

    pending = []
    def flush():
        # Conditions are batched by the same indices as states and I. Missing
        # metadata is represented by the model's existing zero-vector defaults.
        condition = collate_metadata([metadata[i] for i, _ in pending], device)
        windows, keys = [], []
        for i, s in pending:
            anchor = anchors[i]
            positions = torch.arange(s - margin, s - margin + width, device=device).clamp(0, anchor.shape[-1] - 1)
            windows.append(anchor[:, positions] / scales[0])
            keys.append(f"{record_keys[i]}:core={s}")
        predicted = sample(model, torch.stack(windows), keys, condition, config, batch_size=batch_size)
        for j, (i, s) in enumerate(pending):
            results[i][:, :, s:s + core] = predicted[j:j + 1, :, margin:margin + core]
        pending.clear()

    for i, anchor in enumerate(anchors):
        for s in range(0, anchor.shape[-1], core):
            pending.append((i, s))
            if len(pending) == batch_size:
                flush()
    if pending:
        flush()
    outputs = []
    for anchor, result in zip(anchors, results):
        result = result * scales[list(GENERATED_INDICES)][None, :, None]
        output = compose_d12(anchor.unsqueeze(0), result)[0]
        if not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite physical waveform")
        outputs.append(output)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor", required=True, help=".npy: current visible I only, raw microvolts at 500Hz")
    parser.add_argument("--record-key", required=True)
    parser.add_argument("--metadata", help="JSON of already encoded B7 metadata arrays")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, help="Concurrent ODE trajectories; default4 for120s,32 otherwise")
    parser.add_argument('--window-seconds', type=int, choices=(10, 40, 120))
    parser.add_argument('--core-seconds', type=int, choices=(10, 20, 120))
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Use a new prediction output")
    checkpoint = load_checkpoint(args.checkpoint)
    model = model_from_checkpoint(checkpoint, args.device)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8-sig")) if args.metadata else None
    config = SamplingConfig(**checkpoint["sampling"])
    if (args.window_seconds is None) != (args.core_seconds is None):
        parser.error('Provide both --window-seconds and --core-seconds')
    if args.window_seconds is not None:
        config = replace(config, window_seconds=args.window_seconds, core_seconds=args.core_seconds)
    args.batch_size = args.batch_size if args.batch_size is not None else (4 if config.window_seconds == 120 else 32)
    prediction = predict_record(model, np.load(args.anchor, allow_pickle=False), checkpoint["scales_uV"],
                                args.record_key, metadata, config, batch_size=args.batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, prediction.cpu().numpy())
    path.with_suffix(path.suffix + ".json").write_text(json.dumps({"sampling": asdict(config), "batch_size": args.batch_size,
        "shape": list(prediction.shape), "unit": "uV", "target_used": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
