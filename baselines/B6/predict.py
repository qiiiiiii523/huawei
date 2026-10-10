"""Target-free arbitrary-length synchronous-I inference, no download or training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .metadata import encode_demographics


def canonical_anchor(raw: np.ndarray, fs: int, unit: str) -> tuple[np.ndarray, int]:
    from scipy.signal import resample_poly
    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2 or raw.shape[0] != 1 or not raw.shape[1] or not np.isfinite(raw).all():
        raise ValueError("Input must be a finite nonempty synchronous I array [T] or [1,T]")
    if fs < 1:
        raise ValueError("Sampling rate must be a positive integer")
    factors = {"uV": 1., "mV": 1000., "V": 1e6}
    if unit not in factors:
        raise ValueError("Unsupported voltage unit")
    length = raw.shape[1]
    canonical = raw.astype(np.float32) * factors[unit]
    if fs != 500:
        divisor = math.gcd(fs, 500)
        canonical = resample_poly(canonical, 500 // divisor, fs // divisor, axis=-1).astype(np.float32)
    return canonical, length


def restore_rate(prediction: np.ndarray, input_length: int, input_fs: int, output_fs: int) -> np.ndarray:
    from scipy.signal import resample_poly
    if output_fs < 1:
        raise ValueError("Output sampling rate must be positive")
    expected = max(1, round(input_length * output_fs / input_fs))
    if output_fs != 500:
        divisor = math.gcd(500, output_fs)
        prediction = resample_poly(prediction, output_fs // divisor, 500 // divisor, axis=-1).astype(np.float32)
    if prediction.shape[-1] < expected:
        prediction = np.pad(prediction, ((0, 0), (0, expected - prediction.shape[-1])), mode="edge")
    return prediction[:, :expected]


def predict_record(model: Any, preprocessor: Any, anchor_uV: np.ndarray, demographics: dict[str, np.ndarray],
                   record_id: str, device: Any, settings: dict[str, Any], batch_size: int = 8) -> np.ndarray:
    import torch
    from .flow import sample
    if batch_size < 1 or not record_id:
        raise ValueError("Positive batch size and stable record_id are required")
    anchor = preprocessor.transform_observed_record(anchor_uV, "ecg_machine_i").model_signal
    result = np.empty((12, anchor.shape[-1]), dtype=np.float32)
    starts = list(range(0, anchor.shape[-1], 5000))
    model.eval()
    for group_start in range(0, len(starts), batch_size):
        group = starts[group_start:group_start + batch_size]
        windows = []
        for start in group:
            window = anchor[:, start:start + 5000]
            windows.append(np.pad(window, ((0, 0), (0, 5000 - window.shape[-1])), mode="edge"))
        inputs = {"anchor": torch.from_numpy(np.stack(windows)).to(device)}
        for key in ("numeric", "sex", "field_mask", "age_topcoded"):
            inputs[key] = torch.from_numpy(np.stack([demographics[key] for _ in group])).to(device)
        prediction = sample(model, inputs, [f"{record_id}:{start}" for start in group], int(settings["seed"]),
                            int(settings["steps"]), settings["solver"], int(settings["samples"]))
        restored = prediction.float().cpu().numpy() * preprocessor.scale_uV_by_source["d12"][None, :, None]
        for index, start in enumerate(group):
            end = min(start + 5000, anchor.shape[-1])
            result[:, start:end] = restored[index, :, :end - start]
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite prediction")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor", required=True, help="Visible synchronous I .npy; never pass a target d12 file")
    parser.add_argument("--metadata-json", help="Optional observed age/gender/height/weight object, no diagnoses")
    parser.add_argument("--record-id", required=True, help="Stable record identity for reproducible noise")
    parser.add_argument("--input-fs", type=int, default=500)
    parser.add_argument("--input-unit", choices=("uV", "mV", "V"), default="uV")
    parser.add_argument("--output-fs", type=int, help="Default: same sampling rate as input")
    parser.add_argument("--output", required=True, help="New .npy file, shape [12,original_length] at default rate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--solver", choices=("heun", "euler"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--copy-observed-i", action="store_true", help="Submission-interface option only; evaluator never replaces I")
    parser.add_argument("--preprocessing-config", default=str(Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".npy" or output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Output must be a new .npy path with a new JSON sidecar")
    if output == Path(args.anchor).resolve() or (args.metadata_json and output == Path(args.metadata_json).resolve()):
        raise ValueError("Cannot overwrite input data")
    from ecg12gen.preprocessing import PreprocessingConfig
    from .checkpoint import checkpoint_preprocessor, load_checkpoint, model_from_checkpoint
    from .runtime import device_from_name, seed_all
    checkpoint = load_checkpoint(args.checkpoint)
    settings = dict(checkpoint["config"]["sampling"])
    for key in ("steps", "samples", "solver", "seed"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    seed_all(int(settings["seed"]), True)
    device = device_from_name(args.device)
    model = model_from_checkpoint(checkpoint, device)
    preprocessor = checkpoint_preprocessor(checkpoint, PreprocessingConfig.from_yaml(args.preprocessing_config))
    raw = np.load(args.anchor, allow_pickle=False)
    anchor, original_length = canonical_anchor(raw, args.input_fs, args.input_unit)
    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8-sig")) if args.metadata_json else {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata-json must contain an object")
    demographics = encode_demographics(metadata)
    prediction = predict_record(model, preprocessor, anchor, demographics, args.record_id, device, settings, args.batch_size)
    if args.copy_observed_i:
        prediction[0] = anchor[0]
    output_fs = args.output_fs if args.output_fs is not None else args.input_fs
    prediction = restore_rate(prediction, original_length, args.input_fs, output_fs)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, prediction, allow_pickle=False)
    description = {"record_id": args.record_id, "unit": "uV", "sampling_rate_hz": output_fs,
                   "length": prediction.shape[-1], "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                   "sampling": settings, "copied_observed_i": args.copy_observed_i,
                   "condition_schema": checkpoint["condition_schema"], "field_available": demographics["field_mask"].tolist()}
    output.with_suffix(".json").write_text(json.dumps(description, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}; shape={prediction.shape}, unit=uV")


if __name__ == "__main__":
    main()
