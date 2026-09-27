"""Run B4 inference from explicit machine-I input; hidden targets are forbidden."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b4_diffusion import B4Diffusion
from ecg12gen.b4_model import ARCHITECTURE_ID, ARCHITECTURE_VERSION, B4ConditionalUNet1D, architecture_config_hash
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor-npy", required=True, help="Raw-uV machine-I array [N,1,5000]")
    parser.add_argument("--run-dir")
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION or checkpoint.get("architecture_id") != ARCHITECTURE_ID:
        raise SystemExit("B4 checkpoint architecture mismatch")
    architecture = checkpoint.get("architecture_config")
    if checkpoint.get("architecture_config_hash") != architecture_config_hash(architecture):
        raise SystemExit("B4 checkpoint architecture fingerprint mismatch")
    if checkpoint.get("diffused_lead_indices") != list(range(1, 12)):
        raise SystemExit("B4 checkpoint does not diffuse exactly II--V6")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else Path(args.checkpoint).resolve().parent
    scales = json.loads((run_dir / "preprocessing_scales.json").read_text(encoding="utf-8"))
    preprocessing_config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    preprocessor = ECGPreprocessor(
        preprocessing_config,
        {key: np.asarray(value, dtype=np.float32) for key, value in scales.items()},
    )
    anchor_raw = np.asarray(np.load(args.anchor_npy), dtype=np.float32)
    if anchor_raw.ndim != 3 or anchor_raw.shape[1:] != (1, 5000):
        raise SystemExit("anchor-npy must have shape [N,1,5000]")
    anchor_model, _, _ = preprocessor.transform_batch(anchor_raw, "ecg_machine_i")
    device = torch.device(args.device)
    model = B4ConditionalUNet1D(architecture)
    model.load_state_dict(checkpoint["model"], strict=True)
    diffusion = B4Diffusion(model, training_steps=int(checkpoint["training_steps"])).to(device).eval()
    sampling_steps = args.sampling_steps or int(checkpoint.get("sampling_steps", 50))
    eta = float(checkpoint.get("eta", 0.0) if args.eta is None else args.eta)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(anchor_model), args.batch_size):
            anchor_batch = torch.from_numpy(anchor_model[start:start + args.batch_size]).to(device)
            model_prediction = diffusion.sample(
                anchor_batch,
                sampling_steps=sampling_steps,
                eta=eta,
                generator=generator,
            ).cpu().numpy()
            raw_prediction = model_prediction * np.asarray(scales["d12"], dtype=np.float32)[None, :, None]
            raw_prediction[:, :1] = anchor_raw[start:start + args.batch_size]
            predictions.append(raw_prediction.astype(np.float32))
    prediction = np.concatenate(predictions)
    if prediction.shape != (len(anchor_raw), 12, 5000) or not np.array_equal(prediction[:, :1], anchor_raw):
        raise RuntimeError("B4 output contract failed")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "prediction_raw.npy", prediction)
    np.save(output / "prediction_submit.npy", prediction)
    (output / "inference_contract.json").write_text(
        json.dumps({
            "architecture_version": ARCHITECTURE_VERSION,
            "condition": "machine_I_only",
            "diffused_leads": "II--V6",
            "anchor_i_reinserted_exactly": True,
            "output_shape": list(prediction.shape),
            "hidden_target_argument_used": False,
            "sampling_steps": sampling_steps,
            "eta": eta,
            "seed": args.seed,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote B4 [N,12,5000] predictions to {output}")


if __name__ == "__main__":
    main()
