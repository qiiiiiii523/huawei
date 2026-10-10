"""Evaluate an existing trained checkpoint by main's actual record-level evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sampling_settings import add_sampling_arguments, sampling_from_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    add_sampling_arguments(parser)
    args = parser.parse_args()
    settings = sampling_from_arguments(args)
    from .checkpoint import checkpoint_preprocessor, load_checkpoint, model_from_checkpoint
    from .config import ModelConfig, load_config
    from .data import preprocessing_config
    from .runtime import build_validation_datasets, device_from_name, seed_all, validate
    config = load_config(args.config)
    config["sampling"] = settings
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint["architecture_hash"] != ModelConfig.from_dict(config["model"]).fingerprint:
        raise ValueError("Validation configuration differs from the trained architecture/condition schema")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty validation output directory")
    seed_all(int(config["sampling"]["seed"]), bool(config["training"]["deterministic"]))
    device = device_from_name(args.device)
    preprocessor = checkpoint_preprocessor(checkpoint, preprocessing_config(config))
    datasets = build_validation_datasets(config, preprocessor)
    model = model_from_checkpoint(checkpoint, device)
    print("sampling=", json.dumps(settings, sort_keys=True), flush=True)
    summaries = validate(model, datasets, config, preprocessor, device, output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "inference_sampling.json").write_text(json.dumps({
        "sampling": settings, "sampling_config": str(Path(args.sampling_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve())}, indent=2), encoding="utf-8")
    for name, summary in summaries.items():
        print(name, "r_missing11=", summary["r_missing11"])
        if "task2_missing_lead_mean_rmse_uV" in summary:
            print(name, "chest_rmse_uV=", summary["task2_missing_lead_mean_rmse_uV"])


if __name__ == "__main__":
    main()
