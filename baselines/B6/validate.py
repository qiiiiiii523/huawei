"""Evaluate an existing trained checkpoint by main's actual record-level evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from .checkpoint import checkpoint_preprocessor, load_checkpoint, model_from_checkpoint
    from .config import ModelConfig, load_config
    from .data import preprocessing_config
    from .runtime import build_validation_datasets, device_from_name, seed_all, validate
    config = load_config(args.config)
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
    summaries = validate(model, datasets, config, preprocessor, device, output)
    for name, summary in summaries.items():
        print(name, "r_missing11=", summary["r_missing11"],
              "missing11_mean_rmse_uV=", summary['missing11_mean_rmse_uV'])
        if name == 'task2':
            print('task2 chest RMSE (uV)=', summary['task2_missing_lead_mean_rmse_uV'])


if __name__ == "__main__":
    main()
