"""Train B4 on main's strict synchronized P0 data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b4_data import build_b4_datasets, fit_b4_preprocessor
from ecg12gen.b4_diffusion import B4Diffusion
from ecg12gen.b4_model import B4ConditionalUNet1D
from ecg12gen.b4_train import fit_b4
from ecg12gen.training import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "common.yaml"))
    parser.add_argument("--b4-config", default=str(ROOT / "configs" / "b4.yaml"))
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.b4_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    architecture = config["architecture"]
    diffusion_config = config["diffusion"]
    training = config["training"]
    seed_everything(int(training["seed"]), deterministic=True)
    preprocessor = fit_b4_preprocessor(args.config, args.task_id)
    train_dataset, validation_dataset = build_b4_datasets(args.config, args.task_id, preprocessor)
    model = B4ConditionalUNet1D(architecture)
    diffusion = B4Diffusion(model, training_steps=int(diffusion_config["training_steps"]))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scales = {key: value.tolist() for key, value in preprocessor.scale_uV_by_source.items()}
    (output / "preprocessing_scales.json").write_text(json.dumps(scales, indent=2), encoding="utf-8")
    run = {
        **model.architecture_metadata,
        "task_id": args.task_id,
        "stage": "P0_anchor_only",
        "training_data": "main:metadata/d12_strict_pretrain_index.csv",
        "condition": "same_window_machine_I_only",
        "diffused_lead_indices": list(range(1, 12)),
        "uses_watch": False,
        "uses_body_scale": False,
        "uses_report_or_diagnosis": False,
        "checkpoint_metric": "r_missing11",
        "seed": 42,
    }
    (output / "b4_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"B4 data: train={len(train_dataset)} validation={len(validation_dataset)}")
    checkpoint = fit_b4(
        diffusion,
        train_dataset,
        validation_dataset,
        preprocessor.scale_uV_by_source["d12"],
        output,
        epochs=args.epochs or int(training["epochs"]),
        batch_size=args.batch_size or int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        sampling_steps=args.sampling_steps or int(diffusion_config["sampling_steps"]),
        eta=float(diffusion_config["eta"]),
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
    )
    print(f"B4 checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
