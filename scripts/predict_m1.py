"""Write M1 validation predictions and run the unchanged official V0 evaluator."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b2_data import b2_collate, build_weak_dataset
from ecg12gen.m1_model import M1MaskedCNNLeadTimeTransformer
from ecg12gen.m1_train import _forward
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def _load_preprocessor(run_dir: Path) -> ECGPreprocessor:
    config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    scales = json.loads((run_dir / "preprocessing_scales.json").read_text(encoding="utf-8"))
    return ECGPreprocessor(config, {key: np.asarray(value, dtype=np.float32) for key, value in scales.items()})


def _write_metadata(path: Path, metadata: list[dict[str, object]]) -> None:
    fields = ["array_index", "subject_id", "input_type", "split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(metadata):
            writer.writerow({
                "array_index": index,
                "subject_id": row.get("subject_id", "unknown"),
                "input_type": row.get("device_type", "watch_ecg"),
                "split": "validation",
            })


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "common.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default=None, help="Directory containing preprocessing_scales.json")
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--task2-variant", choices=("A_raw_window", "B_detrend_0p2Hz_then_window"), default="A_raw_window")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw-baseline-policy", choices=("fixed_zero_uV",), default="fixed_zero_uV")
    parser.add_argument("--centered-diagnostic", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else checkpoint_path.parent
    preprocessor = _load_preprocessor(run_dir)
    dataset = build_weak_dataset(args.config, args.task_id, "validation", preprocessor, args.task2_variant)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False, collate_fn=b2_collate)
    model = M1MaskedCNNLeadTimeTransformer().to(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    morphology, target, masks, metadata = [], [], [], []
    d12_scale = preprocessor.scale_uV_by_source["d12"]
    for batch in loader:
        moved = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        morphology.append(_forward(model, moved).cpu().numpy() * d12_scale[None, :, None])
        target.append(batch["raw_target_uV"].numpy())
        masks.append(batch["lead_mask"].numpy())
        metadata.extend(batch["meta"])

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction = np.concatenate(morphology).astype(np.float32)
    target_array = np.concatenate(target).astype(np.float32)
    np.save(output / "prediction.npy", prediction)
    np.save(output / "validation_target_for_v0.npy", target_array)
    np.save(output / "morphology_uV.npy", prediction)
    np.save(output / "lead_mask.npy", np.concatenate(masks))
    metadata_path = output / "prediction_metadata.csv"
    _write_metadata(metadata_path, metadata)

    command = [
        sys.executable, "-m", "ecg12gen.evaluate", "--prediction", str(output / "prediction.npy"),
        "--target", str(output / "validation_target_for_v0.npy"), "--metadata", str(metadata_path),
        "--task-id", args.task_id, "--output-dir", str(output),
    ]
    if args.centered_diagnostic:
        command.append("--write-centered-diagnostic")
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Wrote M1 predictions with raw_baseline_policy={args.raw_baseline_policy}: {output}")


if __name__ == "__main__":
    main()
