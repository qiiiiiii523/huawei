"""Small read-only integration check for D0, D1, and V0."""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg12gen.contracts import D12_LEADS, ContractError, SupervisionMode
from ecg12gen.dataset import ECGDataConfig, UnifiedECGDataset
from ecg12gen.evaluate import evaluate_predictions, evaluate_task2_diagnostics, write_report, write_task2_diagnostics

def _all_subjects_match_fixed_split(config: ECGDataConfig, task_id: str) -> None:
    task_dir = config.path(f"{task_id}_output")
    with (task_dir / f"{task_id}_window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with config.path("subject_split_csv").open(encoding="utf-8-sig", newline="") as handle:
        fixed = {row["subject_id"]: row["split"] for row in csv.DictReader(handle)}
    observed: dict[str, set[str]] = {}
    for row in rows:
        assert fixed.get(row["subject_id"]) == row["split"], f"split mismatch: {row['subject_id']}"
        observed.setdefault(row["subject_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in observed.values()), "subject leakage between train and validation"

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/common.yaml")
    parser.add_argument("--output-dir", default="check_results/v0_synthetic")
    args = parser.parse_args(); config = ECGDataConfig.from_yaml(args.config)
    task1_train, task1_val = UnifiedECGDataset(config, "task1", "train"), UnifiedECGDataset(config, "task1", "validation")
    task2_train, task2_val = UnifiedECGDataset(config, "task2", "train"), UnifiedECGDataset(config, "task2", "validation")
    s1, s2 = task1_train[0], task2_train[0]
    assert s1.X_ecg.shape == (1, 5000) and s1.Y_12lead.shape == (12, 5000)
    assert s2.X_ecg.shape == (6, 5000) and s2.Y_12lead.shape == (12, 5000)
    assert tuple(config.signal["twelve_lead_order"]) == D12_LEADS
    assert UnifiedECGDataset(config, "task1", "train", SupervisionMode.D12_I_PRETRAIN.value)[0].X_ecg.shape == (1, 5000)
    assert UnifiedECGDataset(config, "task2", "train", SupervisionMode.D12_SIX_PRETRAIN.value)[0].X_ecg.shape == (6, 5000)
    try:
        UnifiedECGDataset(config, "task1", "validation", SupervisionMode.D12_I_PRETRAIN.value)
        raise AssertionError("Validation D12 pretraining unexpectedly accepted")
    except ContractError:
        pass
    for dataset in (task1_train, task1_val, task2_train, task2_val):
        assert len(dataset) > 0 and all(sample.pair_status == "paired" for sample in dataset)
        assert all(not sample.meta["pointwise_mse_allowed"] for sample in dataset)
    _all_subjects_match_fixed_split(config, "task1"); _all_subjects_match_fixed_split(config, "task2")
    # Synthetic data only: no competition validation target is used as prediction.
    target = np.linspace(-1, 1, num=2 * 12 * 5000, dtype=np.float32).reshape(2, 12, 5000)
    prediction = target + 0.1
    overall, details = evaluate_predictions(prediction, target, "task2")
    report_paths = write_report(args.output_dir, overall, details)
    diagnostic_metadata = [
        {"subject_id": "synthetic_machine", "input_type": "ecg_machine_d6"},
        {"subject_id": "synthetic_scale", "input_type": "body_scale_d6"},
    ]
    subject_rows, device_rows = evaluate_task2_diagnostics(prediction, target, diagnostic_metadata)
    diagnostic_paths = write_task2_diagnostics(args.output_dir, subject_rows, device_rows, report_paths[2])
    assert len(subject_rows) == 2 and {row["input_type"] for row in device_rows} == {"all", "ecg_machine_d6", "body_scale_d6"}
    assert all("generated_v1_v6_mean_pearson_r" in row for row in device_rows)
    assert all(path.is_file() for path in diagnostic_paths)
    print("PASS: D0 shapes/order, D1 gates/split protection, and V0 synthetic report")
    print(f"Rows: task1 train={len(task1_train)}, validation={len(task1_val)}; task2 train={len(task2_train)}, validation={len(task2_val)}")

if __name__ == "__main__":
    main()
