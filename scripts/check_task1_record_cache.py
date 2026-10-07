"""Audit the configured Task-1 cache without training a model."""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.evaluate import evaluate_record_predictions


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = ECGDataConfig.from_yaml(ROOT / "configs" / "common.yaml")
    task_dir = config.path("task1_output")
    metadata = _rows(task_dir / "task1_window_metadata.csv")
    authoritative = {row["subject_id"]: row["split"]
                     for row in _rows(config.path("subject_split_csv"))}
    if any(authoritative.get(row["subject_id"]) != row["split"] for row in metadata):
        raise AssertionError("Task-1 cache changed the authoritative subject split")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata:
        grouped[row["pair_id"]].append(row)
    pair_counts = Counter(rows[0]["split"] for rows in grouped.values())
    if len(grouped) != 104 or pair_counts != Counter({"train": 83, "validation": 21}):
        raise AssertionError(f"Unexpected Task-1 pair counts: {pair_counts}")
    for pair_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["start_sample_500hz"]))
        expected = int(ordered[0]["expected_window_count"])
        starts = [int(row["start_sample_500hz"]) for row in ordered]
        if starts != list(range(0, expected * 5000, 5000)):
            raise AssertionError(f"Incomplete record cache: {pair_id}")
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "validation"):
        arrays[f"{split}_input"] = np.load(task_dir / f"task1_{split}_input.npy", mmap_mode="r")
        arrays[f"{split}_target"] = np.load(task_dir / f"task1_{split}_target.npy", mmap_mode="r")
        arrays[f"{split}_mask"] = np.load(task_dir / f"task1_{split}_context_valid_mask.npy", mmap_mode="r")
        count = sum(row["split"] == split for row in metadata)
        if arrays[f"{split}_input"].shape != (count, 1, 5000):
            raise AssertionError(f"Wrong {split} input shape")
        if arrays[f"{split}_target"].shape != (count, 12, 5000):
            raise AssertionError(f"Wrong {split} target shape")
        if arrays[f"{split}_mask"].shape != (count, 5000):
            raise AssertionError(f"Wrong {split} context mask shape")
    validation_rows = [row for row in metadata if row["split"] == "validation"]
    target = arrays["validation_target"]
    identity, _ = evaluate_record_predictions(target, target, "task1", validation_rows)
    if identity["n_records"] != 21 or not np.isclose(identity["missing11_mean_pearson_r"], 1.0):
        raise AssertionError("Record evaluator did not recover the 21 complete validation records")
    train = JointAnchorDataset(config, "task1", "train")
    validation = JointAnchorDataset(config, "task1", "validation")
    gap_sample = next(sample for sample in validation
                      if not np.asarray(sample.context_time_mask).all())
    if gap_sample.context_time_mask.shape != gap_sample.context_ecg.shape:
        raise AssertionError("Dataset did not expose the context time mask")
    print(
        "PASS: Task-1 V2 cache; "
        f"pairs={dict(pair_counts)}, windows={len(metadata)}, "
        f"joint_train={len(train)}, validation={len(validation)}, "
        f"validation_gap_samples={int((~arrays['validation_mask']).sum())}"
    )


if __name__ == "__main__":
    main()
