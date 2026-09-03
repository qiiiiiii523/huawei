"""Read-only integration check for task2 body-scale A/B variants."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.body_scale import BodyScaleVariantDataset
from ecg12gen.dataset import ECGDataConfig


def main() -> None:
    config = ECGDataConfig.from_yaml(ROOT / "configs" / "common.yaml")
    a_train = BodyScaleVariantDataset(config, "train", "A_raw_window")
    b_train = BodyScaleVariantDataset(config, "train", "B_detrend_0p2Hz_then_window")
    a_val = BodyScaleVariantDataset(config, "validation", "A_raw_window")
    b_val = BodyScaleVariantDataset(config, "validation", "B_detrend_0p2Hz_then_window")
    assert (len(a_train), len(a_val), len(b_train), len(b_val)) == (350, 60, 350, 60)
    assert a_train[0].X_ecg.shape == b_train[0].X_ecg.shape == (6, 5000)
    assert a_train[0].Y_12lead.shape == b_train[0].Y_12lead.shape == (12, 5000)
    assert a_train[0].meta["input_processing_variant"] == "A_raw_window"
    assert b_train[0].meta["input_processing_variant"] == "B_detrend_0p2Hz_then_window"
    task_meta = {}
    with config.path("task2_output").joinpath("task2_window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        task_meta = {(r["split"], int(r["array_index"])): r for r in csv.DictReader(handle)}
    with config.path("task2_body_scale_b_metadata").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            canonical = task_meta[(row["split"], int(row["canonical_array_index"]))]
            assert canonical["input_type"] == "body_scale_d6"
            assert canonical["subject_id"] == row["subject_id"]
            assert canonical["target_record_id"] == row["target_record_id"]
    print("PASS: task2 body-scale A/B variants; same counts, shapes, split, and canonical target alignment")


if __name__ == "__main__":
    main()