"""Task2 body-scale A/B input variant reader with canonical target alignment."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import numpy as np

from .contracts import ContractError, ECGSample, SupervisionMode, canonical_lead_mask
from .dataset import ECGDataConfig


class BodyScaleVariantDataset:
    """Read body-scale A or B inputs while keeping the canonical task2 target.

    A selects body-scale rows from task2_output. B selects its local array index
    and uses canonical_array_index to retrieve the exact task2 target window.
    Both variants are filtered by the same pair/quality gates and fixed subject
    split; no input or target array is rewritten.
    """

    def __init__(self, config: ECGDataConfig | str | Path, split: str, variant: str = "A_raw_window") -> None:
        self.config = ECGDataConfig.from_yaml(config) if not isinstance(config, ECGDataConfig) else config
        if split not in {"train", "validation"}:
            raise ContractError("split must be train or validation")
        if variant not in {"A_raw_window", "B_detrend_0p2Hz_then_window"}:
            raise ContractError("Unknown body-scale input variant")
        self.split, self.variant = split, variant
        task_dir = self.config.path("task2_output")
        self.target = np.load(task_dir / f"task2_{split}_target.npy", mmap_mode="r")
        with (task_dir / "task2_window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
            task_rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        task_rows = sorted(task_rows, key=lambda row: int(row["array_index"]))
        if [int(row["array_index"]) for row in task_rows] != list(range(len(task_rows))):
            raise ContractError("task2 window metadata array_index must be contiguous")
        with self.config.path("subject_split_csv").open(encoding="utf-8-sig", newline="") as handle:
            subject_split = {row["subject_id"]: row["split"] for row in csv.DictReader(handle)}
        with self.config.path("task2_pair_manifest_csv").open(encoding="utf-8-sig", newline="") as handle:
            pairs = {row["pair_id"]: row for row in csv.DictReader(handle)}
        if variant == "A_raw_window":
            self.inputs = np.load(task_dir / f"task2_{split}_input.npy", mmap_mode="r")
            candidates = [{"row": row, "input_index": int(row["array_index"]), "target_index": int(row["array_index"])} for row in task_rows if row.get("input_type") == "body_scale_d6"]
        else:
            ablation_dir = self.config.path("task2_body_scale_ablation")
            self.inputs = np.load(ablation_dir / f"body_scale_{split}_input_B_raw_detrended_0p2Hz.npy", mmap_mode="r")
            with self.config.path("task2_body_scale_b_metadata").open(encoding="utf-8-sig", newline="") as handle:
                b_rows = [row for row in csv.DictReader(handle) if row["split"] == split]
            candidates = [{"row": row, "input_index": int(row["local_array_index"]), "target_index": int(row["canonical_array_index"])} for row in b_rows]
        if self.inputs.ndim != 3 or self.inputs.shape[1:] != (6, 5000) or self.target.ndim != 3 or self.target.shape[1:] != (12, 5000):
            raise ContractError("Body-scale A/B arrays must be [N,6,5000] and task2 targets [N,12,5000]")
        self.rows = []
        for item in candidates:
            row, target_index = item["row"], item["target_index"]
            if target_index < 0 or target_index >= len(self.target):
                raise ContractError(f"Target index out of bounds: {target_index}")
            if subject_split.get(row["subject_id"]) != split:
                raise ContractError(f"Subject split mismatch for {row['subject_id']}")
            pair = pairs.get(row["pair_id"])
            if pair is None or pair.get("pair_status") != "paired" or pair.get("input_quality_status") != "usable" or pair.get("target_quality_status") != "usable":
                continue
            if int(item["input_index"]) >= len(self.inputs):
                raise ContractError("Input index out of bounds")
            self.rows.append({**item, "pair": pair})
        self.rows.sort(key=lambda item: int(item["input_index"]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ECGSample:
        item = self.rows[index]
        row, pair = item["row"], item["pair"]
        x = np.asarray(self.inputs[item["input_index"]], dtype=np.float32)
        y = np.asarray(self.target[item["target_index"]], dtype=np.float32)
        mask = canonical_lead_mask(6)
        sample = ECGSample(
            X_ecg=x, lead_mask=mask, Y_12lead=y, missing_mask=~mask, task_id="task2", ppg=None, acc=None,
            meta={"subject_id": row["subject_id"], "window_id": row["window_id"], "pair_id": row["pair_id"], "device_type": "body_scale_d6", "input_processing_variant": self.variant, "canonical_array_index": item["target_index"], "alignment_quality_score": 0.0, "pointwise_mse_allowed": False, "pointwise_loss_allowed": False},
            modality_mask={"ppg": False, "acc": False}, split=self.split, supervision_mode=SupervisionMode.CROSS_DEVICE_WEAK_ADAPTATION.value,
            pairing_type="reliable_subject_id_cross_device", alignment_mode="weak_subject_pair_record_start", pair_confidence=pair.get("pair_confidence", "not_applicable"), pair_status=pair.get("pair_status", "unknown"),
        )
        sample.validate()
        return sample

    def __iter__(self) -> Iterator[ECGSample]:
        for index in range(len(self)):
            yield self[index]