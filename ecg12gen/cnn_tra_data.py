"""Strict, same-window I-to-seven-lead Huawei data access.

No demographic or asynchronous context data are opened by this module.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


LEAD_INDICES = (1, 6, 7, 8, 9, 10, 11)  # II, V1..V6 in canonical D12
TASKS = ("task1", "task2")
WINDOW = 5000


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _mask(text: str, size: int = 12) -> np.ndarray:
    pieces = text.split("|")
    if len(pieces) != size or any(piece not in {"0", "1"} for piece in pieces):
        raise ValueError(f"invalid {size}-lead quality mask: {text!r}")
    return np.asarray([piece == "1" for piece in pieces], dtype=bool)


@dataclass(frozen=True)
class TrainItem:
    task: str
    index: int
    target_record_id: str
    mask7: np.ndarray


class HuaweiSevenLeadData:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.train_arrays = {}
        self.validation_arrays = {}
        self.validation_rows = {}
        self.validation_masks = {}
        for task in TASKS:
            folder = self.root / "datasets" / task
            train = np.load(folder / f"{task}_train_target.npy", mmap_mode="r")
            valid = np.load(folder / f"{task}_validation_target.npy", mmap_mode="r")
            if train.ndim != 3 or train.shape[1:] != (12, WINDOW):
                raise ValueError(f"unexpected {task} train shape: {train.shape}")
            if valid.ndim != 3 or valid.shape[1:] != (12, WINDOW):
                raise ValueError(f"unexpected {task} validation shape: {valid.shape}")
            self.train_arrays[task] = train
            self.validation_arrays[task] = valid
            rows = _rows(folder / f"{task}_window_metadata.csv")
            for split, length in (("train", len(train)), ("validation", len(valid))):
                split_rows = [row for row in rows if row["split"] == split]
                if len(split_rows) != length or sorted(int(row["array_index"]) for row in split_rows) != list(range(length)):
                    raise ValueError(f"{task} {split} metadata is not aligned with its array")
                if split == "validation":
                    self.validation_rows[task] = sorted(split_rows, key=lambda row: int(row["array_index"]))
            quality = np.load(self.root / "validation" / task / "target_quality_mask.npy")
            if quality.shape != (len(valid), 12) or not np.isin(quality, (0, 1)).all():
                raise ValueError(f"{task} validation quality mask misaligned")
            self.validation_masks[task] = quality.astype(bool)
            self._check_complete_records(task)

        qc = {}
        for row in _rows(self.root / "metadata" / "device_interpretation_qc.csv"):
            if row["device_type"] == "ecg_machine_d12":
                qc[row["record_id"]] = _mask(row["target_lead_mask"])
        self.train_items = []
        strict_rows = _rows(self.root / "metadata" / "d12_strict_pretrain_index.csv")
        if not strict_rows:
            raise ValueError("empty strict training index")
        seen = set()
        for row in strict_rows:
            task = row["source_task_id"]
            if task not in TASKS or row["source_split"] != "train":
                raise ValueError("strict index must reference task1/task2 train arrays only")
            index = int(row["source_array_index"])
            if not 0 <= index < len(self.train_arrays[task]):
                raise ValueError("strict array index out of range")
            key = (task, index)
            if key in seen:
                raise ValueError("strict index repeats an array row")
            seen.add(key)
            record = row["target_record_id"]
            if record not in qc:
                raise ValueError(f"missing D12 QC for {record}")
            mask = qc[record]
            if not mask[0]:
                continue  # the actual input I is unreliable for this record
            mask7 = mask[list(LEAD_INDICES)]
            if mask7.any():
                self.train_items.append(TrainItem(task, index, record, mask7.copy()))
        if not self.train_items:
            raise ValueError("no eligible training windows")
        self.scale = self._fit_train_scales()

    def _check_complete_records(self, task: str) -> None:
        groups = {}
        for row in self.validation_rows[task]:
            groups.setdefault(row["pair_id"], []).append(row)
        for pair_id, rows in groups.items():
            indices = sorted(int(row["window_index"]) for row in rows)
            starts = sorted(int(row["start_sample_500hz"]) for row in rows)
            if indices != list(range(12)) or starts != list(range(0, 60000, 5000)):
                raise ValueError(f"{task} pair {pair_id} does not contain all 120 seconds")

    def _fit_train_scales(self) -> np.ndarray:
        # Per-lead RMS in raw microvolts, estimated from eligible training rows only.
        squares = np.zeros(8, dtype=np.float64)
        counts = np.zeros(8, dtype=np.int64)
        channels = (0,) + LEAD_INDICES
        for item in self.train_items:
            values = np.asarray(self.train_arrays[item.task][item.index, list(channels)], dtype=np.float64)
            eligible = np.concatenate(([True], item.mask7))
            squares[eligible] += np.square(values[eligible]).sum(axis=1)
            counts[eligible] += WINDOW
        if np.any(counts == 0):
            raise ValueError("some output lead has no eligible training target")
        return np.sqrt(squares / counts).clip(1.0, None).astype(np.float32)


class StrictTrainDataset(Dataset):
    def __init__(self, data: HuaweiSevenLeadData) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data.train_items)

    def __getitem__(self, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        item = self.data.train_items[position]
        raw = self.data.train_arrays[item.task][item.index]
        anchor = np.array(raw[0:1] / self.data.scale[0], dtype=np.float32)
        target = np.array(raw[list(LEAD_INDICES)] / self.data.scale[1:, None], dtype=np.float32)
        return anchor, target, item.mask7.astype(np.float32)
