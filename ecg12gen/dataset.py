"""D0 unified reader and D1 supervision safeguards.

This module consumes only the team's already-windowed NPY products through
memory maps. It does not reparse raw ECG files or modify source data.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import load_yaml_config, resolve_config_path
from .contracts import D12_LEADS, ECG_SAMPLING_RATE_HZ, WINDOW_SAMPLES, ContractError, ECGSample, SupervisionMode, canonical_lead_mask


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class ECGDataConfig:
    raw: dict[str, Any]
    repository_root: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ECGDataConfig":
        raw, root = load_yaml_config(config_path)
        return cls(raw=raw, repository_root=root)

    def path(self, key: str) -> Path:
        return resolve_config_path(self.raw, self.repository_root, key)

    @property
    def signal(self) -> dict[str, Any]:
        return self.raw["signal"]


class UnifiedECGDataset:
    """Framework-neutral indexed samples for one task, split, and mode.

    Default cross-device selection is conservative: only `paired` plus usable
    records are exposed. `review` and `unmatched` records cannot automatically
    enter training through this interface.
    """
    def __init__(self, config: ECGDataConfig | str | Path, task_id: str, split: str,
                 supervision_mode: str = SupervisionMode.CROSS_DEVICE_WEAK_ADAPTATION.value) -> None:
        self.config = ECGDataConfig.from_yaml(config) if not isinstance(config, ECGDataConfig) else config
        if task_id not in {"task1", "task2"}:
            raise ContractError("task_id must be task1 or task2")
        if split not in {"train", "validation"}:
            raise ContractError("split must be train or validation")
        self.task_id, self.split = task_id, split
        self.supervision_mode = SupervisionMode(supervision_mode)
        if self.supervision_mode != SupervisionMode.CROSS_DEVICE_WEAK_ADAPTATION and split != "train":
            raise ContractError("D12 reconstruction pretraining may use only split=train")
        self._validate_config()
        self._load_sources()

    def _validate_config(self) -> None:
        signal = self.config.signal
        if signal["ecg_sampling_rate_hz"] != ECG_SAMPLING_RATE_HZ:
            raise ContractError("D0 requires ECG sampling rate of 500 Hz")
        if signal["window_samples"] != WINDOW_SAMPLES or signal["window_seconds"] != 10:
            raise ContractError("D0 requires 10-second / 5000-point ECG windows")
        if tuple(signal["twelve_lead_order"]) != D12_LEADS:
            raise ContractError("D12 lead order differs from the required canonical order")
        if tuple(signal["six_lead_order"]) != D12_LEADS[:6]:
            raise ContractError("D6 lead order differs from the required limb-lead order")

    def _load_sources(self) -> None:
        task_dir = self.config.path(f"{self.task_id}_output")
        prefix = self.task_id
        self._inputs = np.load(task_dir / f"{prefix}_{self.split}_input.npy", mmap_mode="r")
        self._targets = np.load(task_dir / f"{prefix}_{self.split}_target.npy", mmap_mode="r")
        expected_channels = 1 if self.task_id == "task1" else 6
        if self._inputs.ndim != 3 or self._inputs.shape[1:] != (expected_channels, WINDOW_SAMPLES):
            raise ContractError(f"Unexpected {self.task_id} input array shape: {self._inputs.shape}")
        if self._targets.ndim != 3 or self._targets.shape[1:] != (12, WINDOW_SAMPLES):
            raise ContractError(f"Unexpected {self.task_id} target array shape: {self._targets.shape}")
        metadata_path = task_dir / f"{prefix}_window_metadata.csv"
        all_rows = _read_csv(metadata_path)
        self._rows = sorted((r for r in all_rows if r["split"] == self.split), key=lambda r: int(r["array_index"]))
        if len(self._rows) != len(self._inputs) or len(self._inputs) != len(self._targets):
            raise ContractError("Array rows and split metadata rows do not agree")
        if [int(r["array_index"]) for r in self._rows] != list(range(len(self._rows))):
            raise ContractError("array_index must be contiguous within each split")

        split_rows = _read_csv(self.config.path("subject_split_csv"))
        self._subject_split = {r["subject_id"]: r["split"] for r in split_rows}
        if len(self._subject_split) != len(split_rows):
            raise ContractError("subject_split.csv has duplicate subject_id entries")
        for row in self._rows:
            if self._subject_split.get(row["subject_id"]) != self.split:
                raise ContractError(f"Subject split mismatch for {row['subject_id']}")

        manifest_key = f"{self.task_id}_pair_manifest_csv"
        self._pairs = {r["pair_id"]: r for r in _read_csv(self.config.path(manifest_key))}
        self._indices = [i for i, row in enumerate(self._rows) if self._is_default_candidate(row)]

    def _is_default_candidate(self, row: dict[str, str]) -> bool:
        pair = self._pairs.get(row["pair_id"])
        if pair is None:
            raise ContractError(f"Window refers to an unknown pair_id: {row['pair_id']}")
        allowed = (pair.get("pair_status") == "paired" and pair.get("input_quality_status") == "usable"
                   and pair.get("target_quality_status") == "usable" and row.get("quality_status") == "usable")
        return allowed and pair.get("training_policy", "") not in {"review", "exclude", "drop"}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> ECGSample:
        array_index = self._indices[index]
        row, pair = self._rows[array_index], self._pairs[self._rows[array_index]["pair_id"]]
        target = np.asarray(self._targets[array_index], dtype=np.float32)
        if self.supervision_mode == SupervisionMode.D12_I_PRETRAIN:
            x_ecg, lead_mask, pairing_type, alignment_mode = target[:1], canonical_lead_mask(1), "within_d12_sync", "same_window"
        elif self.supervision_mode == SupervisionMode.D12_SIX_PRETRAIN:
            x_ecg, lead_mask, pairing_type, alignment_mode = target[:6], canonical_lead_mask(6), "within_d12_sync", "same_window"
        else:
            x_ecg = np.asarray(self._inputs[array_index], dtype=np.float32)
            lead_mask, pairing_type, alignment_mode = canonical_lead_mask(x_ecg.shape[0]), "reliable_subject_id_cross_device", "weak_subject_pair_record_start"
        device_type = pair.get("input_type") or row.get("input_type") or "watch_ecg"
        sample = ECGSample(
            X_ecg=x_ecg, lead_mask=lead_mask, Y_12lead=target, missing_mask=~lead_mask,
            task_id="task1" if x_ecg.shape[0] == 1 else "task2", ppg=None, acc=None,
            meta={"subject_id": row["subject_id"], "window_id": row["window_id"], "pair_id": row["pair_id"],
                  "device_type": device_type, "source_task_id": self.task_id,
                  "pointwise_mse_allowed": self.supervision_mode != SupervisionMode.CROSS_DEVICE_WEAK_ADAPTATION},
            modality_mask={"ppg": False, "acc": False}, split=self.split, supervision_mode=self.supervision_mode.value,
            pairing_type=pairing_type, alignment_mode=alignment_mode,
            pair_confidence=pair.get("pair_confidence", "not_applicable"), pair_status=pair.get("pair_status", "unknown"))
        sample.validate()
        return sample

    def __iter__(self) -> Iterator[ECGSample]:
        for index in range(len(self)):
            yield self[index]

    @property
    def excluded_rows(self) -> int:
        return len(self._rows) - len(self._indices)
