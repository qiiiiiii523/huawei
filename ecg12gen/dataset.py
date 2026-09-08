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
        if self.supervision_mode != SupervisionMode.CROSS_DEVICE_WEAK_ADAPTATION:
            raise ContractError("UnifiedECGDataset is for cross-device weak pairs only; use StrictD12PretrainDataset for de-duplicated d12 pretraining")
        self._validate_config()
        self._load_sources()

    def _validate_config(self) -> None:
        signal = self.config.signal
        if signal["ecg_sampling_rate_hz"] != ECG_SAMPLING_RATE_HZ:
            raise ContractError("D0 requires ECG sampling rate of 500 Hz")
        if signal["window_samples"] != WINDOW_SAMPLES or signal["window_seconds"] != 10:
            raise ContractError("D0 requires 10-second / 5000-point, non-overlapping ECG windows")
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
        x_ecg = np.asarray(self._inputs[array_index], dtype=np.float32)
        lead_mask, pairing_type, alignment_mode = canonical_lead_mask(x_ecg.shape[0]), "reliable_subject_id_cross_device", "weak_subject_pair_record_start"
        device_type = pair.get("input_type") or row.get("input_type") or "watch_ecg"
        sample = ECGSample(
            X_ecg=x_ecg, lead_mask=lead_mask, Y_12lead=target, missing_mask=~lead_mask,
            task_id="task1" if x_ecg.shape[0] == 1 else "task2", ppg=None, acc=None,
            meta={"subject_id": row["subject_id"], "window_id": row["window_id"], "pair_id": row["pair_id"],
                  "device_type": device_type, "source_task_id": self.task_id,
                  "alignment_quality_score": 0.0,
                  "pointwise_mse_allowed": False,
                  "pointwise_loss_allowed": False},
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


class JointAnchorDataset:
    """Read-only main joint-anchor dataset with explicit test-like I anchor."""

    def __init__(self, config: ECGDataConfig | str | Path, task_id: str, split: str,
                 body_scale_variant: str = "A_raw_window",
                 context_channel_indices: tuple[int, ...] | list[int] | None = None) -> None:
        self.config = ECGDataConfig.from_yaml(config) if not isinstance(config, ECGDataConfig) else config
        if task_id not in {"task1", "task2"} or split not in {"train", "validation"}:
            raise ContractError("JointAnchorDataset requires task1/task2 and train/validation")
        if body_scale_variant not in {"A_raw_window", "B_detrend_0p2Hz_then_window"}:
            raise ContractError("Unknown body-scale input variant")
        indices = tuple(range(6)) if context_channel_indices is None else tuple(context_channel_indices)
        if task_id == "task1" and context_channel_indices is not None:
            raise ContractError("task1 has a fixed single watch context channel")
        if task_id == "task2" and (not indices or len(set(indices)) != len(indices) or any(i not in range(6) for i in indices)):
            raise ContractError("task2 context_channel_indices must be unique canonical d6 indices")
        self.task_id, self.split, self.body_scale_variant = task_id, split, body_scale_variant
        self.context_channel_indices = (0,) if task_id == "task1" else indices
        self._validate_config(); self._load_sources()

    def _validate_config(self) -> None:
        signal = self.config.signal
        if signal["ecg_sampling_rate_hz"] != ECG_SAMPLING_RATE_HZ or signal["window_samples"] != WINDOW_SAMPLES or signal["window_seconds"] != 10:
            raise ContractError("Joint-anchor requires 500 Hz, 10 seconds, 5000 points")
        if tuple(signal["twelve_lead_order"]) != D12_LEADS or tuple(signal["six_lead_order"]) != D12_LEADS[:6]:
            raise ContractError("Joint-anchor requires canonical d12/d6 lead order")

    def _load_sources(self) -> None:
        task_dir, prefix = self.config.path(f"{self.task_id}_output"), self.task_id
        self._inputs = np.load(task_dir / f"{prefix}_{self.split}_input.npy", mmap_mode="r")
        self._targets = np.load(task_dir / f"{prefix}_{self.split}_target.npy", mmap_mode="r")
        channels = 1 if self.task_id == "task1" else 6
        if self._inputs.ndim != 3 or self._inputs.shape[1:] != (channels, WINDOW_SAMPLES) or self._targets.ndim != 3 or self._targets.shape[1:] != (12, WINDOW_SAMPLES):
            raise ContractError("Existing task arrays violate joint-anchor shapes")
        self._rows = sorted((r for r in _read_csv(task_dir / f"{prefix}_window_metadata.csv") if r["split"] == self.split), key=lambda r: int(r["array_index"]))
        if len(self._rows) != len(self._inputs) or len(self._rows) != len(self._targets) or [int(r["array_index"]) for r in self._rows] != list(range(len(self._rows))):
            raise ContractError("Array rows and split metadata do not agree")
        split_rows = _read_csv(self.config.path("subject_split_csv"))
        self._subject_split = {r["subject_id"]: r["split"] for r in split_rows}
        if len(self._subject_split) != len(split_rows):
            raise ContractError("subject_split.csv has duplicate subject_id")
        self._pairs = {r["pair_id"]: r for r in _read_csv(self.config.path(f"{self.task_id}_pair_manifest_csv"))}
        self._body_b_inputs, self._body_b_rows = None, {}
        if self.task_id == "task2" and self.body_scale_variant == "B_detrend_0p2Hz_then_window":
            ablation = self.config.path("task2_body_scale_ablation")
            self._body_b_inputs = np.load(ablation / f"body_scale_{self.split}_input_B_raw_detrended_0p2Hz.npy", mmap_mode="r")
            self._body_b_rows = {int(r["canonical_array_index"]): r for r in _read_csv(self.config.path("task2_body_scale_b_metadata")) if r["split"] == self.split}
        self._indices = [i for i, row in enumerate(self._rows) if self._eligible(row)]

    def _eligible(self, row: dict[str, str]) -> bool:
        pair = self._pairs.get(row["pair_id"])
        if self._subject_split.get(row["subject_id"]) != self.split or row.get("quality_status") != "usable" or not pair:
            return False
        if pair.get("pair_status") != "paired" or pair.get("input_quality_status") != "usable" or pair.get("target_quality_status") != "usable" or pair.get("training_policy") in {"review", "exclude", "drop"}:
            return False
        if pair.get("subject_id") != row["subject_id"] or pair.get("target_record_id") != row["target_record_id"] or pair.get("split") != self.split:
            raise ContractError("Context and target must retain same subject/split target pair identity")
        return not (self.task_id == "task2" and self.body_scale_variant == "B_detrend_0p2Hz_then_window" and row.get("input_type") == "body_scale_d6" and int(row["array_index"]) not in self._body_b_rows)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        from .contracts import JointAnchorSample
        array_index = self._indices[index]; row = self._rows[array_index]; pair = self._pairs[row["pair_id"]]
        input_type = row.get("input_type") or pair.get("input_type") or "watch_ecg"
        context = np.asarray(self._inputs[array_index], dtype=np.float32)
        meta: dict[str, Any] = {"anchor_construction": "simulated_from_target_i_for_test_available_input",
            "anchor_target_record_id": row["target_record_id"], "anchor_window_id": row["window_id"],
            "anchor_start_sample_500hz": row["start_sample_500hz"], "anchor_end_sample_500hz_exclusive": row["end_sample_500hz_exclusive"],
            "context_record_id": row["input_record_id"], "context_window_id": row["window_id"],
            "context_window_relation": "pair_row_window_index_only_not_time_sync"}
        if self.task_id == "task2" and input_type == "body_scale_d6" and self.body_scale_variant == "B_detrend_0p2Hz_then_window":
            b_row = self._body_b_rows[array_index]; context = np.asarray(self._body_b_inputs[int(b_row["local_array_index"])], dtype=np.float32); meta["input_processing_variant"] = self.body_scale_variant
        elif self.task_id == "task2":
            meta["input_processing_variant"] = "A_raw_window" if input_type == "body_scale_d6" else "not_applicable"
        if self.task_id == "task2":
            context = context[list(self.context_channel_indices)]; context_mask = np.zeros(6, dtype=bool); context_mask[list(self.context_channel_indices)] = True
        else:
            context_mask = np.ones(1, dtype=bool)
        target = np.asarray(self._targets[array_index], dtype=np.float32)
        sample = JointAnchorSample(context_ecg=context, context_source_type=input_type, anchor_i_ecg=target[:1].copy(), anchor_source_type="ecg_machine_i", Y_12lead=target, anchor_lead_mask=canonical_lead_mask(1), context_lead_mask=context_mask, task_id=self.task_id, split=self.split, subject_id=row["subject_id"], pair_id=row["pair_id"], target_record_id=row["target_record_id"], window_id=row["window_id"], meta=meta, input_type=input_type)
        sample.validate(); return sample
