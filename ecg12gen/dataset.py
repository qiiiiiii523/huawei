"""Read-only joint-anchor datasets over existing windowed NPY products."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import numpy as np

from .config import load_yaml_config, resolve_config_path
from .contracts import D12_LEADS, ECG_SAMPLING_RATE_HZ, WINDOW_SAMPLES, ContractError, JointAnchorSample, canonical_lead_mask
from .device_qc import d12_target_mask, d6_input_mask, load_device_interpretation_qc
from .record_baseline import build_record_baselines

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def _baseline_from_text(value: str, expected_leads: int, *, field: str) -> np.ndarray:
    values = np.fromstring(value, sep="|", dtype=np.float32)
    if values.shape != (expected_leads,) or not np.isfinite(values).all():
        raise ContractError(f"{field} must contain {expected_leads} finite record-baseline values")
    return values

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

class LegacyJointAnchorDataset:
    """Quality-gated context plus same-record/window d12-I anchor samples.

    Training and validation construct the anchor from target I solely to
    simulate the organizer-provided test input. Existing arrays are memory
    mapped read-only and never re-windowed or rewritten.
    """
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
        context_time_mask_path = task_dir / f"{prefix}_{self.split}_context_valid_mask.npy"
        self._context_time_masks = (np.load(context_time_mask_path, mmap_mode="r")
                                    if context_time_mask_path.is_file() else None)
        channels = 1 if self.task_id == "task1" else 6
        if self._inputs.ndim != 3 or self._inputs.shape[1:] != (channels, WINDOW_SAMPLES) or self._targets.ndim != 3 or self._targets.shape[1:] != (12, WINDOW_SAMPLES):
            raise ContractError("Existing task arrays violate joint-anchor shapes")
        if self._context_time_masks is not None and self._context_time_masks.shape != (len(self._inputs), WINDOW_SAMPLES):
            raise ContractError("Context validity mask must have shape [N,5000]")
        self._rows = sorted((r for r in _read_csv(task_dir / f"{prefix}_window_metadata.csv") if r["split"] == self.split), key=lambda r: int(r["array_index"]))
        if len(self._rows) != len(self._inputs) or len(self._rows) != len(self._targets) or [int(r["array_index"]) for r in self._rows] != list(range(len(self._rows))):
            raise ContractError("Array rows and split metadata do not agree")
        self._device_qc = load_device_interpretation_qc(self.config.path("device_interpretation_qc_csv"))
        split_rows = _read_csv(self.config.path("subject_split_csv"))
        self._subject_split = {r["subject_id"]: r["split"] for r in split_rows}
        if len(self._subject_split) != len(split_rows): raise ContractError("subject_split.csv has duplicate subject_id")
        self._pairs = {r["pair_id"]: r for r in _read_csv(self.config.path(f"{self.task_id}_pair_manifest_csv"))}
        self._body_b_inputs, self._body_b_rows = None, {}
        if self.task_id == "task2" and self.body_scale_variant == "B_detrend_0p2Hz_then_window":
            ablation = self.config.path("task2_body_scale_ablation")
            self._body_b_inputs = np.load(ablation / f"body_scale_{self.split}_input_B_raw_detrended_0p2Hz.npy", mmap_mode="r")
            self._body_b_rows = {int(r["canonical_array_index"]): r for r in _read_csv(self.config.path("task2_body_scale_b_metadata")) if r["split"] == self.split}
        baseline_rows = [{**row, "input_type": row.get("input_type") or ("watch_ecg" if self.task_id == "task1" else "")}
                         for row in self._rows]
        if self._rows and all(row.get("target_record_baseline_uV") for row in self._rows):
            self._target_record_baselines = {}
            self._context_record_baselines = {}
            for row in baseline_rows:
                target_baseline = _baseline_from_text(
                    row["target_record_baseline_uV"], 12, field="target_record_baseline_uV")
                context_baseline = _baseline_from_text(
                    row["input_record_baseline_uV"], channels, field="input_record_baseline_uV")
                previous_target = self._target_record_baselines.setdefault(
                    row["target_record_id"], target_baseline)
                context_key = (row["input_type"], row["input_record_id"])
                previous_context = self._context_record_baselines.setdefault(
                    context_key, context_baseline)
                if not np.array_equal(previous_target, target_baseline) or not np.array_equal(previous_context, context_baseline):
                    raise ContractError("One physical record has inconsistent stored baselines")
        else:
            self._target_record_baselines = build_record_baselines(
                self._targets, self._rows, record_id_field="target_record_id")
            self._context_record_baselines = build_record_baselines(
                self._inputs, baseline_rows, record_id_field="input_record_id", source_type_field="input_type")
        if self._body_b_inputs is not None:
            body_rows = sorted(self._body_b_rows.values(), key=lambda item: int(item["local_array_index"]))
            self._context_record_baselines.update(build_record_baselines(
                self._body_b_inputs, body_rows, record_id_field="input_record_id", source_type_field="input_type"))
        self._indices = [i for i, row in enumerate(self._rows) if self._eligible(row)]

    def _target_qc(self, row: dict[str, str]) -> dict[str, str]:
        qc = self._device_qc.get(row["target_record_id"])
        if not qc or qc["device_type"] != "ecg_machine_d12":
            raise ContractError("Every d12 target must have a device-interpretation QC row")
        return qc

    def _eligible(self, row: dict[str, str]) -> bool:
        pair = self._pairs.get(row["pair_id"])
        if self._subject_split.get(row["subject_id"]) != self.split or row.get("quality_status") != "usable" or not pair:
            return False
        if pair.get("pair_status") != "paired" or pair.get("input_quality_status") != "usable" or pair.get("target_quality_status") != "usable" or pair.get("training_policy") in {"review", "exclude", "drop"}:
            return False
        if pair.get("subject_id") != row["subject_id"] or pair.get("target_record_id") != row["target_record_id"] or pair.get("split") != self.split:
            raise ContractError("Context and target must retain the same subject/split target pair identity")
        target_qc = self._target_qc(row)
        if self.split == "train" and target_qc["d12_direct_supervision_eligible"] != "true":
            return False
        # Record-complete caches retain validation windows whose optional
        # context contains a filled timestamp gap.  Do not teach the context
        # encoder from those synthetic spans, but keep validation complete.
        if self.split == "train" and row.get("context_quality_status", "usable") != "usable":
            return False
        if self.task_id == "task2" and row.get("input_type") == "ecg_machine_d6":
            input_qc = self._device_qc.get(row["input_record_id"])
            if not input_qc or input_qc["device_type"] != "ecg_machine_d6":
                raise ContractError("Every ecg_machine_d6 context must have a device-interpretation QC row")
            if self.split == "train" and input_qc["d6_context_training_eligible"] != "true":
                return False
        return not (self.task_id == "task2" and self.body_scale_variant == "B_detrend_0p2Hz_then_window" and row.get("input_type") == "body_scale_d6" and int(row["array_index"]) not in self._body_b_rows)

    def __len__(self) -> int: return len(self._indices)
    def __iter__(self) -> Iterator[JointAnchorSample]:
        for index in range(len(self)): yield self[index]
    @property
    def excluded_rows(self) -> int: return len(self._rows) - len(self._indices)

    def __getitem__(self, index: int) -> JointAnchorSample:
        array_index = self._indices[index]; row = self._rows[array_index]; pair = self._pairs[row["pair_id"]]
        input_type = row.get("input_type") or pair.get("input_type") or "watch_ecg"
        context = np.asarray(self._inputs[array_index], dtype=np.float32)
        meta: dict[str, Any] = {
            "anchor_construction": "simulated_from_target_i_for_test_available_input",
            "anchor_target_record_id": row["target_record_id"], "anchor_window_id": row["window_id"],
            "anchor_start_sample_500hz": row["start_sample_500hz"], "anchor_end_sample_500hz_exclusive": row["end_sample_500hz_exclusive"],
            "context_record_id": row["input_record_id"], "context_window_id": row["window_id"],
            "context_window_relation": "pair_row_window_index_only_not_time_sync"}
        if row.get("expected_window_count"):
            meta["expected_window_count"] = int(row["expected_window_count"])
        if row.get("context_valid_fraction"):
            meta["context_valid_fraction"] = float(row["context_valid_fraction"])
        if self.task_id == "task2" and input_type == "body_scale_d6" and self.body_scale_variant == "B_detrend_0p2Hz_then_window":
            b_row = self._body_b_rows[array_index]
            context = np.asarray(self._body_b_inputs[int(b_row["local_array_index"])], dtype=np.float32)
            meta["input_processing_variant"] = self.body_scale_variant
        elif self.task_id == "task2": meta["input_processing_variant"] = "A_raw_window" if input_type == "body_scale_d6" else "not_applicable"
        if self.task_id == "task2":
            context = context[list(self.context_channel_indices)]
            context_mask = np.zeros(6, dtype=bool); context_mask[list(self.context_channel_indices)] = True
        else: context_mask = np.ones(1, dtype=bool)
        target = np.asarray(self._targets[array_index], dtype=np.float32)
        context_time_mask = (np.asarray(self._context_time_masks[array_index], dtype=bool)[None, :]
                             if self._context_time_masks is not None
                             else np.ones(context.shape, dtype=bool))
        target_record_baseline = self._target_record_baselines[row["target_record_id"]]
        context_record_baseline = self._context_record_baselines[(input_type, row["input_record_id"])][list(self.context_channel_indices)]
        target_qc = self._target_qc(row)
        target_quality_mask = d12_target_mask(target_qc)
        input_quality_mask = np.ones(context.shape[0], dtype=bool)
        if self.task_id == "task2" and input_type == "ecg_machine_d6":
            input_qc = self._device_qc[row["input_record_id"]]
            input_quality_mask = d6_input_mask(input_qc)[list(self.context_channel_indices)]
            meta["input_device_qc_warning"] = input_qc["has_signal_quality_warning"]
            meta["input_bad_observed_leads"] = input_qc["bad_observed_input_leads"]
        meta["target_device_qc_warning"] = target_qc["has_signal_quality_warning"]
        meta["target_bad_leads"] = target_qc["bad_leads_all"]
        sample = JointAnchorSample(context_ecg=context, context_source_type=input_type, anchor_i_ecg=target[:1].copy(), anchor_source_type="ecg_machine_i", Y_12lead=target, anchor_lead_mask=canonical_lead_mask(1), context_lead_mask=context_mask, task_id=self.task_id, split=self.split, subject_id=row["subject_id"], pair_id=row["pair_id"], target_record_id=row["target_record_id"], window_id=row["window_id"], target_record_baseline_uV=target_record_baseline.copy(), context_record_baseline_uV=context_record_baseline.copy(), meta=meta, input_type=input_type, target_quality_mask=target_quality_mask, input_quality_mask=input_quality_mask, context_time_mask=context_time_mask)
        sample.validate(); return sample


# Default public interface uses independent records; legacy is explicit audit-only.
from .record_dataset import JointAnchorDataset  # noqa: E402
