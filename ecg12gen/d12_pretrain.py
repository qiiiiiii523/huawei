"""Indexed train-only d12 windows for strict self-supervised pretraining."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import numpy as np

from .contracts import ContractError, ECGSample, SupervisionMode, canonical_lead_mask
from .dataset import ECGDataConfig
from .device_qc import d12_target_mask, load_device_interpretation_qc
from .record_baseline import build_record_baselines


class StrictD12PretrainDataset:
    """Read only the de-duplicated train d12 index; validation is impossible."""

    def __init__(self, config: ECGDataConfig | str | Path, mode: str) -> None:
        self.config = ECGDataConfig.from_yaml(config) if not isinstance(config, ECGDataConfig) else config
        if mode != SupervisionMode.D12_I_PRETRAIN.value:
            raise ContractError("Strict d12 dataset mode must be d12_i_pretrain")
        self.mode = SupervisionMode(mode)
        index_path = self.config.repository_root / "metadata" / "d12_strict_pretrain_index.csv"
        with index_path.open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self._device_qc = load_device_interpretation_qc(self.config.path("device_interpretation_qc_csv"))
        self.rows = [row for row in self.rows if self._direct_supervision_eligible(row)]
        if not self.rows or any(row.get("source_split") != "train" for row in self.rows):
            raise ContractError("Strict d12 index must be non-empty and train-only")
        self.targets = {
            task: np.load(self.config.path(f"{task}_output") / f"{task}_train_target.npy", mmap_mode="r")
            for task in {row["source_task_id"] for row in self.rows}
        }
        # A new cache changes array ordering. Never silently apply an old index.
        for task, targets in self.targets.items():
            task_dir = self.config.path(f"{task}_output")
            if (task_dir / f"{task}_context_window_metadata.csv").is_file():
                with (task_dir / f"{task}_window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
                    current = {int(r["array_index"]): r for r in csv.DictReader(handle) if r["split"] == "train"}
                for row in (r for r in self.rows if r["source_task_id"] == task):
                    cached = current.get(int(row["source_array_index"]))
                    if (not cached or cached["target_record_id"] != row["target_record_id"] or
                            str(cached.get("target_physical_start_sample_500hz", cached["start_sample_500hz"])) != row["start_sample_500hz"]):
                        raise ContractError("Strict index is stale for the independent cache; rebuild d12_strict_pretrain_index.csv")
        self._record_baselines: dict[str, np.ndarray] = {}
        for row in self.rows:
            text = row.get("target_record_baseline_uV", "")
            if not text:
                continue
            baseline = np.fromstring(text, sep="|", dtype=np.float32)
            if baseline.shape != (12,) or not np.isfinite(baseline).all():
                raise ContractError("Strict index contains an invalid stored target baseline")
            previous = self._record_baselines.setdefault(row["target_record_id"], baseline)
            if not np.array_equal(previous, baseline):
                raise ContractError("Strict index contains inconsistent physical-record baselines")
        for task, targets in self.targets.items():
            task_rows = [row for row in self.rows if row["source_task_id"] == task and
                         row["target_record_id"] not in self._record_baselines]
            if not task_rows:
                continue
            task_windows = np.asarray(targets[[int(row["source_array_index"]) for row in task_rows]])
            normalized_rows = [{**row, "start_sample_500hz": row.get("start_sample_500hz") or
                                str(int(row.get("window_index", "0")) * 5000)} for row in task_rows]
            self._record_baselines.update(build_record_baselines(
                task_windows, normalized_rows, record_id_field="target_record_id"))

    def _direct_supervision_eligible(self, row: dict[str, str]) -> bool:
        qc = self._device_qc.get(row["target_record_id"])
        if not qc or qc["device_type"] != "ecg_machine_d12":
            raise ContractError("Strict d12 row lacks a d12 device-interpretation QC record")
        return qc["d12_direct_supervision_eligible"] == "true"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ECGSample:
        row = self.rows[index]
        target = np.asarray(self.targets[row["source_task_id"]][int(row["source_array_index"])], dtype=np.float32)
        channels = 1
        target_qc = self._device_qc[row["target_record_id"]]
        sample = ECGSample(
            X_ecg=target[:channels], lead_mask=canonical_lead_mask(channels), Y_12lead=target,
            missing_mask=~canonical_lead_mask(channels), task_id="task1",
            ppg=None, acc=None, meta={"subject_id": row["subject_id"], "window_id": row["window_id"], "strict_id": row["strict_id"], "target_record_id": row["target_record_id"], "alignment_quality_score": 1.0, "pointwise_loss_allowed": True},
            modality_mask={"ppg": False, "acc": False}, split="train", supervision_mode=self.mode.value,
            pairing_type="within_d12_sync", alignment_mode="same_window", pair_confidence="not_applicable", pair_status="paired",
            target_quality_mask=d12_target_mask(target_qc), input_quality_mask=np.ones(channels, dtype=bool),
            target_record_baseline_uV=self._record_baselines[row["target_record_id"]].copy(),
        )
        sample.validate()
        return sample

    def __iter__(self) -> Iterator[ECGSample]:
        for index in range(len(self)):
            yield self[index]
