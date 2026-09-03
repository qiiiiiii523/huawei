"""Indexed train-only d12 windows for strict self-supervised pretraining."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import numpy as np

from .contracts import ContractError, ECGSample, SupervisionMode, canonical_lead_mask
from .dataset import ECGDataConfig


class StrictD12PretrainDataset:
    """Read only the de-duplicated train d12 index; validation is impossible."""

    def __init__(self, config: ECGDataConfig | str | Path, mode: str) -> None:
        self.config = ECGDataConfig.from_yaml(config) if not isinstance(config, ECGDataConfig) else config
        if mode not in {SupervisionMode.D12_I_PRETRAIN.value, SupervisionMode.D12_SIX_PRETRAIN.value}:
            raise ContractError("Strict d12 dataset mode must be d12_i_pretrain or d12_six_pretrain")
        self.mode = SupervisionMode(mode)
        index_path = self.config.repository_root / "metadata" / "d12_strict_pretrain_index.csv"
        with index_path.open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows or any(row.get("source_split") != "train" for row in self.rows):
            raise ContractError("Strict d12 index must be non-empty and train-only")
        self.targets = {
            task: np.load(self.config.path(f"{task}_output") / f"{task}_train_target.npy", mmap_mode="r")
            for task in {row["source_task_id"] for row in self.rows}
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ECGSample:
        row = self.rows[index]
        target = np.asarray(self.targets[row["source_task_id"]][int(row["source_array_index"])], dtype=np.float32)
        channels = 1 if self.mode == SupervisionMode.D12_I_PRETRAIN else 6
        sample = ECGSample(
            X_ecg=target[:channels], lead_mask=canonical_lead_mask(channels), Y_12lead=target,
            missing_mask=~canonical_lead_mask(channels), task_id="task1" if channels == 1 else "task2",
            ppg=None, acc=None, meta={"subject_id": row["subject_id"], "window_id": row["window_id"], "strict_id": row["strict_id"], "target_record_id": row["target_record_id"], "alignment_quality_score": 1.0, "pointwise_loss_allowed": True},
            modality_mask={"ppg": False, "acc": False}, split="train", supervision_mode=self.mode.value,
            pairing_type="within_d12_sync", alignment_mode="same_window", pair_confidence="not_applicable", pair_status="paired",
        )
        sample.validate()
        return sample

    def __iter__(self) -> Iterator[ECGSample]:
        for index in range(len(self)):
            yield self[index]