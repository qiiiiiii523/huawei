"""B4 adapters over main's frozen strict-P0 and validation datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import SupervisionMode
from .d12_pretrain import StrictD12PretrainDataset
from .dataset import ECGDataConfig, JointAnchorDataset
from .preprocessing import ECGPreprocessor, PreprocessingConfig


@dataclass(frozen=True)
class B4Item:
    anchor_model: torch.Tensor
    missing_target_model: torch.Tensor
    target_quality_mask: torch.Tensor
    raw_target_uV: torch.Tensor
    raw_anchor_uV: torch.Tensor
    meta: dict[str, Any]


class B4PreparedDataset(Dataset[B4Item]):
    """Expose only synchronized machine-I and its same-window d12 target."""

    def __init__(self, samples: Sequence[Any], preprocessor: ECGPreprocessor, *, training: bool) -> None:
        self.samples = samples
        self.preprocessor = preprocessor
        self.training = training
        if not samples:
            raise ValueError("B4 dataset cannot be empty")
        for sample in samples:
            sample.validate()
            if training and sample.supervision_mode != SupervisionMode.D12_I_PRETRAIN.value:
                raise ValueError("B4 training must use main's strict d12_i_pretrain samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> B4Item:
        sample = self.samples[index]
        target_raw = np.asarray(sample.Y_12lead, dtype=np.float32)
        anchor_raw = np.asarray(
            sample.X_ecg[:1] if self.training else sample.anchor_i_ecg,
            dtype=np.float32,
        ).copy()
        if not np.array_equal(anchor_raw, target_raw[:1]):
            raise ValueError("B4 requires exact same-window machine-I conditioning")
        target_model = self.preprocessor.transform_d12_target(target_raw).model_signal
        anchor_model = self.preprocessor.transform_window(anchor_raw, "ecg_machine_i").model_signal
        quality = sample.target_quality_mask
        if quality is None:
            quality = np.ones(12, dtype=bool)
        return B4Item(
            anchor_model=torch.from_numpy(anchor_model),
            missing_target_model=torch.from_numpy(target_model[1:].copy()),
            target_quality_mask=torch.from_numpy(np.asarray(quality[1:], dtype=bool).copy()),
            raw_target_uV=torch.from_numpy(target_raw.copy()),
            raw_anchor_uV=torch.from_numpy(anchor_raw),
            meta=dict(sample.meta),
        )


def b4_collate(items: list[B4Item]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty B4 batch")
    return {
        "anchor_model": torch.stack([item.anchor_model for item in items]),
        "missing_target_model": torch.stack([item.missing_target_model for item in items]),
        "target_quality_mask": torch.stack([item.target_quality_mask for item in items]),
        "raw_target_uV": torch.stack([item.raw_target_uV for item in items]),
        "raw_anchor_uV": torch.stack([item.raw_anchor_uV for item in items]),
        "meta": [item.meta for item in items],
    }


def fit_b4_preprocessor(config_path: str | Path, task_id: str) -> ECGPreprocessor:
    config = ECGDataConfig.from_yaml(config_path)
    preprocessing = PreprocessingConfig.from_yaml(config.repository_root / "configs" / "preprocessing.yaml")
    train_targets = np.load(
        config.path(f"{task_id}_output") / f"{task_id}_train_target.npy",
        mmap_mode="r",
    )
    return ECGPreprocessor.fit(preprocessing, {"d12": np.asarray(train_targets, dtype=np.float32)})


def build_b4_datasets(
    config_path: str | Path,
    task_id: str,
    preprocessor: ECGPreprocessor,
) -> tuple[B4PreparedDataset, B4PreparedDataset]:
    config = ECGDataConfig.from_yaml(config_path)
    strict = StrictD12PretrainDataset(config, SupervisionMode.D12_I_PRETRAIN.value)
    validation = JointAnchorDataset(config, task_id, "validation")
    return (
        B4PreparedDataset(strict, preprocessor, training=True),
        B4PreparedDataset(validation, preprocessor, training=False),
    )
