from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.dataset import ECGDataConfig, UnifiedECGDataset
from ecg12gen.preprocessing import (
    ECGPreprocessor,
    PreprocessingConfig,
)


def load_frozen_preprocessor(
    data_config: ECGDataConfig,
    scales_path: str | Path,
) -> ECGPreprocessor:
    """只加载已有训练尺度，不重新拟合。"""
    config = PreprocessingConfig.from_yaml(
        data_config.path("preprocessing_config")
    )

    with np.load(scales_path, allow_pickle=False) as data:
        scales = {
            key: np.asarray(data[key], dtype=np.float32).copy()
            for key in data.files
        }

    for source, channels in config.expected_leads.items():
        if source not in scales:
            raise ValueError(f"Missing scale: {source}")

        scale = scales[source]
        if scale.shape != (channels,):
            raise ValueError(f"Invalid scale shape: {source}")

        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"Invalid scale values: {source}")

    return ECGPreprocessor(
        config=config,
        scale_uV_by_source=scales,
    )


def to_tensor(array: np.ndarray) -> torch.Tensor:
    # 独立副本，避免修改原始数据或只读内存映射。
    return torch.from_numpy(np.array(array, copy=True))


class B0RouteDataset(Dataset):
    """
    strict：严格同步D12，仅允许train。
    weak：跨设备弱配对，支持train/validation。

    当前weak使用原始数据版本A。
    """

    def __init__(
        self,
        data_config: ECGDataConfig,
        preprocessor: ECGPreprocessor,
        task_id: str,
        split: str,
        mode: str,
    ) -> None:
        if task_id not in {"task1", "task2"}:
            raise ValueError("task_id must be task1 or task2")

        self.task_id = task_id
        self.split = split
        self.mode = mode
        self.preprocessor = preprocessor
        self.channels = 1 if task_id == "task1" else 6

        if mode == "strict":
            if split != "train":
                raise ValueError(
                    "Strict pretraining reader is train-only"
                )

            strict_mode = (
                "d12_i_pretrain"
                if task_id == "task1"
                else "d12_six_pretrain"
            )
            self.source = StrictD12PretrainDataset(
                data_config, mode=strict_mode
            )

        elif mode == "weak":
            self.source = UnifiedECGDataset(
                data_config,
                task_id=task_id,
                split=split,
            )
        else:
            raise ValueError("mode must be strict or weak")

        if len(self.source) == 0:
            raise ValueError("No usable windows found")

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict:
        sample = self.source[index]

        target = self.preprocessor.transform_d12_target(
            sample.Y_12lead
        ).model_signal

        d12_scale = self.preprocessor.scale_uV_by_source["d12"]

        if self.mode == "strict":
            # 与旧Ridge保持相同的严格同步输入定义。
            inputs = target[:self.channels].copy()
            observed_visible = inputs.copy()

            source_type = "d12"
            alignment_mode = "same_window"
            pointwise_allowed = True

        else:
            source_type = str(sample.meta["device_type"])

            transformed = self.preprocessor.transform_window(
                sample.X_ecg,
                source_type=source_type,
            )
            inputs = transformed.model_signal

            # 网络输入仍使用各设备自己的尺度。
            # observed loss必须换算到输出所用的d12尺度。
            observed_visible = (
                inputs
                * transformed.scale_uV[:, None]
                / d12_scale[:self.channels, None]
            ).astype(np.float32)

            alignment_mode = sample.alignment_mode
            pointwise_allowed = bool(
                sample.meta["pointwise_loss_allowed"]
            )

            if pointwise_allowed:
                raise ValueError(
                    "Weak pairs must forbid pointwise target loss"
                )

        observed = np.zeros_like(target)
        observed[:self.channels] = observed_visible

        return {
            "inputs": to_tensor(inputs),
            "target": to_tensor(target),
            "observed_input": to_tensor(observed),
            "lead_mask": to_tensor(sample.lead_mask),
            "missing_mask": to_tensor(sample.missing_mask),
            "raw_input_uV": to_tensor(sample.X_ecg),
            "raw_target_uV": to_tensor(sample.Y_12lead),
            "subject_id": str(sample.meta["subject_id"]),
            "window_id": str(sample.meta["window_id"]),
            "device_type": source_type,
            "alignment_mode": alignment_mode,
            "pointwise_loss_allowed": pointwise_allowed,
            "split": self.split,
        }