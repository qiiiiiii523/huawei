"""可审计、模型无关的 ECG 动态预处理提案。

本模块永远不写回原始数组。所有尺度仅可用训练集拟合；验证和推理只
应用已冻结的统计量。不同设备有不同输入 transform，d12 target 始终使用
同一个 canonical ``d12`` transform。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


class PreprocessingError(ValueError):
    """Raised when a model-space transformation violates the protocol."""


@dataclass(frozen=True)
class PreprocessingConfig:
    baseline_method: str
    scaling_method: str
    minimum_scale_uV: float
    clip_model_signal: float
    expected_leads: dict[str, int]
    target_transform: str = "d12"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PreprocessingConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw.get("raw_data_mutation") is not False:
            raise PreprocessingError("The preprocessing contract must not mutate raw data")
        if raw["baseline"]["method"] != "per_window_per_lead_median":
            raise PreprocessingError("Only the proposed robust-median baseline method is supported")
        if raw["scaling"]["method"] != "train_median_p5_p95_range" or raw["scaling"]["fit_split"] != "train":
            raise PreprocessingError("Scale must be fitted from train with the protocol method")
        return cls(
            baseline_method=raw["baseline"]["method"],
            scaling_method=raw["scaling"]["method"],
            minimum_scale_uV=float(raw["scaling"]["minimum_scale_uV"]),
            clip_model_signal=float(raw["scaling"]["clip_model_signal"]),
            expected_leads={name: int(spec["expected_leads"]) for name, spec in raw["sources"].items()},
            target_transform=str(raw.get("target_transform", "d12")),
        )


@dataclass(frozen=True)
class ModelSignal:
    """A non-destructive model view of one [lead, time] ECG window."""

    model_signal: np.ndarray
    baseline_uV: np.ndarray
    scale_uV: np.ndarray
    source_type: str


@dataclass(frozen=True)
class ECGPreprocessor:
    """Frozen train-fitted device/lead transforms.

    A separate instance is not needed per task: source type selects watch,
    machine d6, body-scale d6, or canonical d12 statistics.
    """

    config: PreprocessingConfig
    scale_uV_by_source: dict[str, np.ndarray]

    @classmethod
    def fit(cls, config: PreprocessingConfig, train_signals: Mapping[str, np.ndarray]) -> "ECGPreprocessor":
        """Fit one robust scale per source and lead from training arrays only.

        Each array must have shape [N, C, T]. Callers must pass only the fixed
        train split; the API deliberately has no validation fitting path.
        """
        if config.target_transform not in train_signals:
            raise PreprocessingError("Every task fit must include training d12 targets")
        scales: dict[str, np.ndarray] = {}
        for source, values in train_signals.items():
            expected_leads = config.expected_leads.get(source)
            if expected_leads is None:
                raise PreprocessingError(f"Unknown source in train_signals: {source!r}")
            array = np.asarray(values)
            if array.ndim != 3 or array.shape[1] != expected_leads or array.shape[0] == 0:
                raise PreprocessingError(f"{source} train array must be non-empty [N, {expected_leads}, T]")
            if not np.isfinite(array).all():
                raise PreprocessingError(f"{source} contains non-finite training values")
            window_ranges = np.percentile(array, 95, axis=2) - np.percentile(array, 5, axis=2)
            scale = np.maximum(np.median(window_ranges, axis=0), config.minimum_scale_uV).astype(np.float32)
            scales[source] = scale
        return cls(config=config, scale_uV_by_source=scales)

    def transform_window(self, raw_window: np.ndarray, source_type: str) -> ModelSignal:
        """Return a centered, scaled, clipped model view without mutating input."""
        if source_type not in self.config.expected_leads:
            raise PreprocessingError(f"Unknown source type: {source_type}")
        raw = np.asarray(raw_window)
        expected = self.config.expected_leads[source_type]
        if raw.ndim != 2 or raw.shape[0] != expected:
            raise PreprocessingError(f"{source_type} window must have shape [{expected}, T]")
        if not np.isfinite(raw).all():
            raise PreprocessingError("Cannot transform non-finite ECG values")
        baseline = np.median(raw, axis=1).astype(np.float32)
        if source_type not in self.scale_uV_by_source:
            raise PreprocessingError(f"No frozen train scale for source type: {source_type}")
        scale = self.scale_uV_by_source[source_type]
        transformed = (raw.astype(np.float32, copy=False) - baseline[:, None]) / scale[:, None]
        transformed = np.clip(transformed, -self.config.clip_model_signal, self.config.clip_model_signal)
        return ModelSignal(model_signal=transformed, baseline_uV=baseline, scale_uV=scale.copy(), source_type=source_type)

    def transform_d12_target(self, raw_d12: np.ndarray) -> ModelSignal:
        """Apply the same canonical d12 transform in every task and pretraining mode."""
        return self.transform_window(raw_d12, self.config.target_transform)

    def transform_batch(self, raw_batch: np.ndarray, source_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized non-destructive transform for [N, C, T] arrays."""
        raw = np.asarray(raw_batch)
        expected = self.config.expected_leads.get(source_type)
        if expected is None or raw.ndim != 3 or raw.shape[1] != expected:
            raise PreprocessingError(f"{source_type} batch must have shape [N, {expected}, T]")
        if not np.isfinite(raw).all():
            raise PreprocessingError("Cannot transform non-finite ECG values")
        baseline = np.median(raw, axis=2).astype(np.float32)
        if source_type not in self.scale_uV_by_source:
            raise PreprocessingError(f"No frozen train scale for source type: {source_type}")
        scale = self.scale_uV_by_source[source_type]
        model = (raw.astype(np.float32, copy=False) - baseline[:, :, None]) / scale[None, :, None]
        return np.clip(model, -self.config.clip_model_signal, self.config.clip_model_signal), baseline, np.broadcast_to(scale, baseline.shape).copy()
