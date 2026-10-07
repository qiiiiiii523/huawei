"""可审计、模型无关的 ECG 动态预处理协议。

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
    clip_model_signal: float | None
    expected_leads: dict[str, int]
    target_transform: str = "d12"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PreprocessingConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw.get("raw_data_mutation") is not False:
            raise PreprocessingError("The preprocessing contract must not mutate raw data")
        if raw["baseline"]["method"] != "context_only_per_record_per_lead_median":
            raise PreprocessingError("Expected v3 context-only record centering; old centered-target checkpoints are incompatible")
        if set(raw["baseline"].get("scale_only_sources", [])) != {"ecg_machine_i", "d12"}:
            raise PreprocessingError("Anchor I and d12 must preserve raw voltage")
        if raw["scaling"].get("clip_model_signal") is not None:
            raise PreprocessingError("The reversible raw-voltage protocol forbids amplitude clipping")
        if raw["scaling"]["method"] != "train_median_p5_p95_range" or raw["scaling"]["fit_split"] != "train":
            raise PreprocessingError("Scale must be fitted from train with the protocol method")
        return cls(
            baseline_method=raw["baseline"]["method"],
            scaling_method=raw["scaling"]["method"],
            minimum_scale_uV=float(raw["scaling"]["minimum_scale_uV"]),
            clip_model_signal=None,
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

    def save(self, path: str | Path) -> None:
        """Persist train-fitted scales with the output protocol for inference."""
        with Path(path).open("wb") as handle:
            np.savez(handle, protocol=np.asarray("v3-raw-target-record-context"),
                     **self.scale_uV_by_source)

    @classmethod
    def load(cls, config: PreprocessingConfig, path: str | Path) -> "ECGPreprocessor":
        """Load frozen scales; never fit on validation/test records."""
        with np.load(path, allow_pickle=False) as archive:
            if "protocol" not in archive or str(archive["protocol"].item()) != "v3-raw-target-record-context":
                raise PreprocessingError("Incompatible preprocessing artifact; v3 raw-target scales required")
            scales = {key: archive[key].astype(np.float32).copy()
                      for key in archive.files if key != "protocol"}
        for source, scale in scales.items():
            if source not in config.expected_leads or scale.shape != (config.expected_leads[source],):
                raise PreprocessingError(f"Invalid frozen scale shape for {source}")
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise PreprocessingError(f"Invalid frozen scale values for {source}")
        if config.target_transform not in scales or "ecg_machine_i" not in scales:
            raise PreprocessingError("Frozen artifact must include d12 and anchor scales")
        if not np.array_equal(scales["ecg_machine_i"], scales[config.target_transform][:1]):
            raise PreprocessingError("Anchor scale must match d12 I")
        return cls(config=config, scale_uV_by_source=scales)

    @classmethod
    def fit(cls, config: PreprocessingConfig, train_signals: Mapping[str, np.ndarray]) -> "ECGPreprocessor":
        """Fit one robust scale per source and lead from training arrays only.

        Each array must have shape [N, C, T]. Callers must pass only the fixed
        train split; the API deliberately has no validation fitting path.
        """
        if config.target_transform not in train_signals:
            raise PreprocessingError("Every task fit must include training d12 targets")
        if "ecg_machine_i" in train_signals:
            raise PreprocessingError("ecg_machine_i scale is derived from train d12 I; do not fit a separate source array")
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
        scales["ecg_machine_i"] = scales[config.target_transform][:1].copy()
        return cls(config=config, scale_uV_by_source=scales)

    def transform_window(self, raw_window: np.ndarray, source_type: str,
                         record_baseline_uV: np.ndarray | None = None) -> ModelSignal:
        """Scale raw anchor/target; center context using its complete record.

        For d12/anchor, a legacy supplied baseline is intentionally ignored.
        ModelSignal.baseline_uV is the offset actually subtracted (zero there).
        """
        if source_type not in self.config.expected_leads:
            raise PreprocessingError(f"Unknown source type: {source_type}")
        raw = np.asarray(raw_window)
        expected = self.config.expected_leads[source_type]
        if raw.ndim != 2 or raw.shape[0] != expected:
            raise PreprocessingError(f"{source_type} window must have shape [{expected}, T]")
        if not np.isfinite(raw).all():
            raise PreprocessingError("Cannot transform non-finite ECG values")
        if source_type in {"ecg_machine_i", self.config.target_transform}:
            baseline = np.zeros(expected, dtype=np.float32)
        elif record_baseline_uV is None:
            raise PreprocessingError("Context windows require a complete-record baseline; do not center per window")
        else:
            baseline = np.asarray(record_baseline_uV, dtype=np.float32)
        if baseline.shape != (expected,) or not np.isfinite(baseline).all():
            raise PreprocessingError(f"{source_type} record_baseline_uV must have shape [{expected}]")
        if source_type not in self.scale_uV_by_source:
            raise PreprocessingError(f"No frozen train scale for source type: {source_type}")
        scale = self.scale_uV_by_source[source_type]
        transformed = (raw.astype(np.float32, copy=False) - baseline[:, None]) / scale[:, None]
        return ModelSignal(model_signal=transformed, baseline_uV=baseline, scale_uV=scale.copy(), source_type=source_type)

    def transform_observed_record(self, raw_record: np.ndarray, source_type: str) -> ModelSignal:
        """Transform a complete visible input before any windowing.

        This is the test-time counterpart of the Dataset baseline index.  It is
        valid for organizer-provided context/anchor signals, but must never be
        called on hidden target leads at inference.
        """
        raw = np.asarray(raw_record)
        expected = self.config.expected_leads.get(source_type)
        if expected is None or raw.ndim != 2 or raw.shape[0] != expected or raw.shape[1] == 0:
            raise PreprocessingError(f"{source_type} record must have shape [{expected}, T]")
        baseline = (np.zeros(expected, dtype=np.float32) if source_type == "ecg_machine_i"
                    else np.median(raw, axis=1).astype(np.float32))
        return self.transform_window(raw, source_type, baseline)

    def transform_d12_target(self, raw_d12: np.ndarray, record_baseline_uV: np.ndarray | None = None) -> ModelSignal:
        """Scale raw d12 without subtracting target medians, in P0 and P1."""
        return self.transform_window(raw_d12, self.config.target_transform, record_baseline_uV)

    def model_view_to_uV(self, model_window: np.ndarray, source_type: str) -> np.ndarray:
        """Invert scale: d12/anchor return raw μV; context returns centered μV."""
        model = np.asarray(model_window, dtype=np.float32)
        expected = self.config.expected_leads.get(source_type)
        if expected is None or model.ndim != 2 or model.shape[0] != expected:
            raise PreprocessingError(f"{source_type} model window must have shape [{expected}, T]")
        if source_type not in self.scale_uV_by_source:
            raise PreprocessingError(f"No frozen train scale for source type: {source_type}")
        return model * self.scale_uV_by_source[source_type][:, None]

    def d12_model_view_to_morphology_uV(self, d12_model_window: np.ndarray) -> np.ndarray:
        """Legacy name: v3 returns raw μV. Prefer d12_model_view_to_raw_uV."""
        return self.model_view_to_uV(d12_model_window, self.config.target_transform)

    def d12_model_view_to_raw_uV(self, d12_model_window: np.ndarray) -> np.ndarray:
        """Restore raw target voltage without access to hidden target medians."""
        return self.model_view_to_uV(d12_model_window, self.config.target_transform)

    def compose_raw_d12_prediction(self, d12_model_window: np.ndarray,
                                   predicted_baseline_uV: np.ndarray | None = None) -> np.ndarray:
        """Restore v3 raw voltage; adding a baseline again is a protocol error."""
        if predicted_baseline_uV is not None:
            raise PreprocessingError("v3 predicts raw voltage: do not add any target baseline")
        return self.d12_model_view_to_raw_uV(d12_model_window)

    def transform_batch(self, raw_batch: np.ndarray, source_type: str,
                        record_baseline_uV: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized non-destructive transform for [N, C, T] arrays."""
        raw = np.asarray(raw_batch)
        expected = self.config.expected_leads.get(source_type)
        if expected is None or raw.ndim != 3 or raw.shape[1] != expected:
            raise PreprocessingError(f"{source_type} batch must have shape [N, {expected}, T]")
        if not np.isfinite(raw).all():
            raise PreprocessingError("Cannot transform non-finite ECG values")
        if source_type in {"ecg_machine_i", self.config.target_transform}:
            baseline = np.zeros(raw.shape[:2], dtype=np.float32)
        elif record_baseline_uV is None:
            raise PreprocessingError("Context batches require complete-record baselines")
        else:
            baseline = np.asarray(record_baseline_uV, dtype=np.float32)
        if baseline.shape == (expected,):
            baseline = np.broadcast_to(baseline, raw.shape[:2])
        if baseline.shape != raw.shape[:2] or not np.isfinite(baseline).all():
            raise PreprocessingError(
                f"{source_type} record_baseline_uV must have shape [{expected}] or {raw.shape[:2]}")
        if source_type not in self.scale_uV_by_source:
            raise PreprocessingError(f"No frozen train scale for source type: {source_type}")
        scale = self.scale_uV_by_source[source_type]
        model = (raw.astype(np.float32, copy=False) - baseline[:, :, None]) / scale[None, :, None]
        return model, baseline, np.broadcast_to(scale, baseline.shape).copy()
