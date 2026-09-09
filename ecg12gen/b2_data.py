"""B2 adapters for the latest main joint-anchor data contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import ECGSample, JointAnchorSample, SupervisionMode, canonical_lead_mask
from .dataset import ECGDataConfig, JointAnchorDataset
from .d12_pretrain import StrictD12PretrainDataset
from .preprocessing import ECGPreprocessor, PreprocessingConfig


def _stack(values: list[np.ndarray], name: str) -> np.ndarray:
    if not values:
        raise ValueError(f"No train samples available for {name}")
    return np.stack(values).astype(np.float32, copy=False)


def _source_context(sample: JointAnchorSample) -> np.ndarray:
    """Restore a selected task-2 context to canonical d6 before scaling."""
    if sample.task_id != "task2":
        return np.asarray(sample.context_ecg, dtype=np.float32)
    full = np.zeros((6, sample.context_ecg.shape[-1]), dtype=np.float32)
    full[np.flatnonzero(sample.context_lead_mask)] = sample.context_ecg
    return full


def fit_b2_preprocessor(config_path: str | Path, task_id: str,
                        body_scale_variant: str = "A_raw_window",
                        context_channel_indices: tuple[int, ...] | None = None,
                        d12_scale_uV: np.ndarray | None = None) -> ECGPreprocessor:
    """Fit frozen scales using train-only strict targets and context windows."""
    root = ECGDataConfig.from_yaml(config_path).repository_root
    preprocessing = PreprocessingConfig.from_yaml(root / "configs" / "preprocessing.yaml")
    strict = list(StrictD12PretrainDataset(config_path, SupervisionMode.D12_I_PRETRAIN.value))
    joint = list(JointAnchorDataset(config_path, task_id, "train", body_scale_variant, context_channel_indices))
    signals: dict[str, list[np.ndarray]] = {"d12": [sample.Y_12lead for sample in strict]}
    for sample in joint:
        signals.setdefault(sample.context_source_type, []).append(_source_context(sample))
    fitted = ECGPreprocessor.fit(preprocessing, {key: _stack(value, key) for key, value in signals.items()})
    if d12_scale_uV is not None:
        scale = np.asarray(d12_scale_uV, dtype=np.float32)
        if scale.shape != (12,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("d12_scale_uV must be finite, positive, and have shape [12]")
        fitted.scale_uV_by_source["d12"] = scale.copy()
    # Main defines machine-I as a view of the train d12 I scale.
    fitted.scale_uV_by_source["ecg_machine_i"] = fitted.scale_uV_by_source["d12"][:1].copy()
    return fitted


def _transform_anchor(preprocessor: ECGPreprocessor, raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != 1:
        raise ValueError("anchor_i_ecg must have shape [1,5000]")
    return preprocessor.transform_window(raw, "ecg_machine_i").model_signal


@dataclass(frozen=True)
class B2Item:
    anchor_model: torch.Tensor
    target_model: torch.Tensor
    raw_anchor_i_uV: torch.Tensor
    raw_target_uV: torch.Tensor
    anchor_lead_mask: torch.Tensor
    context_model: torch.Tensor
    context_lead_mask: torch.Tensor
    context_source_type: str
    meta: dict[str, Any]


class B2PreparedDataset(Dataset[B2Item]):
    """Torch view over strict P0 or joint-anchor P1 samples."""

    def __init__(self, samples: Sequence[ECGSample | JointAnchorSample],
                 preprocessor: ECGPreprocessor, mode: str) -> None:
        if mode not in {"strict_anchor_pretrain", "joint_anchor"}:
            raise ValueError("mode must be strict_anchor_pretrain or joint_anchor")
        self.samples, self.preprocessor, self.mode = list(samples), preprocessor, mode
        for sample in self.samples:
            sample.validate()
            if mode == "strict_anchor_pretrain":
                if not isinstance(sample, ECGSample) or sample.split != "train":
                    raise ValueError("strict anchor data must be train-only ECGSample rows")
            elif not isinstance(sample, JointAnchorSample):
                raise ValueError("joint-anchor data must contain JointAnchorSample rows")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> B2Item:
        sample = self.samples[index]
        if self.mode == "strict_anchor_pretrain":
            assert isinstance(sample, ECGSample)
            target_raw = np.asarray(sample.Y_12lead, dtype=np.float32)
            context_model = np.zeros((1, target_raw.shape[-1]), dtype=np.float32)
            context_mask = np.zeros((1,), dtype=bool)
            source = "watch_ecg"
            meta = dict(sample.meta)
        else:
            assert isinstance(sample, JointAnchorSample)
            target_raw = np.asarray(sample.Y_12lead, dtype=np.float32)
            context_raw = np.asarray(sample.context_ecg, dtype=np.float32)
            if sample.task_id == "task2":
                full = _source_context(sample)
                context_model_full = self.preprocessor.transform_window(full, sample.context_source_type).model_signal
                context_model = context_model_full[np.flatnonzero(sample.context_lead_mask)]
                context_mask = np.asarray(sample.context_lead_mask, dtype=bool)
            else:
                context_model = self.preprocessor.transform_window(context_raw, sample.context_source_type).model_signal
                context_mask = np.ones((1,), dtype=bool)
            source = sample.context_source_type
            meta = dict(sample.meta)
            meta.update({"subject_id": sample.subject_id, "input_type": sample.context_source_type,
                         "split": sample.split, "window_id": sample.window_id})
        anchor_raw = target_raw[:1].copy()
        target_model = self.preprocessor.transform_d12_target(target_raw).model_signal
        anchor_model = _transform_anchor(self.preprocessor, anchor_raw)
        return B2Item(
            anchor_model=torch.from_numpy(np.asarray(anchor_model, np.float32)),
            target_model=torch.from_numpy(np.asarray(target_model, np.float32)),
            raw_anchor_i_uV=torch.from_numpy(anchor_raw.copy()),
            raw_target_uV=torch.from_numpy(target_raw.copy()),
            anchor_lead_mask=torch.from_numpy(canonical_lead_mask(1)),
            context_model=torch.from_numpy(np.asarray(context_model, np.float32)),
            context_lead_mask=torch.from_numpy(np.asarray(context_mask, bool)),
            context_source_type=source, meta=meta,
        )


def b2_collate(items: Sequence[B2Item]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty B2 batch")
    return {
        "anchor_model": torch.stack([x.anchor_model for x in items]),
        "target_model": torch.stack([x.target_model for x in items]),
        "raw_anchor_i_uV": torch.stack([x.raw_anchor_i_uV for x in items]),
        "raw_target_uV": torch.stack([x.raw_target_uV for x in items]),
        "anchor_lead_mask": torch.stack([x.anchor_lead_mask for x in items]),
        "context_model": torch.stack([x.context_model for x in items]),
        "context_lead_mask": torch.stack([x.context_lead_mask for x in items]),
        "context_source_type": [x.context_source_type for x in items],
        "meta": [x.meta for x in items],
    }


def build_strict_dataset(config_path: str | Path, preprocessor: ECGPreprocessor) -> B2PreparedDataset:
    samples = list(StrictD12PretrainDataset(config_path, SupervisionMode.D12_I_PRETRAIN.value))
    return B2PreparedDataset(samples, preprocessor, "strict_anchor_pretrain")


def build_joint_dataset(config_path: str | Path, task_id: str, split: str,
                        preprocessor: ECGPreprocessor,
                        body_scale_variant: str = "A_raw_window",
                        context_channel_indices: tuple[int, ...] | None = None,
                        context_view: str = "auto") -> B2PreparedDataset:
    if context_view not in {"auto", "machine", "body", "watch", "none"}:
        raise ValueError("unknown context_view")
    dataset = JointAnchorDataset(config_path, task_id, split, body_scale_variant, context_channel_indices)
    samples = list(dataset)
    if context_view in {"machine", "body", "watch"}:
        expected = {"machine": "ecg_machine_d6", "body": "body_scale_d6", "watch": "watch_ecg"}[context_view]
        samples = [x for x in samples if x.context_source_type == expected]
    if context_view == "none":
        samples = []
    return B2PreparedDataset(samples, preprocessor, "joint_anchor")


def build_joint_anchor_dataset(*args: Any, **kwargs: Any) -> B2PreparedDataset:
    return build_joint_dataset(*args, **kwargs)


def dataset_summary(dataset: B2PreparedDataset) -> dict[str, Any]:
    return {"n_samples": len(dataset), "mode": dataset.mode,
            "context_sources": sorted({x.context_source_type for x in dataset.samples if isinstance(x, JointAnchorSample)})}
