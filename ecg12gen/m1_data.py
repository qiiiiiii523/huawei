"""M1-only adapters over the frozen main Dataset and Preprocessor contracts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import D6_LEADS, D12_LEADS, SupervisionMode
from .dataset import ECGDataConfig, JointAnchorDataset
from .d12_pretrain import StrictD12PretrainDataset
from .preprocessing import ECGPreprocessor, PreprocessingConfig


def _source_for_task(task_id: str, source_type: str) -> str:
    if task_id == "task1":
        if source_type != "watch_ecg":
            raise ValueError("task1 context must be watch_ecg")
        return "watch_ecg"
    if source_type not in {"ecg_machine_d6", "body_scale_d6"}:
        raise ValueError("task2 context must be exactly one d6 source")
    return source_type


def transform_context_window(preprocessor: ECGPreprocessor, raw_context: np.ndarray,
                             source_type: str, lead_mask: np.ndarray | None = None) -> np.ndarray:
    """Apply the shared transform, also supporting the main 5-of-6 diagnostic."""
    raw = np.asarray(raw_context, dtype=np.float32)
    expected = preprocessor.config.expected_leads[source_type]
    if raw.shape[0] == expected:
        return preprocessor.transform_window(raw, source_type).model_signal.copy()
    if source_type not in {"ecg_machine_d6", "body_scale_d6"} or lead_mask is None:
        raise ValueError(f"{source_type} context has invalid shape {raw.shape}")
    indices = np.flatnonzero(np.asarray(lead_mask, dtype=bool))
    if len(indices) != raw.shape[0] or len(indices) != 5:
        raise ValueError("reduced d6 context must carry a five-lead canonical mask")
    baseline = np.median(raw, axis=1).astype(np.float32)
    scale = preprocessor.scale_uV_by_source[source_type][indices]
    model = (raw - baseline[:, None]) / scale[:, None]
    return np.clip(model, -preprocessor.config.clip_model_signal, preprocessor.config.clip_model_signal)


def fit_m1_preprocessor(config: ECGDataConfig | str | Path, task_id: str,
                        body_scale_variant: str = "A_raw_window",
                        context_channel_indices: tuple[int, ...] | None = None,
                        context_source_type: str | None = None) -> ECGPreprocessor:
    """Fit shared scales from train-only strict d12 and joint context arrays."""
    data_config = config if isinstance(config, ECGDataConfig) else ECGDataConfig.from_yaml(config)
    preprocessing = PreprocessingConfig.from_yaml(data_config.path("preprocessing_config"))
    strict = StrictD12PretrainDataset(data_config, SupervisionMode.D12_I_PRETRAIN.value)
    joint = JointAnchorDataset(data_config, task_id, "train", body_scale_variant, context_channel_indices)
    train_signals: dict[str, list[np.ndarray]] = {"d12": []}
    for index in range(len(strict)):
        train_signals["d12"].append(np.asarray(strict[index].Y_12lead, dtype=np.float32))
    for index in range(len(joint)):
        sample = joint[index]
        source = _source_for_task(task_id, sample.context_source_type)
        if context_source_type is not None and source != context_source_type:
            continue
        # The public dataset may expose a five-lead ablation, while the frozen
        # source scale is still fitted on canonical six-lead training windows.
        raw_context = sample.context_ecg
        if task_id == "task2" and raw_context.shape[0] != 6:
            array_index = joint._indices[index]
            if source == "body_scale_d6" and body_scale_variant == "B_detrend_0p2Hz_then_window":
                raw_context = joint._body_b_inputs[int(joint._body_b_rows[array_index]["local_array_index"])]
            else:
                raw_context = joint._inputs[array_index]
        train_signals.setdefault(source, []).append(np.asarray(raw_context, dtype=np.float32))
    arrays = {key: np.stack(value, axis=0) for key, value in train_signals.items() if value}
    return ECGPreprocessor.fit(preprocessing, arrays)


class M1StrictDataset(Dataset[dict[str, Any]]):
    """Train-only strict d12-I -> d12 samples in frozen model space."""

    def __init__(self, config: ECGDataConfig | str | Path, preprocessor: ECGPreprocessor) -> None:
        self.source = StrictD12PretrainDataset(config, SupervisionMode.D12_I_PRETRAIN.value)
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[index]
        target_raw = np.asarray(sample.Y_12lead, dtype=np.float32)
        target = self.preprocessor.transform_d12_target(target_raw)
        anchor = self.preprocessor.transform_window(target_raw[:1], "ecg_machine_i")
        return {
            "anchor_i": torch.from_numpy(anchor.model_signal.copy()),
            "target": torch.from_numpy(target.model_signal.copy()),
            "anchor_raw": torch.from_numpy(target_raw[:1].copy()),
            "target_raw": torch.from_numpy(target_raw.copy()),
            "anchor_lead_mask": torch.tensor([True] + [False] * 11),
            "meta": sample.meta,
        }


class M1JointDataset(Dataset[dict[str, Any]]):
    """Cross-time context plus same-window machine-I anchor for P1."""

    def __init__(self, config: ECGDataConfig | str | Path, task_id: str, split: str,
                 preprocessor: ECGPreprocessor, body_scale_variant: str = "A_raw_window",
                 context_channel_indices: tuple[int, ...] | None = None,
                 context_source_type: str | None = None) -> None:
        self.main = JointAnchorDataset(config, task_id, split, body_scale_variant, context_channel_indices)
        self.preprocessor = preprocessor
        self.task_id = task_id
        self.context_source_type = context_source_type
        self._indices = [index for index in range(len(self.main))
                         if context_source_type is None or self.main[index].context_source_type == context_source_type]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.main[self._indices[index]]
        context_source = _source_for_task(self.task_id, sample.context_source_type)
        context = transform_context_window(self.preprocessor, sample.context_ecg, context_source,
                                            sample.context_lead_mask if self.task_id == "task2" else None)
        anchor = self.preprocessor.transform_window(sample.anchor_i_ecg, "ecg_machine_i")
        target = self.preprocessor.transform_d12_target(sample.Y_12lead)
        return {
            "anchor_i": torch.from_numpy(anchor.model_signal.copy()),
            "context": torch.from_numpy(context.model_signal.copy()),
            "context_source_type": context_source,
            "context_lead_mask": torch.from_numpy(sample.context_lead_mask.copy()),
            "target": torch.from_numpy(target.model_signal.copy()),
            "anchor_raw": torch.from_numpy(sample.anchor_i_ecg.copy()),
            "target_raw": torch.from_numpy(sample.Y_12lead.copy()),
            "subject_id": sample.subject_id,
            "meta": sample.meta,
        }


def collate_m1(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty M1 batch")
    tensor_keys = ("anchor_i", "target", "anchor_raw", "target_raw")
    output: dict[str, Any] = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    if "context" in batch[0]:
        output["context"] = torch.stack([item["context"] for item in batch])
        output["context_lead_mask"] = torch.stack([item["context_lead_mask"] for item in batch])
        output["context_source_type"] = [item["context_source_type"] for item in batch]
        output["subject_id"] = [item["subject_id"] for item in batch]
    output["meta"] = [item["meta"] for item in batch]
    return output
