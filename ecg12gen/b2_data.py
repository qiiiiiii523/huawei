"""B2 data adapters built on the frozen D0/D1 datasets and preprocessing."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .body_scale import BodyScaleVariantDataset
from .canonical_adapter import canonicalize_input_ecg
from .contracts import ECGSample, SupervisionMode, canonical_lead_mask
from .dataset import ECGDataConfig, UnifiedECGDataset
from .d12_pretrain import StrictD12PretrainDataset
from .preprocessing import ECGPreprocessor, PreprocessingConfig


def _source_type(sample: ECGSample) -> str:
    if sample.X_ecg.shape[0] == 1:
        return "watch_ecg"
    device_type = str(sample.meta.get("device_type", ""))
    if device_type == "body_scale_d6":
        return "body_scale_d6"
    return "ecg_machine_d6"


def _stack_or_raise(values: list[np.ndarray], name: str) -> np.ndarray:
    if not values:
        raise ValueError(f"No train samples available for {name}")
    return np.stack(values).astype(np.float32, copy=False)


def _task_train_target_array(config: ECGDataConfig, task_id: str) -> np.ndarray:
    return np.load(config.path(f"{task_id}_output") / f"{task_id}_train_target.npy", mmap_mode="r")


def fit_b2_preprocessor(
    config_path: str | Path,
    task_id: str,
    task2_variant: str = "A_raw_window",
) -> ECGPreprocessor:
    """Fit all source scales from train-only arrays for one B2 experiment."""
    config = ECGDataConfig.from_yaml(config_path)
    preprocessing_config = PreprocessingConfig.from_yaml(config.repository_root / "configs" / "preprocessing.yaml")
    train_samples = list(UnifiedECGDataset(config, task_id, "train"))
    train_signals: dict[str, np.ndarray] = {
        "d12": np.asarray(_task_train_target_array(config, task_id), dtype=np.float32),
    }
    if task_id == "task1":
        train_signals["watch_ecg"] = _stack_or_raise([sample.X_ecg for sample in train_samples], "watch_ecg")
    else:
        machine = [sample.X_ecg for sample in train_samples if sample.meta.get("device_type") == "ecg_machine_d6"]
        train_signals["ecg_machine_d6"] = _stack_or_raise(machine, "ecg_machine_d6")
        if task2_variant == "B_detrend_0p2Hz_then_window":
            body_dataset = BodyScaleVariantDataset(config, "train", task2_variant)
        else:
            body_dataset = BodyScaleVariantDataset(config, "train", "A_raw_window")
        train_signals["body_scale_d6"] = _stack_or_raise([sample.X_ecg for sample in body_dataset], "body_scale_d6")
    return ECGPreprocessor.fit(preprocessing_config, train_signals)


def _canonical_observed_d12(
    input_model: np.ndarray,
    input_scale_uV: np.ndarray,
    d12_scale_uV: np.ndarray,
    lead_mask: np.ndarray,
) -> np.ndarray:
    """Map centered source input to the frozen canonical d12 model scale."""
    canonical, inferred_mask = canonicalize_input_ecg(input_model)
    if not np.array_equal(inferred_mask, lead_mask):
        raise ValueError("canonical input mask disagrees with ECGSample lead_mask")
    output = np.zeros_like(canonical, dtype=np.float32)
    channels = int(lead_mask.sum())
    ratio = np.asarray(input_scale_uV, dtype=np.float32) / np.asarray(d12_scale_uV[:channels], dtype=np.float32)
    output[:, :] = canonical.astype(np.float32, copy=False)
    output[:channels] *= ratio[:, None]
    output[~lead_mask] = 0.0
    return output


@dataclass(frozen=True)
class B2Item:
    input_model: torch.Tensor
    target_model: torch.Tensor
    observed_d12_model: torch.Tensor
    lead_mask: torch.Tensor
    missing_mask: torch.Tensor
    raw_target_uV: torch.Tensor
    meta: dict[str, Any]


class B2PreparedDataset(Dataset[B2Item]):
    """Prepare one supervision route without changing any source arrays."""

    def __init__(self, samples: Sequence[ECGSample], preprocessor: ECGPreprocessor, mode: str) -> None:
        if mode not in {"strict", "strict_eval", "weak", "pseudo"}:
            raise ValueError("mode must be strict, strict_eval, weak, or pseudo")
        self.samples = list(samples)
        self.preprocessor = preprocessor
        self.mode = mode
        for sample in self.samples:
            sample.validate()
            if mode == "strict" and sample.split != "train":
                raise ValueError("strict B2 data is train-only")
            if mode == "pseudo":
                if sample.split != "train" or sample.meta.get("accepted") is not True:
                    raise ValueError("pseudo B2 data must be accepted train samples")
                if sample.alignment_mode != "pseudo_rpeak_monotonic_warp":
                    raise ValueError("pseudo B2 data has an invalid alignment mode")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> B2Item:
        sample = self.samples[index]
        lead_mask = np.asarray(sample.lead_mask, dtype=bool)
        missing_mask = np.asarray(sample.missing_mask, dtype=bool)
        if not np.array_equal(missing_mask, ~lead_mask):
            raise ValueError("lead_mask and missing_mask are not complementary")

        target_model = self.preprocessor.transform_d12_target(sample.Y_12lead).model_signal
        if self.mode in {"strict", "strict_eval"}:
            channels = int(lead_mask.sum())
            input_model = target_model[:channels].copy()
            observed_model = np.zeros_like(target_model)
            observed_model[:channels] = target_model[:channels]
        else:
            source_type = _source_type(sample)
            input_view = self.preprocessor.transform_window(sample.X_ecg, source_type)
            input_model = input_view.model_signal
            observed_model = _canonical_observed_d12(
                input_model, input_view.scale_uV, self.preprocessor.scale_uV_by_source["d12"], lead_mask
            )

        return B2Item(
            input_model=torch.from_numpy(np.asarray(input_model, dtype=np.float32)),
            target_model=torch.from_numpy(np.asarray(target_model, dtype=np.float32)),
            observed_d12_model=torch.from_numpy(np.asarray(observed_model, dtype=np.float32)),
            lead_mask=torch.from_numpy(lead_mask),
            missing_mask=torch.from_numpy(missing_mask),
            raw_target_uV=torch.from_numpy(np.asarray(sample.Y_12lead, dtype=np.float32).copy()),
            meta=dict(sample.meta),
        )


def b2_collate(items: Sequence[B2Item]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty B2 batch")
    return {
        "input_model": torch.stack([item.input_model for item in items]),
        "target_model": torch.stack([item.target_model for item in items]),
        "observed_d12_model": torch.stack([item.observed_d12_model for item in items]),
        "lead_mask": torch.stack([item.lead_mask for item in items]),
        "missing_mask": torch.stack([item.missing_mask for item in items]),
        "raw_target_uV": torch.stack([item.raw_target_uV for item in items]),
        "meta": [item.meta for item in items],
    }


def _weak_samples(config: ECGDataConfig, task_id: str, split: str, task2_variant: str) -> list[ECGSample]:
    unified = list(UnifiedECGDataset(config, task_id, split))
    if task_id == "task1":
        return unified
    if task2_variant == "A_raw_window":
        allowed = {"ecg_machine_d6", "body_scale_d6"}
        return [sample for sample in unified if sample.meta.get("device_type") in allowed]
    machine = [sample for sample in unified if sample.meta.get("device_type") == "ecg_machine_d6"]
    body = list(BodyScaleVariantDataset(config, split, "B_detrend_0p2Hz_then_window"))
    if any(sample.meta.get("input_processing_variant") == "A_raw_window" for sample in body):
        raise ValueError("task2-B body dataset unexpectedly contains raw body-scale A")
    return machine + body


def build_weak_dataset(
    config_path: str | Path,
    task_id: str,
    split: str,
    preprocessor: ECGPreprocessor,
    task2_variant: str = "A_raw_window",
) -> B2PreparedDataset:
    config = ECGDataConfig.from_yaml(config_path)
    return B2PreparedDataset(_weak_samples(config, task_id, split, task2_variant), preprocessor, "weak")


def build_strict_dataset(
    config_path: str | Path,
    task_id: str,
    preprocessor: ECGPreprocessor,
) -> B2PreparedDataset:
    config = ECGDataConfig.from_yaml(config_path)
    mode = SupervisionMode.D12_I_PRETRAIN.value if task_id == "task1" else SupervisionMode.D12_SIX_PRETRAIN.value
    samples = list(StrictD12PretrainDataset(config, mode))
    if any(sample.split != "train" for sample in samples):
        raise ValueError("strict index unexpectedly contains validation rows")
    return B2PreparedDataset(samples, preprocessor, "strict")


def build_strict_validation_dataset(
    config_path: str | Path,
    task_id: str,
    preprocessor: ECGPreprocessor,
) -> B2PreparedDataset:
    """Build a held-out d12-I/d12 or d12-six/d12 diagnostic dataset.

    This dataset is never passed to an optimizer.  It uses the existing
    subject-level validation target arrays and the same frozen train-fitted
    preprocessor as the training run.
    """
    config = ECGDataConfig.from_yaml(config_path)
    target_path = config.path(f"{task_id}_output") / f"{task_id}_validation_target.npy"
    metadata_path = config.path(f"{task_id}_output") / f"{task_id}_window_metadata.csv"
    targets = np.load(target_path, mmap_mode="r")
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(
            (row for row in csv.DictReader(handle) if row.get("split") == "validation"),
            key=lambda row: int(row["array_index"]),
        )
    if len(rows) != len(targets):
        raise ValueError(f"Strict validation metadata and target rows disagree for {task_id}")
    if [int(row["array_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"Strict validation array_index must be contiguous for {task_id}")

    channels = 1 if task_id == "task1" else 6
    supervision_mode = f"d12_{'i' if channels == 1 else 'six'}_validation"
    samples: list[ECGSample] = []
    for index, row in enumerate(rows):
        target = np.asarray(targets[index], dtype=np.float32)
        mask = canonical_lead_mask(channels)
        samples.append(ECGSample(
            X_ecg=target[:channels], lead_mask=mask, Y_12lead=target, missing_mask=~mask,
            task_id=task_id, ppg=None, acc=None,
            meta={"subject_id": row["subject_id"], "window_id": row["window_id"],
                  "strict_validation": True, "pointwise_loss_allowed": True},
            modality_mask={"ppg": False, "acc": False}, split="validation",
            supervision_mode=supervision_mode, pairing_type="within_d12_sync",
            alignment_mode="same_window", pair_confidence="not_applicable", pair_status="paired",
        ))
    return B2PreparedDataset(samples, preprocessor, "strict_eval")


def build_pseudo_dataset(
    config_path: str | Path,
    preprocessor: ECGPreprocessor,
    variant: str = "C1",
) -> B2PreparedDataset:
    config = ECGDataConfig.from_yaml(config_path)
    if variant not in {"C1", "C2"}:
        raise ValueError("pseudo variant must be C1 or C2")
    directory = config.repository_root.parent / ("task1_rpeak_pseudo_output_c2" if variant == "C2" else "task1_rpeak_pseudo_output")
    inputs = np.load(directory / "task1_rpeak_train_input.npy", mmap_mode="r")
    targets = np.load(directory / "task1_rpeak_train_target.npy", mmap_mode="r")
    with (directory / "task1_rpeak_train_window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if inputs.shape[0] != targets.shape[0] or inputs.shape[0] != len(rows):
        raise ValueError("pseudo arrays and metadata have different lengths")
    samples: list[ECGSample] = []
    for index, row in enumerate(rows):
        accepted = row.get("accepted", "").lower() == "true"
        if not accepted or row.get("split") != "train" or row.get("alignment_mode") != "pseudo_rpeak_monotonic_warp":
            raise ValueError("pseudo dataset contains a non-accepted or non-train row")
        quality = float(row["alignment_quality_score"])
        if not 0.0 <= quality <= 1.0:
            raise ValueError("alignment_quality_score must be in [0, 1]")
        mask = canonical_lead_mask(1)
        samples.append(ECGSample(
            X_ecg=np.asarray(inputs[index], dtype=np.float32), lead_mask=mask,
            Y_12lead=np.asarray(targets[index], dtype=np.float32), missing_mask=~mask,
            task_id="task1", ppg=None, acc=None,
            meta={"subject_id": row["subject_id"], "window_id": row["source_window_id"], "accepted": True,
                  "alignment_quality_score": quality, "pointwise_loss_allowed": True},
            modality_mask={"ppg": False, "acc": False}, split="train",
            supervision_mode="rpeak_pseudo_adaptation", pairing_type="rpeak_pseudo",
            alignment_mode="pseudo_rpeak_monotonic_warp", pair_confidence="not_applicable", pair_status="accepted",
        ))
    return B2PreparedDataset(samples, preprocessor, "pseudo")
