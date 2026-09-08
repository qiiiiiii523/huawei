"""Read-only B2 adapters for strict and multi-context joint-anchor data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import ECGSample, JointAnchorSample, SupervisionMode, canonical_lead_mask
from .dataset import ECGDataConfig, JointAnchorDataset
from .d12_pretrain import StrictD12PretrainDataset
from .preprocessing import ECGPreprocessor, PreprocessingConfig


class DualContextIntersectionError(ValueError):
    """Raised instead of inventing a task2 machine/body/anchor triple."""


def _key(sample: JointAnchorSample) -> tuple[str, str, str, str]:
    return sample.subject_id, sample.split, sample.target_record_id, sample.window_id


@dataclass(frozen=True)
class B2DualContextSample:
    """A verified task2 body + machine context pair for one anchor/target window."""
    machine: JointAnchorSample
    body: JointAnchorSample

    def validate(self) -> None:
        if self.machine.context_source_type != "ecg_machine_d6" or self.body.context_source_type != "body_scale_d6":
            raise DualContextIntersectionError("dual sample needs one machine and one body d6 context")
        if _key(self.machine) != _key(self.body):
            raise DualContextIntersectionError("dual contexts must match subject_id, split, target_record_id, and window_id")
        if not np.array_equal(self.machine.anchor_i_ecg, self.body.anchor_i_ecg) or not np.array_equal(self.machine.Y_12lead, self.body.Y_12lead):
            raise DualContextIntersectionError("dual contexts disagree on same-record/window anchor or target")


def _joint_samples(config_path: str | Path, task_id: str, split: str, body_scale_variant: str,
                   context_channel_indices: tuple[int, ...] | None) -> list[JointAnchorSample]:
    return list(JointAnchorDataset(config_path, task_id, split, body_scale_variant, context_channel_indices))


def task2_dual_intersection(config_path: str | Path, split: str, body_scale_variant: str = "A_raw_window",
                            context_channel_indices: tuple[int, ...] | None = None) -> tuple[list[B2DualContextSample], dict[str, int]]:
    samples = _joint_samples(config_path, "task2", split, body_scale_variant, context_channel_indices)
    grouped: dict[tuple[str, str, str, str], dict[str, JointAnchorSample]] = {}
    for item in samples:
        group = grouped.setdefault(_key(item), {})
        if item.context_source_type in group:
            raise DualContextIntersectionError("ambiguous duplicate context source for one target record/window")
        group[item.context_source_type] = item
    pairs: list[B2DualContextSample] = []
    for group in grouped.values():
        if {"ecg_machine_d6", "body_scale_d6"} <= set(group):
            pair = B2DualContextSample(group["ecg_machine_d6"], group["body_scale_d6"]); pair.validate(); pairs.append(pair)
    counts = {"groups": len(grouped), "machine_rows": sum(x.context_source_type == "ecg_machine_d6" for x in samples),
              "body_rows": sum(x.context_source_type == "body_scale_d6" for x in samples), "both_rows": len(pairs),
              "subjects": len({x.subject_id for x in samples}), "both_subjects": len({x.machine.subject_id for x in pairs})}
    return pairs, counts


def _stack(values: list[np.ndarray], name: str) -> np.ndarray:
    if not values: raise ValueError(f"No train samples for {name}")
    return np.stack(values).astype(np.float32, copy=False)


def fit_b2_preprocessor(config_path: str | Path, task_id: str, body_scale_variant: str = "A_raw_window",
                        context_channel_indices: tuple[int, ...] | None = None,
                        d12_scale_uV: np.ndarray | None = None) -> ECGPreprocessor:
    """Fit train-only source scales, with one canonical strict-train d12 scale.

    The d12 target coordinate system is fitted from the de-duplicated strict
    train index.  P1 may pass the P0 scale explicitly so loading a P0
    checkpoint never changes the target coordinate system.
    """
    root = ECGDataConfig.from_yaml(config_path).repository_root
    config = PreprocessingConfig.from_yaml(root / "configs" / "preprocessing.yaml")
    strict = StrictD12PretrainDataset(config_path, SupervisionMode.D12_I_PRETRAIN.value)
    samples = _joint_samples(config_path, task_id, "train", body_scale_variant, context_channel_indices)
    signals: dict[str, list[np.ndarray]] = {"d12": [sample.Y_12lead for sample in strict]}
    for sample in samples:
        signals["d12"].append(sample.Y_12lead)
        context = sample.context_ecg
        if sample.task_id == "task2" and context.shape[0] < 6:
            full = np.zeros((6, 5000), dtype=np.float32); full[np.flatnonzero(sample.context_lead_mask)] = context; context = full
        signals.setdefault(sample.context_source_type, []).append(context)
    preprocessor = ECGPreprocessor.fit(config, {name: _stack(values, name) for name, values in signals.items()})
    if d12_scale_uV is not None:
        scale = np.asarray(d12_scale_uV, dtype=np.float32)
        if scale.shape != (12,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("d12_scale_uV must be finite, positive, and have shape [12]")
        preprocessor.scale_uV_by_source["d12"] = scale.copy()
        preprocessor.scale_uV_by_source["ecg_machine_i"] = scale[:1].copy()
    return preprocessor


def _canonical_d6(raw: np.ndarray, lead_mask: np.ndarray, source: str, preprocessor: ECGPreprocessor) -> np.ndarray:
    full = np.zeros((6, 5000), dtype=np.float32)
    full[np.flatnonzero(lead_mask)] = raw
    return preprocessor.transform_window(full, source).model_signal


@dataclass(frozen=True)
class B2Item:
    anchor_model: torch.Tensor
    target_model: torch.Tensor
    raw_anchor_i_uV: torch.Tensor
    raw_target_uV: torch.Tensor
    anchor_lead_mask: torch.Tensor
    watch_context_model: torch.Tensor
    watch_available: torch.Tensor
    machine_d6_model: torch.Tensor
    machine_d6_mask: torch.Tensor
    machine_available: torch.Tensor
    body_d6_model: torch.Tensor
    body_d6_mask: torch.Tensor
    body_available: torch.Tensor
    meta: dict[str, Any]


class B2PreparedDataset(Dataset[B2Item]):
    def __init__(self, samples: Sequence[ECGSample | JointAnchorSample | B2DualContextSample], preprocessor: ECGPreprocessor,
                 mode: str, context_view: str = "none", shuffle_seed: int = 42) -> None:
        self.samples, self.preprocessor, self.mode, self.context_view = list(samples), preprocessor, mode, context_view
        if not self.samples or mode not in {"strict_anchor_pretrain", "joint_anchor"}:
            raise ValueError("B2 dataset requires samples and a known mode")
        self._donors: dict[int, int] = {}
        if context_view.startswith("shuffle"):
            rng = np.random.default_rng(shuffle_seed)
            sources: dict[str, list[int]] = {}
            for index, item in enumerate(self.samples):
                sample = item.machine if isinstance(item, B2DualContextSample) else item
                source = sample.context_source_type if isinstance(sample, JointAnchorSample) else "strict"
                sources.setdefault(source, []).append(index)
            for index, item in enumerate(self.samples):
                sample = item.machine if isinstance(item, B2DualContextSample) else item
                subject = sample.subject_id if isinstance(sample, JointAnchorSample) else ""
                source = sample.context_source_type if isinstance(sample, JointAnchorSample) else "strict"
                candidates = [j for j in sources[source] if j != index and isinstance(self.samples[j], JointAnchorSample) and self.samples[j].subject_id != subject]
                if not candidates: raise ValueError("shuffle diagnostic needs a different-subject context in the same split/source")
                self._donors[index] = int(rng.choice(candidates))

    def __len__(self) -> int: return len(self.samples)

    def _joint_context(self, sample: JointAnchorSample, watch: np.ndarray, machine: np.ndarray, body: np.ndarray,
                       watch_available: bool, machine_available: bool, body_available: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, bool]:
        if sample.context_source_type == "watch_ecg":
            return self.preprocessor.transform_window(sample.context_ecg, "watch_ecg").model_signal, machine, body, True, machine_available, body_available
        model = _canonical_d6(sample.context_ecg, sample.context_lead_mask, sample.context_source_type, self.preprocessor)
        if sample.context_source_type == "ecg_machine_d6": return watch, model, body, watch_available, True, body_available
        return watch, machine, model, watch_available, machine_available, True

    def __getitem__(self, index: int) -> B2Item:
        item = self.samples[index]
        watch, machine, body = np.zeros((1, 5000), np.float32), np.zeros((6, 5000), np.float32), np.zeros((6, 5000), np.float32)
        machine_mask, body_mask = np.zeros(6, dtype=bool), np.zeros(6, dtype=bool)
        watch_available = machine_available = body_available = False
        if isinstance(item, ECGSample):
            anchor_raw, target_raw, meta = item.X_ecg, item.Y_12lead, {**item.meta, "stage": "P0_anchor_only", "context_view": "none"}
        elif isinstance(item, B2DualContextSample):
            anchor_raw, target_raw = item.machine.anchor_i_ecg, item.machine.Y_12lead
            watch, machine, body, watch_available, machine_available, body_available = self._joint_context(item.machine, watch, machine, body, False, False, False)
            watch, machine, body, watch_available, machine_available, body_available = self._joint_context(item.body, watch, machine, body, watch_available, machine_available, body_available)
            machine_mask, body_mask = item.machine.context_lead_mask, item.body.context_lead_mask
            meta = {**item.machine.meta, "subject_id": item.machine.subject_id, "split": item.machine.split, "input_type": "both_d6", "stage": "P1_joint_anchor", "context_view": "both", "body_context_subject_id": item.body.subject_id, "machine_context_subject_id": item.machine.subject_id}
        else:
            sample = item
            donor = self.samples[self._donors[index]] if index in self._donors else sample
            assert isinstance(donor, JointAnchorSample)
            anchor_raw, target_raw = sample.anchor_i_ecg, sample.Y_12lead
            watch, machine, body, watch_available, machine_available, body_available = self._joint_context(donor, watch, machine, body, False, False, False)
            if donor.context_source_type == "ecg_machine_d6": machine_mask = donor.context_lead_mask
            if donor.context_source_type == "body_scale_d6": body_mask = donor.context_lead_mask
            meta = {**sample.meta, "subject_id": sample.subject_id, "split": sample.split, "input_type": sample.context_source_type, "stage": "P1_joint_anchor", "context_view": self.context_view,
                    "context_subject_id": donor.subject_id, "context_shuffled": donor.subject_id != sample.subject_id}
        anchor_model = self.preprocessor.transform_window(anchor_raw, "ecg_machine_i").model_signal
        target_model = self.preprocessor.transform_d12_target(target_raw).model_signal
        return B2Item(torch.from_numpy(anchor_model), torch.from_numpy(target_model), torch.from_numpy(anchor_raw.copy()), torch.from_numpy(target_raw.copy()),
            torch.from_numpy(canonical_lead_mask(1)), torch.from_numpy(watch), torch.tensor(watch_available), torch.from_numpy(machine),
            torch.from_numpy(machine_mask), torch.tensor(machine_available), torch.from_numpy(body), torch.from_numpy(body_mask), torch.tensor(body_available), meta)


def b2_collate(items: Sequence[B2Item]) -> dict[str, Any]:
    if not items: raise ValueError("empty B2 batch")
    names = ("anchor_model", "target_model", "raw_anchor_i_uV", "raw_target_uV", "anchor_lead_mask", "watch_context_model", "watch_available", "machine_d6_model", "machine_d6_mask", "machine_available", "body_d6_model", "body_d6_mask", "body_available")
    return {name: torch.stack([getattr(item, name) for item in items]) for name in names} | {"meta": [item.meta for item in items]}


def build_strict_dataset(config_path: str | Path, preprocessor: ECGPreprocessor) -> B2PreparedDataset:
    rows = list(StrictD12PretrainDataset(config_path, SupervisionMode.D12_I_PRETRAIN.value))
    if any(row.split != "train" for row in rows): raise ValueError("strict index contains validation")
    return B2PreparedDataset(rows, preprocessor, "strict_anchor_pretrain")


def build_joint_dataset(config_path: str | Path, task_id: str, split: str, preprocessor: ECGPreprocessor,
                        body_scale_variant: str = "A_raw_window", context_channel_indices: tuple[int, ...] | None = None,
                        context_view: str = "auto", shuffle_seed: int = 42,
                        common_intersection: bool = False) -> B2PreparedDataset:
    rows = _joint_samples(config_path, task_id, split, body_scale_variant, context_channel_indices)
    if task_id == "task1":
        if context_view not in {"auto", "watch", "shuffle_watch"}: raise ValueError("task1 context_view must be watch or shuffle_watch")
    elif context_view == "machine": rows = [x for x in rows if x.context_source_type == "ecg_machine_d6"]
    elif context_view == "body": rows = [x for x in rows if x.context_source_type == "body_scale_d6"]
    elif context_view == "both":
        pairs, counts = task2_dual_intersection(config_path, split, body_scale_variant, context_channel_indices)
        if not pairs: raise DualContextIntersectionError(f"T2-both disabled: no exact machine/body intersection in {split}; {counts}")
        return B2PreparedDataset(pairs, preprocessor, "joint_anchor", "both", shuffle_seed)
    elif context_view not in {"auto", "shuffle"}: raise ValueError("unknown task2 context_view")
    if task_id == "task2" and common_intersection:
        pairs, counts = task2_dual_intersection(config_path, split, body_scale_variant, context_channel_indices)
        if not pairs: raise DualContextIntersectionError(f"common task2 comparison is blocked: no exact intersection in {split}; {counts}")
        keys = {_key(pair.machine) for pair in pairs}; rows = [item for item in rows if _key(item) in keys]
    if not rows: raise ValueError(f"no {task_id} rows for context_view={context_view}")
    return B2PreparedDataset(rows, preprocessor, "joint_anchor", context_view, shuffle_seed)


def dataset_summary(dataset: B2PreparedDataset) -> dict[str, int | str]:
    meta = [dataset[index].meta for index in range(len(dataset))]
    return {"context_view": dataset.context_view, "n_windows": len(dataset), "n_subjects": len({str(x.get("subject_id", "")) for x in meta}),
            "n_machine_available": sum(bool(dataset[index].machine_available) for index in range(len(dataset))),
            "n_body_available": sum(bool(dataset[index].body_available) for index in range(len(dataset)))}
