"""Framework-neutral independent context windowing and masked batch collation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .contracts import ContractError, WINDOW_SAMPLES

TARGET_RECORD_SAMPLES = 60000
TARGET_RECORD_WINDOWS = 12
CONTEXT_CHANNELS = {"watch_ecg": 1, "ecg_machine_d6": 6, "body_scale_d6": 6}


@dataclass(frozen=True)
class ContextWindows:
    raw_uV: np.ndarray                    # [W,C,5000], missing/padding filled with record median
    time_mask: np.ndarray                 # [W,C,5000]
    window_mask: np.ndarray               # [W], at least one valid sample
    valid_lengths: np.ndarray             # [W], physical length before tail padding
    starts: np.ndarray                    # [W], context positions, not target positions
    baseline_uV: np.ndarray               # [C], valid original samples only


def window_context_record(raw_uV: np.ndarray | None, source_type: str,
                          valid_mask: np.ndarray | None = None) -> ContextWindows:
    """Keep every context sample and the final partial window; never use target length."""
    channels = CONTEXT_CHANNELS.get(source_type)
    if channels is None:
        raise ContractError("Unknown context source")
    if raw_uV is None:
        raw = np.empty((channels, 0), dtype=np.float32)
    else:
        raw = np.asarray(raw_uV, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != channels:
        raise ContractError(f"Context must be [{channels},T]")
    if valid_mask is None:
        valid = np.isfinite(raw)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape == (raw.shape[1],):
            valid = np.broadcast_to(valid, raw.shape)
        if valid.shape != raw.shape:
            raise ContractError("Context validity mask must be [T] or [C,T]")
        valid = valid & np.isfinite(raw)
    baseline = np.asarray([np.median(raw[c, valid[c]]) if valid[c].any() else 0
                           for c in range(channels)], dtype=np.float32)
    count = (raw.shape[1] + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
    windows = np.broadcast_to(baseline[None, :, None], (count, channels, WINDOW_SAMPLES)).copy()
    masks = np.zeros(windows.shape, dtype=bool)
    starts = np.arange(count, dtype=np.int64) * WINDOW_SAMPLES
    lengths = np.minimum(WINDOW_SAMPLES, raw.shape[1] - starts)
    for i, (start, length) in enumerate(zip(starts, lengths)):
        mask = valid[:, start:start + length]
        windows[i, :, :length] = np.where(mask, raw[:, start:start + length], baseline[:, None])
        masks[i, :, :length] = mask
    return ContextWindows(windows, masks, masks.any(axis=(1, 2)), lengths, starts, baseline)


def window_target_record(raw_uV: np.ndarray, start_sample: int = 0) -> np.ndarray:
    """Select exactly 120 seconds of real target data; never pad hidden target values."""
    raw = np.asarray(raw_uV, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != 12 or start_sample < 0:
        raise ContractError("Target must be [12,T] with a nonnegative interval start")
    if raw.shape[1] < start_sample + TARGET_RECORD_SAMPLES:
        raise ContractError("Target has fewer than 120 seconds; no context truncation or target padding is allowed")
    selected = raw[:, start_sample:start_sample + TARGET_RECORD_SAMPLES]
    if not np.isfinite(selected).all():
        raise ContractError("Target interval contains nonfinite samples")
    return selected.reshape(12, TARGET_RECORD_WINDOWS, WINDOW_SAMPLES).transpose(1, 0, 2).copy()


def transform_context_windows(context: ContextWindows, source_type: str, preprocessor: Any) -> np.ndarray:
    """Center using original valid samples, scale, and enforce zero on all invalid positions."""
    if not len(context.raw_uV):
        return np.empty_like(context.raw_uV)
    transformed, _, _ = preprocessor.transform_batch(context.raw_uV, source_type, context.baseline_uV)
    return np.where(context.time_mask & context.window_mask[:, None, None], transformed, 0).astype(np.float32)


def prepare_record_context_inference(anchor_uV: np.ndarray, context_uV: np.ndarray | None,
                                     source_type: str, preprocessor: Any,
                                     context_valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    """Target-free 120-second inference input; context length is entirely independent."""
    anchor = np.asarray(anchor_uV, dtype=np.float32)
    if anchor.ndim != 2 or anchor.shape[0] != 1 or anchor.shape[1] < 60000 or not np.isfinite(anchor[:, :60000]).all():
        raise ContractError("A real finite 120-second synchronous I is required")
    context = window_context_record(context_uV, source_type, context_valid_mask)
    signal = (transform_context_windows(context, source_type, preprocessor)
              if context.window_mask.any() else np.zeros_like(context.raw_uV))
    counts = context.time_mask.sum(axis=(1, 2)).astype(np.float32)
    return {"anchor_i": preprocessor.transform_observed_record(anchor[:, :60000], "ecg_machine_i").model_signal.reshape(1, 12, 5000).transpose(1, 0, 2),
            "context": signal, "context_time_mask": context.time_mask,
            "context_window_mask": context.window_mask, "context_valid_lengths": context.valid_lengths,
            "context_window_weights": counts / max(float(counts.sum()), 1),
            "context_available": bool(context.window_mask.any()), "context_source_type": source_type}


@dataclass(frozen=True)
class RecordContextSample:
    """One target window associated with its entire independent context record."""
    context: ContextWindows
    context_source_type: str
    anchor_i_ecg: np.ndarray
    Y_12lead: np.ndarray
    target_quality_mask: np.ndarray
    input_quality_mask: np.ndarray
    task_id: str
    split: str
    subject_id: str
    pair_id: str
    target_record_id: str
    context_record_id: str
    window_id: str
    meta: dict[str, Any]

    @property
    def context_ecg(self) -> np.ndarray:
        return self.context.raw_uV

    @property
    def context_time_mask(self) -> np.ndarray:
        return self.context.time_mask

    @property
    def context_window_mask(self) -> np.ndarray:
        return self.context.window_mask

    @property
    def context_record_baseline_uV(self) -> np.ndarray:
        return self.context.baseline_uV

    def validate(self) -> None:
        channels = CONTEXT_CHANNELS.get(self.context_source_type)
        if self.task_id not in {"task1", "task2"} or self.split not in {"train", "validation"}:
            raise ContractError("Invalid record-context task/split")
        if channels is None or (self.task_id == "task1") != (self.context_source_type == "watch_ecg"):
            raise ContractError("Context source disagrees with task")
        if self.anchor_i_ecg.shape != (1, 5000) or self.Y_12lead.shape != (12, 5000):
            raise ContractError("Invalid anchor/target window shape")
        if not np.array_equal(self.anchor_i_ecg, self.Y_12lead[:1]):
            raise ContractError("Anchor must be the same-record/window target I")
        shape = (len(self.context.raw_uV), channels, 5000)
        if self.context.raw_uV.shape != shape or self.context.time_mask.shape != shape:
            raise ContractError("Context must be an independent [W,C,5000] collection")
        if self.context.window_mask.shape != (shape[0],) or self.context.valid_lengths.shape != (shape[0],):
            raise ContractError("Context window masks/lengths disagree")
        if self.context.starts.shape != (shape[0],) or self.context.baseline_uV.shape != (channels,):
            raise ContractError("Invalid context index or baseline")
        if np.any(self.context.valid_lengths < 1) or np.any(self.context.valid_lengths > 5000):
            raise ContractError("Invalid context physical window lengths")
        if self.target_quality_mask.shape != (12,) or self.input_quality_mask.shape != (channels,):
            raise ContractError("Invalid quality mask shape")
        if self.meta.get("expected_window_count") != 12 or self.meta.get("context_target_sync") is not False:
            raise ContractError("Expected 12 target windows and independent cross-time context")
        if not np.isfinite(self.context.raw_uV).all() or not np.isfinite(self.Y_12lead).all():
            raise ContractError("Nonfinite sample")


def collate_record_context(samples: list[RecordContextSample], preprocessor: Any,
                           context_dropout: float = 0, rng: np.random.Generator | None = None) -> dict[str, Any]:
    """NumPy batch adapter. Model branches convert tensors and perform masked encoding/pooling.

    time_mask excludes gaps, tail padding, and bad leads. window_mask excludes
    empty/disabled/padded windows. weights normalize valid sample counts.
    """
    if not samples or not 0 <= context_dropout <= 1:
        raise ContractError("Need a nonempty batch and dropout in [0,1]")
    if context_dropout and (rng is None or any(s.split != "train" for s in samples)):
        raise ContractError("Context dropout is train-only and requires an explicit seeded generator")
    for sample in samples:
        sample.validate()
    windows = max(1, max(len(s.context.raw_uV) for s in samples))
    channels = max(len(s.input_quality_mask) for s in samples)
    shape = (len(samples), windows, channels, 5000)
    signals = np.zeros(shape, dtype=np.float32)
    masks = np.zeros(shape, dtype=bool)
    window_masks = np.zeros(shape[:2], dtype=bool)
    lengths = np.zeros(shape[:2], dtype=np.int64)
    for i, sample in enumerate(samples):
        n, c = len(sample.context.raw_uV), len(sample.input_quality_mask)
        lengths[i, :n] = sample.context.valid_lengths
        if not n or (context_dropout and rng.random() < context_dropout):
            continue
        valid = sample.context.time_mask & sample.input_quality_mask[None, :, None]
        valid &= sample.context.window_mask[:, None, None]
        if not valid.any():
            continue  # anchor-only: no context scale required when none is usable
        transformed = transform_context_windows(sample.context, sample.context_source_type, preprocessor)
        masks[i, :n, :c] = valid
        signals[i, :n, :c] = np.where(valid, transformed, 0)
        window_masks[i, :n] = valid.any(axis=(1, 2))
    counts = masks.sum(axis=(2, 3)).astype(np.float32)
    weights = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return {
        "context": signals, "context_time_mask": masks, "context_window_mask": window_masks,
        "context_valid_lengths": lengths, "context_valid_counts": counts, "context_window_weights": weights,
        "context_available": window_masks.any(axis=1),
        "context_source_type": [s.context_source_type for s in samples],
        "anchor_i": np.stack([preprocessor.transform_window(s.anchor_i_ecg, "ecg_machine_i").model_signal for s in samples]),
        "target": np.stack([preprocessor.transform_d12_target(s.Y_12lead).model_signal for s in samples]),
        "target_uV": np.stack([s.Y_12lead for s in samples]),
        "target_quality_mask": np.stack([s.target_quality_mask for s in samples]),
        "evaluation_metadata": [{**s.meta, "pair_id": s.pair_id, "target_record_id": s.target_record_id,
                                  "subject_id": s.subject_id, "input_type": s.context_source_type} for s in samples],
    }
