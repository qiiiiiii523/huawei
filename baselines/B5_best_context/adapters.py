"""Target-free visible-history interface; only base predictions may be shared by target ID."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from . import PROTOCOL
from .fir import HistoricalFIR, apply_postprocess, historical_fir

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/b5_best_context.json"
MODES = ("none", "projection", "fir25")


def load_settings(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    expected = {"ecg_machine_d6": "historical_fir25", "body_scale_d6": "identity", "watch_ecg": "identity"}
    if (value.get("protocol") != PROTOCOL or value.get("machine_mode") not in MODES
            or value.get("fir_strength") != .25 or value.get("routes") != expected):
        raise ValueError("Unsupported context configuration; FIR25 is fixed and body/watch stay identity")
    return value


@dataclass(frozen=True)
class HistoryRecord:
    """Raw physical uV and validity only; deliberately contains no current target."""

    source: str
    record_id: str
    raw_uV: np.ndarray | None
    valid_mask: np.ndarray | None


def history_from_sample(sample):
    """Reassemble physical history independently of target windows, trimming padding only."""
    context = sample.context
    source = sample.context_source_type
    raw = np.asarray(context.raw_uV)
    mask = np.asarray(context.time_mask, dtype=bool)
    n, channels = len(raw), 1 if source == "watch_ecg" else 6
    if raw.shape != (n, channels, 5000) or mask.shape != raw.shape:
        raise ValueError("History requires canonical raw [W,C,5000] and its mask")
    lengths = np.asarray(context.valid_lengths)
    starts = np.asarray(context.starts)
    if (lengths.shape != (n,) or starts.shape != (n,)
            or not np.array_equal(starts, np.arange(n) * 5000)
            or np.any(lengths < 1) or np.any(lengths > 5000)
            or np.any(lengths[:-1] != 5000)):
        raise ValueError("History has missing/overlapping windows or invalid physical lengths")
    quality = np.asarray(sample.input_quality_mask, dtype=bool)
    if quality.shape != (channels,):
        raise ValueError("Missing canonical history lead quality mask")
    if n == 0:
        return HistoryRecord(source, sample.context_record_id, None, None)
    window_mask = np.asarray(context.window_mask, dtype=bool)
    if window_mask.shape != (n,):
        raise ValueError("History window availability has the wrong shape")
    raw_full = np.concatenate([raw[j, :, :int(lengths[j])] for j in range(n)], axis=1)
    mask_full = np.concatenate([mask[j, :, :int(lengths[j])] & window_mask[j] for j in range(n)], axis=1)
    mask_full &= quality[:, None] & np.isfinite(raw_full)
    # Internal missing spans keep their physical positions; no interpolation or median subtraction.
    return HistoryRecord(source, sample.context_record_id, raw_full, mask_full)


def _visible_digest(history):
    digest = hashlib.sha256()
    for value in (history.raw_uV, history.valid_mask):
        if value is None:
            digest.update(b"none")
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


class ContextRouter:
    """Post-combination adapters. Future body adapters implement the same target-free callable.

    A registered body adapter receives (prediction_copy, observed_i, HistoryRecord|None)
    and returns (output, JSON-serializable status). Nothing is registered for body by default.
    """

    def __init__(self, settings=None, *, machine_mode=None, body_adapter=None):
        self.settings = load_settings() if settings is None else dict(settings)
        self.mode = self.settings["machine_mode"] if machine_mode is None else machine_mode
        if self.mode not in MODES or self.settings.get("fir_strength") != .25:
            raise ValueError("Use none/projection/fir25 with fixed 25% personal FIR")
        self.body_adapter = body_adapter
        self._fir_cache = {}

    def _model(self, history):
        if history is None:
            return historical_fir(None, None)
        digest = _visible_digest(history)
        key = (history.source, history.record_id, digest)
        if key not in self._fir_cache:
            self._fir_cache[key] = historical_fir(history.raw_uV, history.valid_mask)
        return self._fir_cache[key]

    def apply(self, prediction_uV, observed_i_uV, *, task_id, input_type, history=None):
        p = np.asarray(prediction_uV)
        anchor = np.asarray(observed_i_uV)
        if (p.shape != (12, 60000) or p.dtype.kind != "f" or anchor.shape not in ((60000,), (1, 60000))
                or not np.isfinite(p).all() or not np.isfinite(anchor).all()):
            raise ValueError("Expected finite full 120s prediction[12,60000] and observed I in uV")
        if ((task_id == "task1" and input_type != "watch_ecg")
                or (task_id == "task2" and input_type not in ("ecg_machine_d6", "body_scale_d6"))
                or task_id not in ("task1", "task2")):
            raise ValueError("Unsupported task/device routing")
        if history is not None and history.source != input_type:
            raise ValueError("History device and requested route differ")
        status = {"protocol": PROTOCOL, "adapter": "identity", "projection_applied": False,
                  "personal_fir_used": False, "fallback": "none", "history_reason": "not_requested",
                  "holdout_r": None, "strength": 0., "history_record_id": history.record_id if history else ""}
        if input_type == "body_scale_d6" and self.body_adapter is not None:
            output, details = self.body_adapter(p.copy(), anchor, history)
            output = np.asarray(output)
            if output.shape != p.shape or not np.isfinite(output).all():
                raise ValueError("Body adapter violates the full-record output contract")
            if not np.array_equal(output[[0, *range(2, 12)]], p[[0, *range(2, 12)]]):
                raise ValueError("Body II adapter must preserve I and all other ten missing leads")
            return output, {**status, **details}
        if input_type != "ecg_machine_d6" or self.mode == "none":
            return p.copy(), status
        model = (self._model(history) if self.mode == "fir25"
                 else HistoricalFIR(None, 0., False, "projection_only"))
        output, details = apply_postprocess(p, anchor, model, strength=.25)
        # Keep the baseline dtype for deployment and score exactly these saved outputs.
        output = output.astype(p.dtype, copy=False)
        status.update(details)
        status.update(adapter="machine_" + self.mode, projection_applied=True,
                      fallback="none" if details["personal_fir_used"] else "projected_base",
                      strength=.25 if details["personal_fir_used"] else 0., holdout_r=model.holdout_r)
        return output, status
