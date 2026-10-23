"""Original full-record fast extraction and a typed future slow-output interface."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import COMPONENT_PROTOCOL, MISSING_LEADS, PROTOCOL


def _wave(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} requires finite {shape} data in uV at 500Hz")
    return array


def full_record_slow(wave):
    """Identical float64 MA501/edge algorithm to the original B5-best combiner."""
    value = _wave(wave, (11, 60000), "FM missing leads")
    padded = np.pad(value, ((0, 0), (250, 250)), mode="edge")
    sums = np.concatenate((np.zeros((11, 1)), np.cumsum(padded, axis=1)), axis=1)
    return (sums[:, 501:] - sums[:, :-501]) / 501


def extract_fast(full_fm_uV):
    """Keep float64 subtraction until final composition, matching legacy numerics."""
    full = _wave(full_fm_uV, (12, 60000), "Full FM")
    return full[1:] - full_record_slow(full[1:])


@dataclass(frozen=True)
class WaveComponent:
    """Sample-aligned missing11 component, independent of future model architecture."""

    record_key: str
    values_uV: np.ndarray
    kind: str
    protocol: str
    sampling_rate_hz: int = 500
    start_sample: int = 0
    lead_order: tuple[str, ...] = MISSING_LEADS

    def __post_init__(self):
        if not isinstance(self.record_key, str) or not self.record_key:
            raise ValueError("A stable nonempty record key is required")
        expected = {"fast": PROTOCOL, "slow": COMPONENT_PROTOCOL}
        if self.kind not in expected or self.protocol != expected[self.kind]:
            raise ValueError("Unknown component kind/protocol")
        if self.sampling_rate_hz != 500 or self.start_sample != 0 or tuple(self.lead_order) != MISSING_LEADS:
            raise ValueError("Components require the same 500Hz full-record missing11 coordinates")
        value = _wave(self.values_uV, (11, 60000), self.kind).copy()
        value.setflags(write=False)
        object.__setattr__(self, "values_uV", value)
        object.__setattr__(self, "lead_order", MISSING_LEADS)


def compose(anchor_uV, fast: WaveComponent, slow: WaveComponent | None = None):
    """Return observed I + fast [+ explicit additive slow in uV]. No learned module."""
    anchor = _wave(anchor_uV, (1, 60000), "Observed I")
    if fast.kind != "fast":
        raise ValueError("Expected a fast component")
    missing = fast.values_uV
    if slow is not None:
        if slow.kind != "slow" or slow.record_key != fast.record_key:
            raise ValueError("Slow output must refer to this exact record")
        missing = missing + slow.values_uV
    # coefficient=1: a future slow model must provide its final physical additive output.
    return np.concatenate((anchor, missing), axis=0).astype(np.float32)
