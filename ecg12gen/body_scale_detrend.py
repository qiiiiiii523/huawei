"""Optional body-context B transform, shared by cache building and visible-input inference."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .contracts import ContractError

A_VARIANT = "A_raw_window"
B_VARIANT = "B_detrend_0p2Hz_then_window"
FILTER_PROTOCOL = "body-context-b-hp0p2-order2-sosfiltfilt-valid-runs-v1"
FILTER_SPEC = {
    "protocol": FILTER_PROTOCOL,
    "sampling_rate_hz": 500,
    "design_order": 2,
    "cutoff_hz": 0.2,
    "filter": "Butterworth highpass SOS forward-backward",
    "padtype": "odd",
    "padlen": "min(7500, valid_run_length-2)",
    "gap_policy": "independent_contiguous_valid_runs_per_lead_no_interpolation",
    "short_run_policy": "retain_runs_of_at_most_2_points_unchanged",
    "median_policy": "post_filter_full_record_valid_points_per_lead",
    "target_anchor_machine_policy": "unchanged",
}


def detrend_body_scale_record(raw_uV, valid_mask=None):
    """Filter the entire visible record before windowing; never filter through a gap.

    A gap-free record is one continuous run per lead. Invalid points remain NaN
    until window_context_record applies neutral fill/masks. Tiny valid runs cannot
    support meaningful high-pass filtering and are retained, not silently deleted.
    Full-record forward-backward filtering is offline, not a streaming interface.
    """
    raw = np.asarray(raw_uV, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 6:
        raise ContractError("Body-scale B requires canonical [6,T] raw uV")
    valid = np.isfinite(raw)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape == (raw.shape[1],):
            supplied = np.broadcast_to(supplied, raw.shape)
        if supplied.shape != raw.shape:
            raise ContractError("Body-scale validity must be [T] or [6,T]")
        valid &= supplied
    sos = butter(2, 0.2, fs=500, btype="highpass", output="sos")
    output = np.full(raw.shape, np.nan, dtype=np.float32)
    runs, short_runs = 0, 0
    for lead in range(6):
        edges = np.diff(np.r_[False, valid[lead], False].astype(np.int8))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for start, stop in zip(starts, stops):
            length = int(stop - start)
            segment = raw[lead, start:stop]
            runs += 1
            if length <= 2:
                filtered = segment
                short_runs += 1
            else:
                filtered = sosfiltfilt(sos, segment, padtype="odd", padlen=min(7500, length - 2))
            output[lead, start:stop] = filtered
    if not np.isfinite(output[valid]).all():
        raise ContractError("Body-scale B produced nonfinite valid samples")
    return output, valid.copy(), {"valid_runs": runs, "short_runs_retained": short_runs,
                                 "invalid_lead_points": int((~valid).sum())}
