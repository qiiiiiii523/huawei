"""Versioned visible-history processing, shared by body training and inference."""
from __future__ import annotations

import numpy as np

from .adapters import HistoryRecord

A_POLICY = {"variant": "A_raw_window"}
B_POLICY = {"variant": "B_detrend_0p2Hz_then_window", "filter": {
    "protocol": "body-context-b-hp0p2-order2-sosfiltfilt-valid-runs-v1",
    "sampling_rate_hz": 500, "design_order": 2, "cutoff_hz": .2,
    "filter": "Butterworth highpass SOS forward-backward", "padtype": "odd",
    "padlen": "min(7500, valid_run_length-2)",
    "gap_policy": "independent_contiguous_valid_runs_per_lead_no_interpolation",
    "short_run_policy": "retain_runs_of_at_most_2_points_unchanged",
    "median_policy": "post_filter_full_record_valid_points_per_lead",
    "target_anchor_machine_policy": "unchanged"}}


def history_policy(metadata):
    value = metadata.get("history_preprocessing", A_POLICY)
    if value != A_POLICY and value != B_POLICY:
        raise ValueError("Unsupported body history preprocessing protocol")
    return value


def process_history(history, policy):
    """Accept RAW visible history only. B is applied once to the complete record."""
    history_policy({"history_preprocessing": policy})
    if policy == A_POLICY or history is None or history.raw_uV is None:
        return history
    if history.source != "body_scale_d6":
        raise ValueError("Body B processing cannot be applied to machine/watch history")
    from ecg12gen.body_scale_detrend import FILTER_SPEC, detrend_body_scale_record
    if FILTER_SPEC != policy["filter"]:
        raise ValueError("Body B filter implementation changed")
    filtered, valid, _ = detrend_body_scale_record(history.raw_uV, history.valid_mask)
    # Match record-context invalid fill; the mask still excludes those samples.
    for lead in range(6):
        points = filtered[lead, valid[lead]]
        filtered[lead, ~valid[lead]] = np.median(points) if points.size else 0.
    return HistoryRecord(history.source, history.record_id, filtered, valid)
