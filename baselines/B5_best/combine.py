"""Full-record 501-point replacement with an explicit predicted-slow coefficient."""
from __future__ import annotations

import numpy as np


def slow_curve(wave):
    value = np.asarray(wave, dtype=np.float64)
    if value.ndim != 2 or value.shape[-1] != 60000 or not np.isfinite(value).all():
        raise ValueError("Moving average requires finite [C,60000] full-record data")
    padded = np.pad(value, ((0, 0), (250, 250)), mode="edge")
    sums = np.concatenate((np.zeros((len(value), 1)), np.cumsum(padded, axis=1)), axis=1)
    return (sums[:, 501:] - sums[:, :-501]) / 501


def upsample_native(native):
    value = np.asarray(native)
    if value.shape != (11, 6000) or not np.isfinite(value).all():
        raise ValueError("Predicted slow curve must be finite [11,6000] at 50Hz")
    centers = np.arange(6000) * 10 + 4.5
    return np.stack([np.interp(np.arange(60000), centers, row) for row in value])


def validate_slow_beta(beta):
    if isinstance(beta, (bool, np.bool_)) or not isinstance(beta, (int, float, np.number)):
        raise ValueError("slow_beta must be a finite nonnegative number")
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("slow_beta must be a finite nonnegative number")
    return beta


def combine(fm_uV, predicted_slow_uV, slow_beta=1.0):
    # Keep the historical utility default; deployed inference passes the bundle's beta explicitly.
    slow_beta = validate_slow_beta(slow_beta)
    baseline, learned = np.asarray(fm_uV), np.asarray(predicted_slow_uV)
    if baseline.shape != (12, 60000) or learned.shape != (11, 60000):
        raise ValueError("Requires FM [12,60000] and predicted slow [11,60000]")
    if not np.isfinite(baseline).all() or not np.isfinite(learned).all():
        raise FloatingPointError("Nonfinite component prediction")
    output = baseline.astype(np.float64).copy()
    output[1:] = baseline[1:] - slow_curve(baseline[1:]) + slow_beta * learned
    # Preserve FM's I head exactly, matching historical evaluation. No target calibration.
    return output.astype(np.float32)
