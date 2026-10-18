"""Record-level gate inputs: visible I, predicted II and historical features ONLY."""
from __future__ import annotations

import numpy as np

from baselines.B5_best.diagnostics_core import moving_average

GATE_PROTOCOL = "B5-best-body-record-gate-v1"
FEATURE_PROTOCOL = "visible-record-gate-48-v1"
FEATURE_COUNT = 48


def visible_record_features(anchor, predicted_ii, beat, stats, available):
    i, p = np.asarray(anchor, float).reshape(-1), np.asarray(predicted_ii, float)
    beat, stats = np.asarray(beat, float), np.asarray(stats, float)
    if i.shape != (60000,) or p.shape != (60000,) or beat.shape != (2, 300) or stats.shape != (27,):
        raise ValueError("Gate inputs require full120s visible I/predicted II and cached history features")
    if any(not np.isfinite(x).all() for x in (i, p, beat, stats)):
        raise ValueError("Gate features must be finite")
    if not bool(available):
        beat, stats = np.zeros_like(beat), np.zeros_like(stats)
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))
    template = []
    for x in beat:
        peak = float(np.max(np.abs(x)))
        # This is a template plateau indicator, NOT the original clipping fraction.
        plateau = float(np.mean(np.abs(np.abs(x) - peak) < max(peak * 1e-5, 1e-8))) if peak > 0 else 0.
        template.extend((rms(x), peak, float(np.quantile(np.abs(x), .95)),
            rms(np.diff(x)), plateau, float(x[np.argmax(np.abs(x))] / max(peak, 1e-8))))
    i, p = i - np.median(i), p - np.median(p)
    ri, rp = max(rms(i), 1e-4), max(rms(p), 1e-4)
    denom = float(np.linalg.norm(i - i.mean()) * np.linalg.norm(p - p.mean()))
    correlation = float(np.dot(i - i.mean(), p - p.mean()) / denom) if denom else 0.
    window_rms = np.sqrt(np.mean(i.reshape(12, 5000) ** 2, axis=1))
    current = (np.log1p(ri), np.log(max(rp / ri, 1e-8)), correlation,
        rms(moving_average(i)) / ri, rms(moving_average(p)) / rp,
        rms(np.diff(i)) / ri, rms(np.diff(p)) / rp,
        float(window_rms.std() / max(window_rms.mean(), 1e-4)), float(bool(available)))
    result = np.r_[stats, template, current].astype(np.float32)
    if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ValueError("Invalid visible record feature vector")
    return result


def fit_normalization(training_features):
    x = np.asarray(training_features, np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_COUNT or not len(x) or not np.isfinite(x).all():
        raise ValueError("Fit gate normalization on training visible features only")
    return {"protocol": FEATURE_PROTOCOL, "mean": x.mean(0).tolist(),
            "std": np.maximum(x.std(0), 1e-4).tolist(), "training_records": len(x)}


def normalize_features(features, normalization):
    if normalization.get("protocol") != FEATURE_PROTOCOL:
        raise ValueError("Gate feature protocol mismatch")
    mean, std = np.asarray(normalization["mean"]), np.asarray(normalization["std"])
    if mean.shape != (FEATURE_COUNT,) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid train-fitted gate normalization")
    x = np.asarray(features)
    if x.shape[-1:] != (FEATURE_COUNT,) or not np.isfinite(x).all():
        raise ValueError("Invalid gate feature shape/value")
    return np.clip((x - mean) / std, -10, 10).astype(np.float32)
