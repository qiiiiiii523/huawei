"""Target-free body features, ported from the teammate's II adapter v1.

Inputs are uncentered physical uV and explicit validity. Historical beats are
never aligned with current-time beats. No hidden current lead enters this file.
"""
from __future__ import annotations

import numpy as np

BODY_PROTOCOL = "B5-best-body-II-adapter-v1"
BEFORE, AFTER, MAX_BEATS = 100, 200, 32


def normalize_i(lead_i):
    x = np.asarray(lead_i, dtype=np.float32)
    if x.shape != (5000,) or not np.isfinite(x).all():
        raise ValueError("Body adapter requires one finite 10s current I")
    centered = x - np.median(x)
    scale = float(np.sqrt(np.mean(centered.astype(np.float64) ** 2)))
    if not np.isfinite(scale) or scale < 1e-4:
        return np.zeros_like(x), 0.
    return np.clip(centered / scale, -8, 8).astype(np.float32), scale


def clipping_summary(normalized_beats):
    """Measure feature preprocessing saturation; it is not an ECG quality verdict."""
    x = np.asarray(normalized_beats)
    if x.ndim != 3 or x.shape[1:] != (2, 300) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Clipping audit requires finite selected normalized paired beats")
    return {"selected_beats": len(x),
        "I_beat_clipped_fraction": float(np.mean(np.abs(x[:, 0]) > 8)),
        "II_beat_clipped_fraction": float(np.mean(np.abs(x[:, 1]) > 8)),
        "I_peak_before_clip": float(np.max(np.abs(x[:, 0]))),
        "II_peak_before_clip": float(np.max(np.abs(x[:, 1])))}


def relation_features(history, *, audit=None):
    if audit is not None:
        audit.update(candidate_beats=0, selected_beats=0, I_beat_clipped_fraction=None,
            II_beat_clipped_fraction=None, I_peak_before_clip=None, II_peak_before_clip=None)
    empty = (np.zeros((2, 300), np.float32), np.zeros(27, np.float32), False)
    if history is None or history.raw_uV is None or history.valid_mask is None:
        return empty
    if history.source != "body_scale_d6":
        raise ValueError("Body features require body-scale history")
    raw = np.asarray(history.raw_uV, dtype=np.float64)
    mask = np.asarray(history.valid_mask, dtype=bool)
    if raw.ndim != 2 or raw.shape[0] != 6 or mask.shape != raw.shape:
        raise ValueError("History must be raw [6,T] and matching validity")
    mask = mask & np.isfinite(raw)
    if raw.shape[1] < 1000 or mask[:2].sum(axis=1).min() < 1000:
        return empty
    # Lazy scipy import keeps cache/checkpoint inspection lightweight.
    from scipy.signal import butter, find_peaks, sosfiltfilt
    qrs_filter = butter(2, [5, 20], btype="bandpass", fs=500, output="sos")
    candidates, vectors, before_clip = [], [], []
    for start in range(0, raw.shape[1], 5000):
        window, valid = raw[:, start:start + 5000], mask[:, start:start + 5000]
        if valid[0].sum() >= 1000:
            observed = np.where(valid[0], window[0], 0.)
            filtered = sosfiltfilt(qrs_filter, observed)
            energy = np.convolve(filtered ** 2, np.ones(50) / 50, mode="same")
            threshold = max(float(np.quantile(energy[valid[0]], .8)) * .35, 1e-8)
            peaks, _ = find_peaks(energy, distance=140, prominence=threshold)
            for peak in peaks:
                left, right = int(peak) - BEFORE, int(peak) + AFTER
                if left < 0 or right > window.shape[-1]:
                    continue
                support = valid[:2, left:right]
                if support.mean(axis=1).min() < .95:
                    continue
                pair = np.where(support, window[:2, left:right], 0.).astype(np.float32)
                for lead in range(2):
                    pair[lead, support[lead]] -= np.median(pair[lead, support[lead]])
                scale = float(np.sqrt(np.mean(pair[0, support[0]] ** 2)))
                if scale >= 1e-4 and np.isfinite(scale):
                    normalized = pair / scale
                    candidates.append(np.clip(normalized, -8, 8))
                    if audit is not None:
                        before_clip.append(normalized.copy())
        usable = valid.all(axis=0)
        if usable.sum() < 1000:
            continue
        x = window[:, usable]
        x = x - np.median(x, axis=1, keepdims=True)
        scale = float(np.sqrt(np.mean(x[0] ** 2)))
        if scale < 1e-4:
            continue
        x = np.clip(x / scale, -8, 8)
        cov = x @ x.T / x.shape[1]
        rms = np.sqrt(np.diag(cov) + 1e-8)
        corr = cov / np.outer(rms, rms)
        vectors.append(np.r_[corr[np.triu_indices(6)], np.log(np.clip(rms, 1e-4, 1e4))])
    if len(candidates) < 4 or not vectors:
        if audit is not None:
            audit["candidate_beats"] = len(candidates)
        return empty
    chosen = np.linspace(0, len(candidates) - 1, min(MAX_BEATS, len(candidates)), dtype=int)
    beat = np.median(np.stack([candidates[i] for i in chosen]), axis=0)
    beat -= np.median(beat, axis=1, keepdims=True)
    if audit is not None:
        selected = np.stack([before_clip[i] for i in chosen])
        uncut = np.median(selected, axis=0)
        uncut -= np.median(uncut, axis=1, keepdims=True)
        audit.update(clipping_summary(selected), candidate_beats=len(candidates),
                     _unclipped_template=uncut, _clipped_template=beat.copy())
    return beat.astype(np.float32), np.median(vectors, axis=0).astype(np.float32), True


def encode_windows(anchor, base_ii):
    anchor, base_ii = np.asarray(anchor), np.asarray(base_ii)
    if anchor.size != 60000 or base_ii.shape != (60000,) or not np.isfinite(base_ii).all():
        raise ValueError("Body inference requires full 120s visible I and predicted II")
    encoded, scales, bases = [], [], []
    for i, b in zip(anchor.reshape(12, 5000), base_ii.reshape(12, 5000)):
        signal, scale = normalize_i(i)
        encoded.append(signal); scales.append(scale)
        bases.append(np.clip((b - np.median(b)) / max(scale, 1e-4), -12, 12))
    return np.stack(encoded), np.stack(bases).astype(np.float32), np.asarray(scales, np.float32)


def restore_ii(base_ii, encoded_base, model_output, scales):
    """Restore the residual only; never remove or substitute a target mean."""
    correction = np.asarray(model_output) - np.asarray(encoded_base)
    return (np.asarray(base_ii).reshape(12, 5000) + np.asarray(scales)[:, None] * correction).reshape(60000)
