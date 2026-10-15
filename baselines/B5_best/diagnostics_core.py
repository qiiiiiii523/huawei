"""Read-only waveform diagnostics. No models, training, or inference imports."""
from __future__ import annotations

import numpy as np

RATE = 500


def corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correlation requires matching finite waveforms")
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.clip(np.sum(a * b) / d, -1, 1)) if d > 1e-12 else None


def moving_average(x, width=501):
    x = np.asarray(x, dtype=float)
    if x.ndim < 1 or width < 3 or width % 2 != 1 or not np.isfinite(x).all():
        raise ValueError("Moving average needs finite data and odd width >=3")
    r = width // 2
    ext = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(r, r)], mode="edge")
    acc = np.concatenate((np.zeros_like(ext[..., :1]), np.cumsum(ext, axis=-1)), axis=-1)
    return (acc[..., width:] - acc[..., :-width]) / width


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-12 else None


def high_energy_fraction(x, frequency=1.):
    """Descriptive FFT statistic; 501-point averaging is NOT an ideal 1-Hz filter."""
    x = np.asarray(x, float) - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    frequencies = np.fft.rfftfreq(len(x), 1 / RATE)
    # Correct the one-sided weights (DC is zero after centering).
    if len(x) % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2
    return ratio(power[frequencies > frequency].sum(), power.sum())


def qrs_candidates(anchor):
    """I-only approximate QRS anchors, NOT clinical annotation/delineation."""
    from scipy.signal import butter, sosfiltfilt, find_peaks
    anchor = np.asarray(anchor, float)
    if anchor.ndim != 1 or not np.isfinite(anchor).all():
        raise ValueError("QRS anchor must be finite observed I")
    if len(anchor) < RATE * 2 or np.std(anchor) < 1e-8:
        return np.array([], dtype=int)
    filtered = sosfiltfilt(butter(3, (5, 25), fs=RATE, btype="bandpass", output="sos"), anchor)
    energy = moving_average(np.diff(filtered, prepend=filtered[0]) ** 2, 41)
    med = np.median(energy)
    mad = np.median(np.abs(energy - med))
    peaks, _ = find_peaks(energy, distance=int(.3 * RATE),
                          height=med + max(3 * mad, .04 * (np.max(energy) - med)),
                          prominence=max(2 * mad, .02 * (np.max(energy) - med)))
    refined = []
    for p in peaks:
        a, b = max(0, p - 35), min(len(anchor), p + 36)
        q = a + int(np.argmax(np.abs(filtered[a:b])))
        if int(.25 * RATE) <= q < len(anchor) - int(.45 * RATE):
            if refined and q - refined[-1] < int(.25 * RATE):
                if abs(filtered[q]) > abs(filtered[refined[-1]]):
                    refined[-1] = q
            else:
                refined.append(q)
    return np.asarray(refined, dtype=int)


def half_width(segment, index, polarity):
    z = polarity * np.asarray(segment)
    if z[index] <= 0:
        return None
    level = z[index] / 2
    left = right = int(index)
    while left > 0 and z[left - 1] >= level:
        left -= 1
    while right + 1 < len(z) and z[right + 1] >= level:
        right += 1
    # An event clipped at the search edge has unknown width, not a real broad peak.
    if left == 0 or right == len(z) - 1:
        return None
    return (right - left + 1) * 1000 / RATE


def local_shape(predicted_fast, target_fast, anchors):
    """Same I anchors for all methods; target defines polarity only for diagnosis."""
    samples = []
    for q in anchors:
        a, b = q - 50, q + 61  # [-100,+120] ms; fixed, not ECG delineation.
        if a < 0 or b > len(target_fast):
            continue
        y = np.asarray(target_fast[a:b], float).copy()
        p = np.asarray(predicted_fast[a:b], float).copy()
        y -= np.median(target_fast[q - 100:q - 65])
        p -= np.median(predicted_fast[q - 100:q - 65])
        iy = int(np.argmax(np.abs(y)))
        sign = 1 if y[iy] >= 0 else -1
        ip = int(np.argmax(sign * p))
        if abs(y[iy]) < 5:  # Avoid noise-level amplitude ratios.
            continue
        wy, wp = half_width(y, iy, sign), half_width(p, ip, sign)
        zero = corr(p, y)
        shifted = []
        for lag in range(-10, 11):  # Diagnostic-only +/-20ms, NEVER applied to scoring.
            pp = p[max(0, lag):min(len(p), len(p) + lag)]
            yy = y[max(0, -lag):min(len(y), len(y) - lag)]
            value = corr(pp, yy)
            if value is not None:
                shifted.append(value)
        samples.append({"abs_peak_time_error_ms": abs(ip - iy) * 1000 / RATE,
                        "peak_amplitude_abs_error_uV": abs(p[ip] - y[iy]),
                        "peak_amplitude_ratio": float(p[ip] / y[iy]),
                        "peak_polarity_wrong": int(sign * p[ip] < 0),
                        "half_width_abs_error_ms": None if wy is None or wp is None else abs(wp - wy),
                        "qrs_r_unshifted": zero,
                        "qrs_r_best_lag_DIAGNOSTIC_ONLY": max(shifted) if shifted else None})
    result = {"n_qrs_candidates": int(len(anchors)), "n_eligible_beats": len(samples)}
    fields = ("abs_peak_time_error_ms", "peak_amplitude_abs_error_uV", "peak_amplitude_ratio",
              "half_width_abs_error_ms", "qrs_r_unshifted", "qrs_r_best_lag_DIAGNOSTIC_ONLY")
    for key in fields:
        values = [x[key] for x in samples if x[key] is not None]
        result[key + "_median"] = float(np.median(values)) if values else None
        result[key + "_p90"] = float(np.quantile(values, .9)) if values else None
    result["peak_polarity_wrong_fraction"] = (float(np.mean([s["peak_polarity_wrong"] for s in samples]))
                                              if samples else None)
    return result


def component_metrics(fm, slow, target):
    """One QC-good lead, complete record. No target-based output calibration."""
    f = fm - moving_average(fm)
    ys = moving_average(target)
    yf = target - ys
    hybrid = f + slow
    hf = hybrid - moving_average(hybrid)
    out = {"fm_raw_r": corr(fm, target), "hybrid_raw_r": corr(hybrid, target),
           "fm_fast_r": corr(f, yf), "hybrid_fast_r": corr(hf, yf),
           "slow_prediction_r": corr(slow, ys),
           "slow_std_ratio_vs_target": ratio(np.std(slow), np.std(ys)),
           "fm_fast_std_ratio_vs_target": ratio(np.std(f), np.std(yf)),
           "predicted_slow_to_fast_std_ratio": ratio(np.std(slow), np.std(f)),
           "target_slow_to_fast_std_ratio": ratio(np.std(ys), np.std(yf)),
           "slow_prediction_fft_energy_above1Hz_fraction": high_energy_fraction(slow),
           "slow_target_fft_energy_above1Hz_fraction": high_energy_fraction(ys),
           "slow_faster_residual_error_rms_uV": rms((slow - moving_average(slow)) - (ys - moving_average(ys)))}
    for beta in (.75, 1., 1.25):
        out[f"raw_r_beta_{beta:g}"] = corr(f + beta * slow, target)
    return out


def boundary_metrics(prediction, target):
    err = np.asarray(prediction) - target
    centered = err - err.mean()
    positions = np.arange(len(err))
    phase = positions % 5000
    near = ((phase < 100) | (phase >= 4900)) & (positions >= 4900) & (positions < len(err) - 4900)
    middle = (phase >= 1000) & (phase < 4000)
    seams = np.arange(5000, len(err), 5000)
    return {"centered_error_rms_near_seam_uV": rms(centered[near]),
            "centered_error_rms_window_middle_uV": rms(centered[middle]),
            "seam_error_one_sample_jump_mae_uV": float(np.mean(np.abs(err[seams] - err[seams - 1])))}
