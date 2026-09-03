"""Task1 train-only R-peak pseudo-pair utilities; pseudo alignment is not physical sync."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Beat:
    peak: int
    start: int
    end: int
    rr: float
    qrs_width: float
    waveform: np.ndarray


def centered_for_detection(signal: np.ndarray) -> np.ndarray:
    return np.asarray(signal, dtype=np.float32) - np.median(signal)


def detect_xqrs(centered_signal: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
    try:
        from wfdb import processing  # type: ignore
    except ImportError as error:
        raise RuntimeError("R-peak pseudo-pairing requires wfdb (wfdb.processing.xqrs_detect)") from error
    return np.asarray(processing.xqrs_detect(sig=centered_signal, fs=sampling_rate_hz, verbose=False), dtype=np.int64)


def _qrs_width(waveform: np.ndarray, r_index: int) -> float:
    peak = abs(float(waveform[r_index]))
    if peak <= 1e-6:
        return 0.0
    active = np.flatnonzero(np.abs(waveform) >= peak * 0.5)
    return float(active[-1] - active[0] + 1) if active.size else 0.0


def beats(signal: np.ndarray, peaks: Iterable[int], pre: int, post: int) -> list[Beat]:
    values = np.asarray(signal, dtype=np.float32)
    valid = [int(peak) for peak in peaks if pre <= int(peak) and int(peak) + post <= values.size]
    result = []
    for index, peak in enumerate(valid):
        wave = values[peak - pre:peak + post]
        rr = float(valid[index] - valid[index - 1]) if index else float("nan")
        result.append(Beat(peak, peak - pre, peak + post, rr, _qrs_width(wave, pre), wave))
    return result


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denominator) if denominator > 1e-8 else -1.0


def monotonic_match(input_beats: list[Beat], target_beats: list[Beat]) -> list[tuple[int, int, float]]:
    """Order-preserving one-to-one dynamic-programming match, allowing gaps."""
    n, m = len(input_beats), len(target_beats)
    score = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)
    score[0, 0] = 0.0
    gap = -0.25
    for i in range(n + 1):
        for j in range(m + 1):
            current = score[i, j]
            if not np.isfinite(current):
                continue
            if i < n and current + gap > score[i + 1, j]: score[i + 1, j], move[i + 1, j] = current + gap, 1
            if j < m and current + gap > score[i, j + 1]: score[i, j + 1], move[i, j + 1] = current + gap, 2
            if i < n and j < m:
                similarity = _corr(input_beats[i].waveform, target_beats[j].waveform)
                rr_delta = 0.0 if not np.isfinite(input_beats[i].rr) or not np.isfinite(target_beats[j].rr) else abs(input_beats[i].rr - target_beats[j].rr) / max(input_beats[i].rr, target_beats[j].rr, 1.0)
                width_delta = abs(input_beats[i].qrs_width - target_beats[j].qrs_width) / 100.0
                pair_score = similarity - 0.25 * rr_delta - 0.10 * width_delta
                if current + pair_score > score[i + 1, j + 1]: score[i + 1, j + 1], move[i + 1, j + 1] = current + pair_score, 3
    matched = []
    i, j = n, m
    while i or j:
        direction = move[i, j]
        if direction == 3:
            similarity = _corr(input_beats[i - 1].waveform, target_beats[j - 1].waveform)
            matched.append((i - 1, j - 1, similarity)); i, j = i - 1, j - 1
        elif direction == 1: i -= 1
        elif direction == 2: j -= 1
        else: break
    return list(reversed(matched))


def target_time_mapping(matches: list[tuple[int, int, float]], input_beats: list[Beat], target_beats: list[Beat], samples: int) -> np.ndarray | None:
    if len(matches) < 2:
        return None
    source = np.asarray([input_beats[i].peak for i, _, _ in matches], dtype=np.float64)
    destination = np.asarray([target_beats[j].peak for _, j, _ in matches], dtype=np.float64)
    if np.any(np.diff(source) <= 0) or np.any(np.diff(destination) <= 0):
        return None
    grid = np.arange(samples, dtype=np.float64)
    mapped = np.interp(grid, source, destination)
    left = grid < source[0]
    right = grid > source[-1]
    mapped[left] = destination[0] + (grid[left] - source[0]) * (destination[1] - destination[0]) / (source[1] - source[0])
    mapped[right] = destination[-1] + (grid[right] - source[-1]) * (destination[-1] - destination[-2]) / (source[-1] - source[-2])
    if mapped.min() < 0 or mapped.max() > samples - 1 or np.any(np.diff(mapped) <= 0):
        return None
    return mapped


def warp_d12_to_input_grid(raw_d12: np.ndarray, mapped_target_time: np.ndarray) -> np.ndarray:
    source = np.arange(raw_d12.shape[1], dtype=np.float64)
    return np.stack([np.interp(mapped_target_time, source, raw_d12[lead]) for lead in range(12)]).astype(np.float32)