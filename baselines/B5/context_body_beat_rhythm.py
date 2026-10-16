"""Independent-record body ECG context with explicit beat and RR branches.

R peaks are detected only on the observed body-scale record. Detection and
beat selection are fixed preprocessing; the beat CNN and rhythm MLP are learned
from the missing-lead flow objective. No target ECG timing is consulted.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from scipy.signal import butter, find_peaks, sosfiltfilt
from torch import nn

from .config import ModelConfig
from .context_body import B5BodyContext
from .context_body_attention import BodyContextEncoderAttention


FS = 500
BEFORE = 100
AFTER = 200
MAX_BEATS = 32
RHYTHM_FEATURES = 6
QRS_FILTER = butter(2, [5, 20], btype="bandpass", fs=FS, output="sos")
MODES = ("beat", "rhythm", "beat_rhythm")


def extract_body_beats(context: np.ndarray, time_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return masked six-lead beat snippets and robust RR statistics for one record."""
    if context.ndim != 3 or context.shape[1:] != (6, 5000) or time_mask.shape != context.shape:
        raise ValueError("Body context must be matching [W,6,5000] arrays")
    beats = np.zeros((MAX_BEATS, 12, BEFORE + AFTER), dtype=np.float32)
    present = np.zeros(MAX_BEATS, dtype=bool)
    rhythm = np.zeros(RHYTHM_FEATURES, dtype=np.float32)
    waveforms, times = [], []
    for window_index, (window, valid) in enumerate(zip(context, time_mask)):
        counts = valid.sum(axis=1)
        lead = 0 if counts[0] >= 1000 else int(np.argmax(counts))
        if counts[lead] < 1000:
            continue
        observed = np.where(valid[lead], window[lead], 0.0).astype(np.float64)
        filtered = sosfiltfilt(QRS_FILTER, observed)
        energy = np.convolve(filtered * filtered, np.ones(50) / 50, mode="same")
        threshold = max(float(np.quantile(energy[valid[lead]], 0.8)) * 0.35, 1e-8)
        peaks, _ = find_peaks(energy, distance=int(0.28 * FS), prominence=threshold)
        for peak in peaks:
            start, stop = int(peak) - BEFORE, int(peak) + AFTER
            if start < 0 or stop > window.shape[-1] or valid[lead, start:stop].mean() < 0.95:
                continue
            support = valid[:, start:stop]
            beat = np.where(support, window[:, start:stop], 0).astype(np.float32)
            for channel in range(6):
                if support[channel].any():
                    beat[channel, support[channel]] -= np.median(beat[channel, support[channel]])
            rms = float(np.sqrt(np.mean(np.square(beat[support])))) if support.any() else 0.0
            if not np.isfinite(rms) or rms < 1e-4:
                continue
            normalized = np.clip(beat / rms, -8, 8)
            waveforms.append(np.concatenate((normalized, support.astype(np.float32)), axis=0))
            times.append(window_index * 10 + int(peak) / FS)

    if waveforms:
        chosen = np.linspace(0, len(waveforms) - 1, min(MAX_BEATS, len(waveforms)), dtype=int)
        for slot, index in enumerate(chosen):
            beats[slot] = waveforms[index]
            present[slot] = True
    rr = np.diff(np.asarray(times, dtype=np.float64))
    good = rr[(rr >= 0.33) & (rr <= 1.8)]
    rhythm_available = len(good) >= 4
    if rhythm_available:
        rhythm[:] = (np.median(good), np.mean(good), np.std(good),
                     np.quantile(good, 0.1), np.quantile(good, 0.9),
                     len(good) / max(len(rr), 1))
    return beats, present, rhythm, rhythm_available


class BodyContextEncoderBeatRhythm(BodyContextEncoderAttention):
    """Frozen attention baseline plus small, independently gated feature paths."""

    def __init__(self, dimension: int = 128, mode: str = "beat_rhythm") -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown beat/rhythm mode: {mode}")
        super().__init__(dimension)
        self.mode = mode
        self.beat_branch = nn.Sequential(
            nn.Conv1d(12, 24, 7, stride=2, padding=3), nn.SiLU(),
            nn.Conv1d(24, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, dimension), nn.SiLU(), nn.LayerNorm(dimension),
        )
        self.rhythm_branch = nn.Sequential(
            nn.Linear(RHYTHM_FEATURES, 32), nn.SiLU(),
            nn.Linear(32, dimension), nn.LayerNorm(dimension),
        )
        self.beat_gate = nn.Parameter(torch.tensor(0.01))
        self.rhythm_gate = nn.Parameter(torch.tensor(0.01))
        self._feature_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = {}

    def _features(self, context: torch.Tensor, time_mask: torch.Tensor):
        raw = context.detach().cpu().numpy()
        masks = time_mask.detach().cpu().numpy().astype(bool, copy=False)
        records = []
        for values, valid in zip(raw, masks):
            occupied = np.flatnonzero(valid.any(axis=(1, 2)))
            end = int(occupied[-1]) + 1 if len(occupied) else 0
            digest = hashlib.sha256(values[:end].tobytes() + valid[:end].tobytes()).hexdigest()
            if digest not in self._feature_cache:
                if len(self._feature_cache) >= 512:
                    self._feature_cache.clear()
                self._feature_cache[digest] = extract_body_beats(values[:end], valid[:end])
            records.append(self._feature_cache[digest])
        beat = torch.from_numpy(np.stack([row[0] for row in records])).to(context.device)
        present = torch.from_numpy(np.stack([row[1] for row in records])).to(context.device)
        rhythm = torch.from_numpy(np.stack([row[2] for row in records])).to(context.device)
        rhythm_available = torch.as_tensor([row[3] for row in records], device=context.device)
        return beat, present, rhythm, rhythm_available

    def forward(self, context: torch.Tensor, time_mask: torch.Tensor,
                window_weights: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        embedding = super().forward(context, time_mask, window_weights, available)
        beat, present, rhythm, rhythm_available = self._features(context, time_mask)
        if self.mode in ("beat", "beat_rhythm"):
            encoded = self.beat_branch(beat.flatten(0, 1)).reshape(len(beat), MAX_BEATS, -1)
            count = present.sum(dim=1)
            pooled = (encoded * present.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1).unsqueeze(-1)
            embedding = embedding + torch.tanh(self.beat_gate) * pooled * (count >= 3).unsqueeze(-1)
        if self.mode in ("rhythm", "beat_rhythm"):
            encoded = self.rhythm_branch(rhythm)
            embedding = embedding + torch.tanh(self.rhythm_gate) * encoded * rhythm_available.unsqueeze(-1)
        return embedding


class B5BodyContextBeatRhythm(B5BodyContext):
    """B5 with the previously trained attention path plus beat/RR branches."""

    def __init__(self, config: ModelConfig = ModelConfig(), mode: str = "beat_rhythm") -> None:
        super().__init__(config)
        self.body_encoder = BodyContextEncoderBeatRhythm(config.metadata_dim, mode)
