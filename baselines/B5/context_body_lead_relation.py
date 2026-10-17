"""Patient-specific body-scale lead-I to lead-II relation conditioning.

Support beats contain simultaneous body-scale I and II. Odd-indexed beats are
held out: their I is a query and their II is used only for the auxiliary loss.
The target-time ECG contributes its observed I through the existing B5 anchor
path; no target II or cross-device beat alignment enters this encoder.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, sosfiltfilt
from torch import nn

from .config import ModelConfig
from .context_body import B5BodyContext
from .context_body_attention import BodyContextEncoderAttention
from .context_body_beat_rhythm import AFTER, BEFORE, FS, MAX_BEATS, QRS_FILTER


KERNEL_WIDTH = 9
MIN_SUPPORT_BEATS = 2
MIN_QUERY_BEATS = 2


def extract_lead_pair_beats(context: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Take paired body I/II beats; normalize using I alone to avoid II leakage."""
    if context.ndim != 3 or context.shape[1:] != (6, 5000) or mask.shape != context.shape:
        raise ValueError("Body context must be matching [W,6,5000] arrays")
    result = np.zeros((MAX_BEATS, 4, BEFORE + AFTER), dtype=np.float32)
    present = np.zeros(MAX_BEATS, dtype=bool)
    candidates = []
    for window, valid in zip(context, mask):
        if valid[0].sum() < 1000:
            continue
        observed_i = np.where(valid[0], window[0], 0).astype(np.float64)
        filtered = sosfiltfilt(QRS_FILTER, observed_i)
        energy = np.convolve(np.square(filtered), np.ones(50) / 50, mode="same")
        threshold = max(float(np.quantile(energy[valid[0]], 0.8)) * 0.35, 1e-8)
        peaks, _ = find_peaks(energy, distance=int(0.28 * FS), prominence=threshold)
        for peak in peaks:
            start, stop = int(peak) - BEFORE, int(peak) + AFTER
            if start < 0 or stop > window.shape[-1]:
                continue
            support = valid[:2, start:stop]
            if support.mean(axis=1).min() < 0.95:
                continue
            pair = np.where(support, window[:2, start:stop], 0).astype(np.float32)
            for lead in range(2):
                pair[lead, support[lead]] -= np.median(pair[lead, support[lead]])
            i_rms = float(np.sqrt(np.mean(np.square(pair[0, support[0]]))))
            if not np.isfinite(i_rms) or i_rms < 1e-4:
                continue
            pair = np.clip(pair / i_rms, -8, 8)
            candidates.append(np.concatenate((pair, support.astype(np.float32)), axis=0))
    if candidates:
        chosen = np.linspace(0, len(candidates) - 1, min(MAX_BEATS, len(candidates)), dtype=int)
        for slot, index in enumerate(chosen):
            result[slot] = candidates[index]
            present[slot] = True
    return result, present


class BodyContextEncoderLeadRelation(BodyContextEncoderAttention):
    """Frozen attention path plus a learned within-patient I-to-II map."""

    def __init__(self, dimension: int = 128) -> None:
        super().__init__(dimension)
        self.relation_encoder = nn.Sequential(
            nn.Conv1d(4, 24, 7, stride=2, padding=3), nn.SiLU(),
            nn.Conv1d(24, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, dimension), nn.LayerNorm(dimension),
        )
        # A patient-conditioned short temporal filter maps each held-out I beat
        # to II. The centre tap begins near identity, without seeing query II.
        self.transfer_head = nn.Linear(dimension, KERNEL_WIDTH + 1)
        nn.init.normal_(self.transfer_head.weight, std=0.005)
        nn.init.zeros_(self.transfer_head.bias)
        with torch.no_grad():
            self.transfer_head.bias[KERNEL_WIDTH // 2] = 1.0
        self.relation_gate = nn.Parameter(torch.tensor(0.03))
        self.last_auxiliary_loss: torch.Tensor | None = None
        self.last_valid_records = 0
        self._pair_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _features(self, context: torch.Tensor, time_mask: torch.Tensor):
        raw = context.detach().cpu().numpy()
        masks = time_mask.detach().cpu().numpy().astype(bool, copy=False)
        records = []
        for values, valid in zip(raw, masks):
            occupied = np.flatnonzero(valid.any(axis=(1, 2)))
            end = int(occupied[-1]) + 1 if len(occupied) else 0
            digest = hashlib.sha256(values[:end].tobytes() + valid[:end].tobytes()).hexdigest()
            if digest not in self._pair_cache:
                if len(self._pair_cache) >= 512:
                    self._pair_cache.clear()
                self._pair_cache[digest] = extract_lead_pair_beats(values[:end], valid[:end])
            records.append(self._pair_cache[digest])
        beat = torch.from_numpy(np.stack([row[0] for row in records])).to(context.device)
        present = torch.from_numpy(np.stack([row[1] for row in records])).to(context.device)
        return beat, present

    @staticmethod
    def _relation_beats(beat: torch.Tensor, present: torch.Tensor):
        # Channels 0/1 are body I/II, and 2/3 are their validity masks.
        support, support_mask = beat[:, ::2], present[:, ::2]
        query, query_mask = beat[:, 1::2], present[:, 1::2]
        return support, support_mask, query, query_mask

    def _encode_relation(self, support: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        b, n = support.shape[:2]
        observed = support.reshape(b * n, 4, support.shape[-1])
        encoded = self.relation_encoder(observed).reshape(b, n, -1)
        return ((encoded * support_mask.unsqueeze(-1)).sum(dim=1)
                / support_mask.sum(dim=1).clamp_min(1).unsqueeze(-1))

    def _held_out_loss(self, query: torch.Tensor, query_mask: torch.Tensor,
                       relation: torch.Tensor, valid_records: torch.Tensor) -> torch.Tensor:
        # Query-II never appears in _encode_relation or in the dynamic filter's
        # input. It is supervised only here, within the same body-scale record.
        query_i = query[:, :, 0]
        query_ii = query[:, :, 1]
        observed = (query[:, :, 2] > 0.5) & (query[:, :, 3] > 0.5)
        observed = observed & query_mask.unsqueeze(-1) & valid_records[:, None, None]
        coefficients = self.transfer_head(relation)
        taps, bias = coefficients[:, :KERNEL_WIDTH], coefficients[:, KERNEL_WIDTH]
        windows = F.pad(query_i, (KERNEL_WIDTH // 2, KERNEL_WIDTH // 2)).unfold(-1, KERNEL_WIDTH, 1)
        predicted = (windows * taps[:, None, None, :]).sum(dim=-1) + bias[:, None, None]
        error = F.smooth_l1_loss(predicted, query_ii, beta=0.2, reduction="none")
        return (error * observed).sum() / observed.sum().clamp_min(1)

    def forward(self, context: torch.Tensor, time_mask: torch.Tensor,
                window_weights: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        # The older beat/RR additions are not used: isolate lead relation.
        baseline = BodyContextEncoderAttention.forward(
            self, context, time_mask, window_weights, available)
        beat, present = self._features(context, time_mask)
        support, support_mask, query, query_mask = self._relation_beats(beat, present)
        valid_records = (support_mask.sum(dim=1) >= MIN_SUPPORT_BEATS) & (
            query_mask.sum(dim=1) >= MIN_QUERY_BEATS) & available.bool()
        relation = self._encode_relation(support, support_mask)
        self.last_valid_records = int(valid_records.sum().item())
        self.last_auxiliary_loss = (
            self._held_out_loss(query, query_mask, relation, valid_records)
            if torch.is_grad_enabled() else context.new_zeros(()))
        return baseline + torch.tanh(self.relation_gate) * relation * valid_records.unsqueeze(-1)


class B5BodyContextLeadRelation(B5BodyContext):
    """B5 conditioned by an explicitly trained body I-to-II relationship."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__(config)
        self.body_encoder = BodyContextEncoderLeadRelation(config.metadata_dim)
