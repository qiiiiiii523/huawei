"""Optional canonical 12-lead view for architectures that require fixed input channels."""
from __future__ import annotations

import numpy as np

from .contracts import ContractError, canonical_lead_mask


def canonicalize_input_ecg(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad canonical 1/6-lead ECG to 12 leads and return its lead mask.

    D0 raw shapes remain unchanged in ``UnifiedECGDataset``. This adapter is
    opt-in for masked fixed-channel models; absent leads are zero placeholders
    and must always be interpreted with the returned mask.
    """
    array = np.asarray(values)
    if array.ndim == 2:
        channels, samples = array.shape
        if channels not in {1, 6}:
            raise ContractError("Canonical adapter accepts 1- or 6-lead ECG")
        output = np.zeros((12, samples), dtype=array.dtype)
        output[:channels] = array
        return output, canonical_lead_mask(channels)
    if array.ndim == 3:
        batch, channels, samples = array.shape
        if channels not in {1, 6}:
            raise ContractError("Canonical adapter accepts [N,1,T] or [N,6,T] ECG")
        output = np.zeros((batch, 12, samples), dtype=array.dtype)
        output[:, :channels] = array
        return output, np.broadcast_to(canonical_lead_mask(channels), (batch, 12)).copy()
    raise ContractError("ECG must have shape [C,T] or [N,C,T]")