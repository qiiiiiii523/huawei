"""D0: stable, framework-neutral data structures and invariants."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np

D12_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
D6_LEADS = D12_LEADS[:6]
ECG_SAMPLING_RATE_HZ = 500
WINDOW_SECONDS = 10
WINDOW_SAMPLES = ECG_SAMPLING_RATE_HZ * WINDOW_SECONDS

class ContractError(ValueError):
    """Raised when data cannot satisfy the agreed D0/D1 contract."""

class SupervisionMode(str, Enum):
    D12_I_PRETRAIN = "d12_i_pretrain"
    D12_SIX_PRETRAIN = "d12_six_pretrain"
    CROSS_DEVICE_WEAK_ADAPTATION = "cross_device_weak_adaptation"

@dataclass(frozen=True)
class ECGSample:
    """One D0 item; masks always use canonical twelve-lead order."""
    X_ecg: np.ndarray
    lead_mask: np.ndarray
    Y_12lead: np.ndarray
    missing_mask: np.ndarray
    task_id: str
    ppg: np.ndarray | None
    acc: np.ndarray | None
    meta: dict[str, Any]
    modality_mask: dict[str, bool]
    split: str
    supervision_mode: str
    pairing_type: str
    alignment_mode: str
    pair_confidence: str
    pair_status: str

    def validate(self) -> None:
        if self.task_id not in {"task1", "task2"}:
            raise ContractError(f"Unknown task_id: {self.task_id}")
        expected_inputs = 1 if self.task_id == "task1" else 6
        if self.X_ecg.shape != (expected_inputs, WINDOW_SAMPLES):
            raise ContractError(f"{self.task_id} X_ecg must be ({expected_inputs}, {WINDOW_SAMPLES}), got {self.X_ecg.shape}")
        if self.Y_12lead.shape != (12, WINDOW_SAMPLES):
            raise ContractError(f"Y_12lead must be (12, {WINDOW_SAMPLES}), got {self.Y_12lead.shape}")
        if self.lead_mask.shape != (12,) or self.missing_mask.shape != (12,):
            raise ContractError("lead_mask and missing_mask must have 12 entries")
        if not np.array_equal(self.missing_mask, ~self.lead_mask):
            raise ContractError("missing_mask must be the complement of lead_mask")
        if int(self.lead_mask.sum()) != expected_inputs:
            raise ContractError("lead_mask does not describe X_ecg")
        if self.split not in {"train", "validation"}:
            raise ContractError(f"Unknown split: {self.split}")
        if self.supervision_mode in {SupervisionMode.D12_I_PRETRAIN.value, SupervisionMode.D12_SIX_PRETRAIN.value} and self.split != "train":
            raise ContractError("D12 pretraining is train-only; validation targets must not be used for it")
        for name, value in (("ppg", self.ppg), ("acc", self.acc)):
            if self.modality_mask.get(name, False) != (value is not None):
                raise ContractError(f"modality_mask[{name!r}] disagrees with {name}")

def canonical_lead_mask(input_leads: int) -> np.ndarray:
    if input_leads not in {1, 6}:
        raise ContractError("Only 1-lead and 6-lead ECG inputs are supported")
    mask = np.zeros(12, dtype=bool)
    mask[:input_leads] = True
    return mask
