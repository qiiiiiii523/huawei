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
    JOINT_ANCHOR_ADAPTATION = "joint_anchor_adaptation"

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
        if self.supervision_mode == SupervisionMode.D12_I_PRETRAIN.value and self.split != "train":
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


@dataclass(frozen=True)
class JointAnchorSample:
    """Cross-time context plus same-record/window target-time machine-I anchor."""
    context_ecg: np.ndarray
    context_source_type: str
    anchor_i_ecg: np.ndarray
    anchor_source_type: str
    Y_12lead: np.ndarray
    anchor_lead_mask: np.ndarray
    context_lead_mask: np.ndarray
    task_id: str
    split: str
    subject_id: str
    pair_id: str
    target_record_id: str
    window_id: str
    meta: dict[str, Any]
    input_type: str | None = None
    context_target_relation: str = "same_subject_cross_time"
    anchor_target_relation: str = "same_record_same_window"
    context_target_sync: bool = False
    anchor_target_sync: bool = True
    pointwise_loss_allowed: bool = True
    anchor_available_at_test: bool = True
    supervision_mode: str = SupervisionMode.JOINT_ANCHOR_ADAPTATION.value

    def validate(self) -> None:
        if self.task_id not in {"task1", "task2"} or self.split not in {"train", "validation"}:
            raise ContractError("Joint-anchor task_id/split is invalid")
        expected = 1 if self.task_id == "task1" else int(self.context_lead_mask.sum())
        if self.context_ecg.shape != (expected, WINDOW_SAMPLES): raise ContractError("context shape does not match actual context channels")
        if self.task_id == "task1" and self.context_source_type != "watch_ecg": raise ContractError("task1 context must be watch_ecg")
        if self.task_id == "task2" and self.context_source_type not in {"ecg_machine_d6", "body_scale_d6"}: raise ContractError("task2 context must be d6")
        if self.anchor_i_ecg.shape != (1, WINDOW_SAMPLES) or self.Y_12lead.shape != (12, WINDOW_SAMPLES): raise ContractError("anchor/target shapes are invalid")
        if self.anchor_source_type != "ecg_machine_i": raise ContractError("anchor must be ecg_machine_i")
        if self.anchor_lead_mask.shape != (12,) or not (self.anchor_lead_mask[0] and self.anchor_lead_mask.sum() == 1): raise ContractError("only I is observed at target time")
        if self.context_target_relation != "same_subject_cross_time" or self.anchor_target_relation != "same_record_same_window": raise ContractError("wrong relation labels")
        if self.context_target_sync or not self.anchor_target_sync or not self.pointwise_loss_allowed or not self.anchor_available_at_test: raise ContractError("wrong sync/supervision flags")
        if self.supervision_mode != SupervisionMode.JOINT_ANCHOR_ADAPTATION.value: raise ContractError("wrong supervision mode")
        required = {"anchor_construction", "anchor_target_record_id", "anchor_window_id"}
        if not required.issubset(self.meta) or self.meta["anchor_target_record_id"] != self.target_record_id or self.meta["anchor_window_id"] != self.window_id: raise ContractError("anchor provenance must be same target record/window")

@dataclass(frozen=True)
class JointAnchorInferenceInput:
    """Public test input; hidden targets are deliberately absent."""
    context_ecg: np.ndarray
    context_source_type: str
    anchor_i_ecg: np.ndarray
    task_id: str
    context_channel_indices: tuple[int, ...] | None = None
    def validate(self) -> None:
        expected = 1 if self.task_id == "task1" else len(self.context_channel_indices or tuple(range(6)))
        if self.task_id not in {"task1", "task2"} or self.context_ecg.shape != (expected, WINDOW_SAMPLES) or self.anchor_i_ecg.shape != (1, WINDOW_SAMPLES): raise ContractError("explicit context and machine-I anchor are required")

def prepare_joint_anchor_inference(context_ecg: np.ndarray, *, context_source_type: str, task_id: str, anchor_i_ecg: np.ndarray | None, context_channel_indices: tuple[int, ...] | None = None) -> JointAnchorInferenceInput:
    if anchor_i_ecg is None: raise ContractError("Joint-anchor inference requires an explicit machine-I anchor")
    item = JointAnchorInferenceInput(np.asarray(context_ecg), context_source_type, np.asarray(anchor_i_ecg), task_id, context_channel_indices)
    item.validate()
    return item
