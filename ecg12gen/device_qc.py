"""Shared reader for record-level device interpretation QC sidecars."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .contracts import ContractError, D12_LEADS, D6_LEADS


def parse_quality_mask(value: str, expected_leads: int, *, field: str) -> np.ndarray:
    """Read a pipe-delimited 1/0 lead reliability mask from the QC sidecar."""
    values = value.split("|") if value else []
    if len(values) != expected_leads or any(item not in {"0", "1"} for item in values):
        raise ContractError(f"device QC {field} must contain {expected_leads} pipe-delimited 1/0 values")
    return np.asarray([item == "1" for item in values], dtype=bool)


def load_device_interpretation_qc(path: str | Path) -> dict[str, dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "record_id", "device_type", "target_lead_mask", "input_lead_mask",
        "reliable_target_lead_count", "d12_direct_supervision_eligible",
        "d6_context_training_eligible",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise ContractError("device interpretation QC sidecar has missing required columns")
    indexed = {row["record_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ContractError("device interpretation QC sidecar has duplicate record_id")
    return indexed


def d12_target_mask(row: dict[str, str]) -> np.ndarray:
    return parse_quality_mask(row["target_lead_mask"], len(D12_LEADS), field="target_lead_mask")


def d6_input_mask(row: dict[str, str]) -> np.ndarray:
    return parse_quality_mask(row["input_lead_mask"], len(D6_LEADS), field="input_lead_mask")
