"""Patient-level demographics, with conservative conflict and missing handling."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import CONDITION_SCHEMA

FIELDS = ("age", "sex", "height", "weight")
NUMERIC_FIELDS = ("age", "height", "weight")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode CSV: {path}")


def subject_key(value: Any) -> str:
    return str(value).strip().upper()


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def canonical_sex(value: Any, source: str) -> int | None:
    value = str(value).strip().lower()
    if source == "ptbxl":
        # PTB-XL sex: 0=female, 1=male, not Huawei's textual gender.
        return {"0": 1, "0.0": 1, "1": 0, "1.0": 0}.get(value)
    return {"男": 0, "male": 0, "m": 0, "女": 1, "female": 1, "f": 1}.get(value)


def encode_demographics(row: dict[str, Any] | None, source: str = "huawei") -> dict[str, np.ndarray]:
    row = row or {}
    numeric = np.zeros(3, dtype=np.float32)
    mask = np.zeros(4, dtype=bool)
    topcoded = np.zeros(1, dtype=np.float32)
    age = number(row.get("age"))
    if age is not None:
        if source == "ptbxl" and 200 <= age <= 400:
            age = 90.0
            topcoded[0] = 1.0
        if 0 <= age <= 120:
            numeric[0] = min(age, 90.0) / 100.0
            topcoded[0] = float(age >= 90)
            mask[0] = True
    sex = canonical_sex(row.get("sex", row.get("gender")), source)
    mask[1] = sex is not None
    for position, field, divisor, maximum in ((1, "height", 200., 260.), (2, "weight", 150., 500.)):
        value = number(row.get(field))
        if value is not None and 0 < value <= maximum:
            numeric[position] = value / divisor
            mask[position + 1] = True
    return {"numeric": numeric, "sex": np.asarray(2 if sex is None else sex, dtype=np.int64),
            "field_mask": mask, "age_topcoded": topcoded}


class DemographicsTable:
    """Never resolve duplicate conflicts by arbitrary row order or future timestamps."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, dict[str, Any]] = {}
        self.conflicts: dict[str, list[str]] = {}
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(path):
            key = subject_key(row.get("externalid", row.get("subject_id", "")))
            if key:
                grouped[key].append(row)
        for key, rows in grouped.items():
            canonical: dict[str, Any] = {}
            for field in FIELDS:
                values = set()
                for row in rows:
                    value = canonical_sex(row.get("gender", row.get("sex")), "huawei") if field == "sex" else number(row.get(field))
                    if value is not None:
                        values.add(value)
                if len(values) == 1:
                    value = next(iter(values))
                    canonical[field] = ("male" if value == 0 else "female") if field == "sex" else value
                elif len(values) > 1:
                    self.conflicts.setdefault(key, []).append(field)
            self.records[key] = canonical

    def get(self, subject: str) -> dict[str, np.ndarray]:
        return encode_demographics(self.records.get(subject_key(subject)), "huawei")

    def audit(self) -> dict[str, Any]:
        coverage = {field: 0 for field in FIELDS}
        for key in self.records:
            for field, valid in zip(FIELDS, self.get(key)["field_mask"]):
                coverage[field] += int(valid)
        # Audit exposes counts only; no subject IDs, gender or health values in logs.
        return {"schema": CONDITION_SCHEMA, "subjects": len(self.records), "field_coverage": coverage,
                "conflicting_subjects": len(self.conflicts),
                "conflicting_fields": sum(len(fields) for fields in self.conflicts.values())}
