"""Body-scale raw CSV reader with explicit gap positions, in canonical d6 order."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .contracts import ContractError


def parse_body_scale_record(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return 500-Hz raw μV and [C,T] validity. Nonzero initial Index is not a gap."""
    metadata = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row and row[0].strip() == "Index":
                columns = [s.strip() for s in row]
                break
            if row and ":" in row[0]:
                key, value = row[0].split(":", 1)
                metadata[key.strip()] = value.strip()
        else:
            raise ContractError("Body-scale CSV has no Index header")
        if float(metadata.get("采样率", "nan")) != 500:
            raise ContractError("Body-scale record must declare 500 Hz")
        unit = metadata.get("数据单位", "").lower()
        factors = {"v": 1e6, "mv": 1e3, "uv": 1., "μv": 1., "µv": 1.}
        if unit not in factors:
            raise ContractError("Unknown body-scale voltage unit")
        channels = [columns.index(c) for c in ("1", "2", "9", "10", "11", "12")]
        rows = [r for r in reader if r and r[0].strip()]
    if not rows:
        return np.empty((6, 0), np.float32), np.empty((6, 0), bool)
    indices = np.asarray([int(r[0]) for r in rows], dtype=np.int64)
    if np.any(np.diff(indices) <= 0):
        raise ContractError("Body-scale Index has duplicates or reversed time")
    positions = indices - indices[0]
    signal = np.full((6, int(positions[-1]) + 1), np.nan, dtype=np.float32)
    values = np.asarray([[float(r[c]) if r[c].strip() else np.nan for c in channels] for r in rows], dtype=np.float32).T
    signal[:, positions] = values * factors[unit]
    return signal, np.isfinite(signal)
