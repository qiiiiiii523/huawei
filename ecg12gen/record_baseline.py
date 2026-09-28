"""Record-level baseline indices over the existing raw-uV window caches."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from .contracts import ContractError


def build_record_baselines(
    windows: np.ndarray,
    rows: Sequence[Mapping[str, str]],
    *,
    record_id_field: str,
    source_type_field: str | None = None,
) -> dict[str | tuple[str, str], np.ndarray]:
    """Return one per-lead median vector for every physical recording.

    ``windows`` remains the existing raw-uV ``[N,C,5000]`` cache.  Duplicate
    references to the same physical record/window (caused by one recording
    participating in more than one pair) are counted once.
    """
    array = np.asarray(windows)
    if array.ndim != 3 or len(rows) != len(array):
        raise ContractError("record baseline rows must align with a [N,C,T] array")

    grouped: dict[str | tuple[str, str], list[int]] = defaultdict(list)
    seen: set[tuple[str | tuple[str, str], int]] = set()
    for row_index, row in enumerate(rows):
        record_id = row.get(record_id_field, "")
        if not record_id:
            raise ContractError(f"record baseline row is missing {record_id_field}")
        source_type = row.get(source_type_field, "") if source_type_field else ""
        if source_type_field and not source_type:
            raise ContractError(f"record baseline row is missing {source_type_field}")
        key: str | tuple[str, str] = (source_type, record_id) if source_type_field else record_id
        try:
            start = int(row["start_sample_500hz"])
        except (KeyError, ValueError) as error:
            raise ContractError("record baseline rows require integer start_sample_500hz") from error
        physical_window = (key, start)
        if physical_window in seen:
            continue
        seen.add(physical_window)
        grouped[key].append(row_index)

    baselines: dict[str | tuple[str, str], np.ndarray] = {}
    for key, indices in grouped.items():
        values = np.asarray(array[indices], dtype=np.float32)
        baselines[key] = np.median(values, axis=(0, 2)).astype(np.float32)
    return baselines


def record_baseline(raw_record: np.ndarray) -> np.ndarray:
    """Compute one median per lead from ``[C,T]`` or ``[W,C,T]`` raw uV data."""
    values = np.asarray(raw_record)
    if values.ndim == 2:
        return np.median(values, axis=1).astype(np.float32)
    if values.ndim == 3:
        return np.median(values, axis=(0, 2)).astype(np.float32)
    raise ContractError("raw record must have shape [C,T] or [W,C,T]")
