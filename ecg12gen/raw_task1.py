"""Raw Task-1 watch/d12 readers used by the reproducible window builder."""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np


LEAD_CODE_TO_NAME = {
    "MDC_ECG_LEAD_I": "I", "MDC_ECG_LEAD_II": "II",
    "MDC_ECG_LEAD_III": "III", "MDC_ECG_LEAD_AVR": "aVR",
    "MDC_ECG_LEAD_AVL": "aVL", "MDC_ECG_LEAD_AVF": "aVF",
    "MDC_ECG_LEAD_V1": "V1", "MDC_ECG_LEAD_V2": "V2",
    "MDC_ECG_LEAD_V3": "V3", "MDC_ECG_LEAD_V4": "V4",
    "MDC_ECG_LEAD_V5": "V5", "MDC_ECG_LEAD_V6": "V6",
}


@dataclass(frozen=True)
class WatchRecord:
    signal_uV: np.ndarray
    sampling_rate_hz: float
    frame_timestamps_ms: np.ndarray
    frame_sample_counts: np.ndarray


@dataclass(frozen=True)
class D12Record:
    signal_uV: np.ndarray
    sampling_rate_hz: float


def _unit_multiplier_to_uV(unit: str) -> float:
    normalized = unit.strip().replace("µ", "μ").lower()
    if normalized in {"μv", "uv"}:
        return 1.0
    if normalized == "mv":
        return 1_000.0
    if normalized == "v":
        return 1_000_000.0
    raise ValueError(f"Unsupported ECG unit: {unit!r}")


def parse_watch_zip(path: str | Path) -> WatchRecord:
    """Parse one JSON-lines watch archive without discarding timestamp gaps."""
    values: list[float] = []
    timestamps: list[int] = []
    frame_counts: list[int] = []
    units: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        candidates = [item for item in archive.infolist()
                      if item.filename.lower().endswith(".txt")]
        if not candidates:
            raise ValueError("No .txt ECG payload was found in the watch ZIP")
        member = max(candidates, key=lambda item: item.file_size)
        with archive.open(member) as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                    timestamps.append(int(row["timeFrame"]["timestamp"]))
                    voltage = row["voltage"]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid watch frame at line {line_number}: {error}") from error
                if not voltage:
                    raise ValueError(f"Empty voltage frame at line {line_number}")
                frame_counts.append(len(voltage))
                for item in voltage:
                    units.add(str(item["unit"]))
                    values.append(float(item["value"]))
    if len(timestamps) < 2 or not values or len(units) != 1:
        raise ValueError("Watch ECG is empty, too short, or has mixed units")
    timestamp_array = np.asarray(timestamps, dtype=np.int64)
    count_array = np.asarray(frame_counts, dtype=np.int64)
    deltas = np.diff(timestamp_array).astype(np.float64)
    positive = deltas[deltas > 0]
    if not positive.size:
        raise ValueError("Watch frame timestamps are not increasing")
    typical_delta_ms = float(np.median(positive))
    sampling_rate_hz = float(np.median(count_array[:-1])) / (typical_delta_ms / 1000.0)
    multiplier = np.float32(_unit_multiplier_to_uV(next(iter(units))))
    return WatchRecord(
        signal_uV=np.asarray(values, dtype=np.float32) * multiplier,
        sampling_rate_hz=sampling_rate_hz,
        frame_timestamps_ms=timestamp_array,
        frame_sample_counts=count_array,
    )


def reconstruct_watch_timeline(record: WatchRecord, *, target_rate_hz: int = 500,
                               gap_threshold_ms: float = 150.0,
                               ) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | float | str]]]:
    """Insert explicit samples for missing frame time instead of deleting windows.

    Observed samples are copied exactly.  Every missing position receives the
    observed-record median, which becomes zero after record-level centering.
    No ECG morphology is invented.  The returned validity vector is false for
    every inserted sample.
    """
    if not (495.0 <= record.sampling_rate_hz <= 505.0):
        raise ValueError(f"Watch sampling rate is {record.sampling_rate_hz:.3f}, not 500 Hz")
    counts = record.frame_sample_counts
    offsets = np.concatenate(([0], np.cumsum(counts)))
    deltas = np.diff(record.frame_timestamps_ms).astype(np.float64)
    typical_delta_ms = float(np.median(deltas[deltas > 0]))
    fill_value = float(np.median(record.signal_uV))
    pieces: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    gaps: list[dict[str, int | float | str]] = []
    output_offset = 0
    for frame_index, count in enumerate(counts):
        frame = record.signal_uV[offsets[frame_index]:offsets[frame_index + 1]]
        pieces.append(frame)
        masks.append(np.ones(int(count), dtype=bool))
        output_offset += int(count)
        if frame_index == len(counts) - 1:
            continue
        delta_ms = float(deltas[frame_index])
        if delta_ms <= 0:
            raise ValueError("Non-monotonic watch timestamp cannot be reconstructed safely")
        if delta_ms <= max(gap_threshold_ms, 1.5 * typical_delta_ms):
            continue
        missing_ms = delta_ms - typical_delta_ms
        missing_count = max(1, int(round(missing_ms * target_rate_hz / 1000.0)))
        inserted = np.full(missing_count, fill_value, dtype=np.float32)
        method = "record_median_fill"
        pieces.append(inserted)
        masks.append(np.zeros(missing_count, dtype=bool))
        gaps.append({
            "start_sample_500hz": output_offset,
            "missing_samples": missing_count,
            "missing_ms": missing_ms,
            "fill_method": method,
        })
        output_offset += missing_count
    return np.concatenate(pieces), np.concatenate(masks), gaps


def _namespace(root: ET.Element) -> dict[str, str]:
    return {"h": root.tag[1:].split("}", 1)[0]} if root.tag.startswith("{") else {"h": ""}


def _find(element: ET.Element, path: str, namespace: dict[str, str]) -> ET.Element | None:
    return element.find(path, namespace) if namespace["h"] else element.find(path.replace("h:", ""))


def _findall(element: ET.Element, path: str, namespace: dict[str, str]) -> list[ET.Element]:
    return element.findall(path, namespace) if namespace["h"] else element.findall(path.replace("h:", ""))


def _sequence_code(node: ET.Element, namespace: dict[str, str]) -> str:
    code = _find(node, "h:code", namespace)
    return "" if code is None else code.attrib.get("code", "")


def _digits_count(node: ET.Element, namespace: dict[str, str]) -> int:
    digits = _find(node, ".//h:digits", namespace)
    return 0 if digits is None or not digits.text else len(digits.text.split())


def _parse_lead(node: ET.Element, namespace: dict[str, str]) -> np.ndarray:
    value = _find(node, "h:value", namespace)
    if value is None:
        raise ValueError("Lead sequence has no value node")
    digits = _find(value, "h:digits", namespace)
    scale = _find(value, "h:scale", namespace)
    origin = _find(value, "h:origin", namespace)
    if digits is None or not digits.text or scale is None:
        raise ValueError("Lead sequence is missing digits or scale")
    scale_unit = scale.attrib.get("unit", "uV")
    origin_unit = origin.attrib.get("unit", scale_unit) if origin is not None else scale_unit
    if _unit_multiplier_to_uV(scale_unit) != _unit_multiplier_to_uV(origin_unit):
        raise ValueError("XML scale and origin units do not match")
    raw = np.fromstring(digits.text, sep=" ", dtype=np.float32)
    result = raw * float(scale.attrib["value"])
    if origin is not None:
        result += float(origin.attrib.get("value", "0"))
    return (result * _unit_multiplier_to_uV(scale_unit)).astype(np.float32, copy=False)


def parse_d12_xml(path: str | Path, lead_order: Iterable[str]) -> D12Record:
    """Select the longest complete 12-lead sequence set from an aECG XML."""
    root = ET.parse(path).getroot()
    namespace = _namespace(root)
    requested = list(lead_order)
    candidates: list[tuple[int, dict[str, ET.Element], float]] = []
    for sequence_set in _findall(root, ".//h:sequenceSet", namespace):
        sequences = _findall(sequence_set, "h:component/h:sequence", namespace)
        if not sequences:
            sequences = _findall(sequence_set, ".//h:sequence", namespace)
        leads: dict[str, ET.Element] = {}
        sampling_rate_hz: float | None = None
        for sequence in sequences:
            code = _sequence_code(sequence, namespace)
            if code == "TIME_ABSOLUTE":
                increment = _find(sequence, ".//h:increment", namespace)
                if increment is not None:
                    value = float(increment.attrib["value"])
                    unit = increment.attrib.get("unit", "s").lower()
                    seconds = value if unit == "s" else value / 1000.0 if unit == "ms" else None
                    if seconds and seconds > 0:
                        sampling_rate_hz = 1.0 / seconds
            lead_name = LEAD_CODE_TO_NAME.get(code)
            if lead_name:
                leads[lead_name] = sequence
        if sampling_rate_hz and all(lead in leads for lead in requested):
            minimum_length = min(_digits_count(leads[lead], namespace) for lead in requested)
            if minimum_length:
                candidates.append((minimum_length, leads, sampling_rate_hz))
    if not candidates:
        raise ValueError("No complete 12-lead sequence set was found in the XML")
    _, selected, sampling_rate_hz = max(candidates, key=lambda item: item[0])
    lead_arrays = [_parse_lead(selected[lead], namespace) for lead in requested]
    if len({len(lead) for lead in lead_arrays}) != 1:
        raise ValueError("Selected XML leads have inconsistent lengths")
    return D12Record(np.stack(lead_arrays).astype(np.float32, copy=False),
                     float(sampling_rate_hz))
