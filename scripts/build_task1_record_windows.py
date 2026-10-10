"""Build record-complete Task-1 windows without mutating the raw data.

The authoritative subject split is preserved verbatim.  A watch timestamp gap
changes only the context signal and its validity metadata; it never deletes the
corresponding d12 target/anchor window.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ecg12gen.contracts import D12_LEADS
from ecg12gen.record_cache import IndependentRecordCacheBuilder
from ecg12gen.raw_task1 import parse_d12_xml, parse_watch_zip, reconstruct_watch_timeline

WINDOW_SAMPLES = 5000
TARGET_RATE_HZ = 500


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(data_root: Path, stored_path: str) -> Path:
    return data_root / Path(stored_path.replace("\\", "/"))


def _resample_d12(signal: np.ndarray, source_rate_hz: float) -> np.ndarray:
    if abs(source_rate_hz - TARGET_RATE_HZ) < 0.5:
        return signal.astype(np.float32, copy=False)
    rounded = int(round(source_rate_hz))
    if abs(source_rate_hz - rounded) > 0.5:
        raise ValueError(f"Unexpected non-integer d12 rate: {source_rate_hz}")
    divisor = gcd(rounded, TARGET_RATE_HZ)
    return resample_poly(signal, TARGET_RATE_HZ // divisor, rounded // divisor,
                         axis=1).astype(np.float32, copy=False)


def _baseline_text(values: np.ndarray) -> str:
    return "|".join(f"{float(value):.9g}" for value in values)


def _validate_candidates(pair_rows: list[dict[str, str]], raw_rows: list[dict[str, str]],
                         split_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    subject_split = {row["subject_id"]: row["split"] for row in split_rows}
    if len(subject_split) != len(split_rows):
        raise ValueError("subject_split.csv contains duplicate subjects")
    raw_status = {row["record_id"]: row["quality_status"] for row in raw_rows}
    candidates: list[dict[str, str]] = []
    for row in pair_rows:
        if (row.get("pair_status") != "paired" or
                row.get("target_quality_status") != "usable"):
            continue
        authoritative = subject_split.get(row["subject_id"])
        if authoritative not in {"train", "validation"}:
            raise ValueError(f"Missing authoritative split for {row['subject_id']}")
        if row.get("split") and row["split"] != authoritative:
            raise ValueError(f"Pair/split disagreement for {row['pair_id']}")
        if raw_status.get(row["target_record_id"]) != "usable":
            raise ValueError(f"Candidate references non-usable raw record: {row['pair_id']}")
        candidates.append({**row, "split": authoritative})
    if not candidates:
        raise ValueError("No eligible target pairs")
    return candidates


def build(data_root: Path, repository_root: Path, output_dir: Path) -> dict[str, object]:
    """Build independent context/target indices; never truncate targets to watch length."""
    metadata_dir = repository_root / "metadata"
    pairs = _validate_candidates(
        _read_csv(metadata_dir / "pair_manifest_task1.csv"),
        _read_csv(metadata_dir / "raw_record_manifest.csv"),
        _read_csv(metadata_dir / "subject_split.csv"),
    )
    writer = IndependentRecordCacheBuilder("task1")
    summaries, gap_rows, rejected = [], [], []
    for pair in pairs:
        target_record = parse_d12_xml(_resolve(data_root, pair["target_path"]), D12_LEADS)
        target = _resample_d12(target_record.signal_uV, target_record.sampling_rate_hz)
        if target.shape[1] < 60000:
            rejected.append({"pair_id": pair["pair_id"], "reason": "target_under_120_seconds",
                             "target_points": target.shape[1]})
            continue
        context, valid, gaps = None, None, []
        if pair.get("input_quality_status") == "usable":
            watch = parse_watch_zip(_resolve(data_root, pair["input_path"]))
            context, valid, gaps = reconstruct_watch_timeline(watch)
            context = context[None, :]
        writer.add_pair({**pair, "input_type": "watch_ecg"}, target, context, valid)
        context_points = 0 if context is None else context.shape[1]
        summaries.append({"pair_id": pair["pair_id"], "split": pair["split"],
                          "target_500hz_points": target.shape[1], "target_used_points": 60000,
                          "target_window_count": 12, "context_points": context_points,
                          "context_window_count": (context_points + 4999) // 5000,
                          "context_tail_samples": context_points % 5000})
        gap_rows.extend({"pair_id": pair["pair_id"], "split": pair["split"], **gap} for gap in gaps)
    if not writer.target_rows:
        raise ValueError("No targets have a real 120-second interval")
    audit = writer.write(output_dir)
    audit.update({"candidate_pairs": len(pairs), "rejected_short_targets": rejected,
                  "pair_counts": dict(Counter(r["split"] for r in summaries)),
                  "subject_split_source": str(metadata_dir / "subject_split.csv"),
                  "target_windows_deleted_for_context_gaps": 0,
                  "gap_fill_policy": "valid_record_median_then_zero_with_mask",
                  "gap_boundaries": len(gap_rows)})
    for name, rows in (("task1_processing_summary.csv", summaries), ("task1_context_gaps.csv", gap_rows)):
        if rows:
            with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                csv_writer.writeheader()
                csv_writer.writerows(rows)
    (output_dir / "task1_build_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 120-second targets and independent full watch contexts")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT.parent)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT.parent / "task1_record_context_v3")
    args = parser.parse_args()
    print(json.dumps(build(args.data_root.resolve(), REPOSITORY_ROOT, args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
