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
                row.get("input_quality_status") != "usable" or
                row.get("target_quality_status") != "usable"):
            continue
        authoritative = subject_split.get(row["subject_id"])
        if authoritative not in {"train", "validation"}:
            raise ValueError(f"Missing authoritative split for {row['subject_id']}")
        if row.get("split") and row["split"] != authoritative:
            raise ValueError(f"Pair/split disagreement for {row['pair_id']}")
        if raw_status.get(row["input_record_id"]) != "usable" or raw_status.get(row["target_record_id"]) != "usable":
            raise ValueError(f"Candidate references non-usable raw record: {row['pair_id']}")
        candidates.append({**row, "split": authoritative})
    counts = Counter(row["split"] for row in candidates)
    if len(candidates) != 104 or counts != Counter({"train": 83, "validation": 21}):
        raise ValueError(f"Expected 104 candidates (83/21), found {len(candidates)} {dict(counts)}")
    return candidates


def build(data_root: Path, repository_root: Path, output_dir: Path) -> dict[str, object]:
    metadata_dir = repository_root / "metadata"
    pairs = _validate_candidates(
        _read_csv(metadata_dir / "pair_manifest_task1.csv"),
        _read_csv(metadata_dir / "raw_record_manifest.csv"),
        _read_csv(metadata_dir / "subject_split.csv"),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, list[np.ndarray]] = {
        "train_input": [], "train_target": [], "train_context_valid": [],
        "validation_input": [], "validation_target": [], "validation_context_valid": [],
    }
    metadata: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for ordinal, pair in enumerate(pairs, start=1):
        watch_path = _resolve(data_root, pair["input_path"])
        target_path = _resolve(data_root, pair["target_path"])
        watch = parse_watch_zip(watch_path)
        input_signal, input_valid, gaps = reconstruct_watch_timeline(watch)
        target_record = parse_d12_xml(target_path, D12_LEADS)
        target_signal = _resample_d12(target_record.signal_uV,
                                      target_record.sampling_rate_hz)
        common_points = min(len(input_signal), target_signal.shape[1])
        expected_window_count = common_points // WINDOW_SAMPLES
        if expected_window_count < 1:
            raise ValueError(f"No complete window for {pair['pair_id']}")
        used_points = expected_window_count * WINDOW_SAMPLES
        input_used = input_signal[:used_points]
        valid_used = input_valid[:used_points]
        target_used = target_signal[:, :used_points]
        if not np.isfinite(input_used).all() or not np.isfinite(target_used).all():
            raise ValueError(f"Non-finite value in {pair['pair_id']}")
        # Baselines belong to physical records, not pair-specific overlap.  A
        # target or input record may be reused by more than one pair whose
        # common lengths differ, so always compute from the complete record.
        input_baseline = np.asarray([
            np.median(input_signal[input_valid]) if input_valid.any() else np.median(input_signal)
        ], dtype=np.float32)
        target_baseline = np.median(target_signal, axis=1).astype(np.float32)
        split = pair["split"]
        for window_index in range(expected_window_count):
            start = window_index * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            context_mask = valid_used[start:end]
            array_index = len(arrays[f"{split}_input"])
            arrays[f"{split}_input"].append(
                input_used[start:end][None, :].astype(np.float32, copy=False))
            arrays[f"{split}_target"].append(
                target_used[:, start:end].astype(np.float32, copy=False))
            arrays[f"{split}_context_valid"].append(context_mask.copy())
            valid_fraction = float(np.mean(context_mask))
            metadata.append({
                "window_id": f"{pair['pair_id']}_W{window_index:03d}",
                "array_index": array_index,
                "pair_id": pair["pair_id"], "subject_id": pair["subject_id"],
                "split": split, "input_record_id": pair["input_record_id"],
                "target_record_id": pair["target_record_id"],
                "window_index": window_index, "start_sample_500hz": start,
                "end_sample_500hz_exclusive": end,
                "expected_window_count": expected_window_count,
                "record_used_points_500hz": used_points,
                "context_valid_fraction": f"{valid_fraction:.9g}",
                "context_gap_samples": int((~context_mask).sum()),
                "context_quality_status": "usable" if context_mask.all() else "gap_filled",
                "input_record_baseline_uV": _baseline_text(input_baseline),
                "target_record_baseline_uV": _baseline_text(target_baseline),
                "sampling_rate_hz": TARGET_RATE_HZ, "duration_sec": 10,
                "input_shape": "1x5000", "target_shape": "12x5000",
                "unit": "μV", "lead_order": ",".join(D12_LEADS),
                "input_path": pair["input_path"], "target_path": pair["target_path"],
                "input_detected_fs_hz": f"{watch.sampling_rate_hz:.9g}",
                "target_native_fs_hz": f"{target_record.sampling_rate_hz:.9g}",
                "input_reconstructed_points": len(input_signal),
                "target_500hz_total_points": target_signal.shape[1],
                "common_points": common_points,
                "target_reuse_count": pair.get("target_reuse_count", ""),
                "pair_relation": pair.get("pair_relation", ""),
                "alignment_mode": "subject_pair_record_start",
                "quality_status": "usable",
            })
        for gap in gaps:
            gap_rows.append({"pair_id": pair["pair_id"], "subject_id": pair["subject_id"],
                             "split": split, **gap})
        summaries.append({
            "pair_id": pair["pair_id"], "subject_id": pair["subject_id"],
            "split": split, "expected_window_count": expected_window_count,
            "generated_window_count": expected_window_count,
            "watch_observed_points": len(watch.signal_uV),
            "watch_reconstructed_points": len(input_signal),
            "watch_inserted_points": int((~input_valid).sum()),
            "target_500hz_points": target_signal.shape[1],
            "record_used_points_500hz": used_points,
        })
        print(f"[{ordinal:03d}/{len(pairs):03d}] {pair['pair_id']} "
              f"{split} windows={expected_window_count} gaps={len(gaps)}")

    fields = list(metadata[0])
    with (output_dir / "task1_window_metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(metadata)
    for split in ("train", "validation"):
        np.save(output_dir / f"task1_{split}_input.npy",
                np.stack(arrays[f"{split}_input"]).astype(np.float32, copy=False))
        np.save(output_dir / f"task1_{split}_target.npy",
                np.stack(arrays[f"{split}_target"]).astype(np.float32, copy=False))
        np.save(output_dir / f"task1_{split}_context_valid_mask.npy",
                np.stack(arrays[f"{split}_context_valid"]).astype(bool, copy=False))
    with (output_dir / "task1_context_gaps.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        gap_fields = ["pair_id", "subject_id", "split", "start_sample_500hz",
                      "missing_samples", "missing_ms", "fill_method"]
        writer = csv.DictWriter(handle, fieldnames=gap_fields)
        writer.writeheader(); writer.writerows(gap_rows)
    with (output_dir / "task1_processing_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    audit = {
        "protocol": "task1-record-complete-v2",
        "subject_split_source": str(metadata_dir / "subject_split.csv"),
        "candidate_pairs": len(pairs),
        "pair_counts": dict(Counter(row["split"] for row in pairs)),
        "window_counts": dict(Counter(str(row["split"]) for row in metadata)),
        "gap_boundaries": len(gap_rows),
        "gap_affected_pairs": len({str(row["pair_id"]) for row in gap_rows}),
        "raw_data_mutated": False,
        "target_windows_deleted_for_context_gaps": 0,
        "gap_fill_policy": "physical_record_median; zero_after_record_centering; never_interpolate_ecg",
    }
    (output_dir / "task1_build_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete-record Task-1 windows")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT.parent)
    parser.add_argument("--output-dir", type=Path,
                        default=REPOSITORY_ROOT.parent / "task1_output_v2")
    args = parser.parse_args()
    audit = build(args.data_root.resolve(), REPOSITORY_ROOT, args.output_dir.resolve())
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
