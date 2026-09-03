"""Build task1 train-only R-peak pseudo d12 windows; never overwrites task1_output."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.dataset import ECGDataConfig
from ecg12gen.rpeak_pseudopair import beats, centered_for_detection, detect_xqrs, monotonic_match, target_time_mapping, warp_d12_to_input_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "common.yaml"))
    parser.add_argument("--rpeak-config", default=str(ROOT / "configs" / "rpeak_pseudopair.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = ECGDataConfig.from_yaml(args.config)
    r_cfg = yaml.safe_load(Path(args.rpeak_config).read_text(encoding="utf-8"))
    task_dir = cfg.path("task1_output")
    out = Path(args.output_dir) if args.output_dir else (ROOT / r_cfg["output"]["derived_dir"])
    out = out.resolve()
    names = ["task1_rpeak_train_input.npy", "task1_rpeak_train_target.npy", "task1_rpeak_train_window_metadata.csv"]
    if any((out / name).exists() for name in names) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite pseudo output: {out}; pass --overwrite for this derived directory only")
    inputs = np.load(task_dir / "task1_train_input.npy", mmap_mode="r")
    targets = np.load(task_dir / "task1_train_target.npy", mmap_mode="r")
    with (task_dir / "task1_window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted((row for row in csv.DictReader(handle) if row["split"] == "train" and row["quality_status"] == "usable"), key=lambda row: int(row["array_index"]))
    if len(rows) != len(inputs):
        raise ValueError("Task1 train metadata and input array disagree")
    candidates = []
    for row in rows:
        index = int(row["array_index"])
        watch = centered_for_detection(inputs[index, 0])
        d12_i = centered_for_detection(targets[index, 0])
        input_peaks, target_peaks = detect_xqrs(watch, r_cfg["sampling_rate_hz"]), detect_xqrs(d12_i, r_cfg["sampling_rate_hz"])
        if len(input_peaks) < r_cfg["min_detected_peaks"] or len(target_peaks) < r_cfg["min_detected_peaks"] or abs(len(input_peaks) - len(target_peaks)) > r_cfg["max_peak_count_difference"]:
            continue
        ib, tb = beats(watch, input_peaks, r_cfg["beat_pre_samples"], r_cfg["beat_post_samples"]), beats(d12_i, target_peaks, r_cfg["beat_pre_samples"], r_cfg["beat_post_samples"])
        matched = monotonic_match(ib, tb)
        ratio = len(matched) / max(len(ib), len(tb), 1)
        if len(matched) < r_cfg["min_matched_beats"] or ratio < r_cfg["min_match_ratio"]:
            continue
        mapping = target_time_mapping(matched, ib, tb, r_cfg["window_samples"])
        if mapping is None:
            continue
        warp_ratios = np.diff(np.asarray([tb[j].peak for _, j, _ in matched])) / np.diff(np.asarray([ib[i].peak for i, _, _ in matched]))
        if warp_ratios.size and (warp_ratios.min() < r_cfg["min_warp_ratio"] or warp_ratios.max() > r_cfg["max_warp_ratio"]):
            continue
        similarity = float(np.mean([score for _, _, score in matched]))
        candidates.append((row, index, ib, tb, matched, mapping, similarity, ratio))
    threshold = float(np.quantile([item[6] for item in candidates], r_cfg["accept_similarity_quantile"])) if candidates else float("inf")
    accepted = [item for item in candidates if item[6] >= threshold]
    accepted_indices = {item[1] for item in accepted}
    try:
        import wfdb  # type: ignore
        detector_version = getattr(wfdb, "__version__", "unknown")
    except ImportError as error:
        raise RuntimeError("wfdb is required to build R-peak pseudo-pairs") from error
    out.mkdir(parents=True, exist_ok=True)
    pseudo_inputs, pseudo_targets, metadata, manifest = [], [], [], []
    for row, index, ib, tb, matched, mapping, similarity, ratio in candidates:
        is_accepted = index in accepted_indices
        quality = max(0.0, min(1.0, (similarity + 1.0) / 2.0 * ratio))
        for i, j, beat_similarity in matched:
            manifest.append({"input_window_id": row["window_id"], "target_window_id": row["window_id"], "input_r_peak": ib[i].peak, "target_r_peak": tb[j].peak, "input_beat_start": ib[i].start, "input_beat_end": ib[i].end, "target_beat_start": tb[j].start, "target_beat_end": tb[j].end, "input_rr": ib[i].rr, "target_rr": tb[j].rr, "morphology_similarity": beat_similarity, "qrs_width_difference": abs(ib[i].qrs_width - tb[j].qrs_width), "time_warp_ratio": "", "alignment_quality_score": quality, "accepted": str(is_accepted).lower(), "loss_weight": quality if is_accepted else 0.0, "detector_name": "wfdb_xqrs", "detector_version": detector_version})
    for output_index, (row, index, ib, tb, matched, mapping, similarity, ratio) in enumerate(accepted):
        pseudo_inputs.append(np.asarray(inputs[index], dtype=np.float32))
        pseudo_targets.append(warp_d12_to_input_grid(np.asarray(targets[index], dtype=np.float32), mapping))
        quality = max(0.0, min(1.0, (similarity + 1.0) / 2.0 * ratio))
        metadata.append({"array_index": output_index, "source_window_id": row["window_id"], "subject_id": row["subject_id"], "split": "train", "alignment_mode": "pseudo_rpeak_monotonic_warp", "physical_sync": "false", "alignment_quality_score": quality, "accepted": "true"})
    input_array = np.stack(pseudo_inputs).astype(np.float32) if pseudo_inputs else np.empty((0, 1, 5000), dtype=np.float32)
    target_array = np.stack(pseudo_targets).astype(np.float32) if pseudo_targets else np.empty((0, 12, 5000), dtype=np.float32)
    np.save(out / names[0], input_array)
    np.save(out / names[1], target_array)
    for path, fields, values in ((out / names[2], list(metadata[0]) if metadata else ["array_index"], metadata), (ROOT / r_cfg["output"]["manifest"], list(manifest[0]) if manifest else ["input_window_id"], manifest)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    print(f"Accepted {len(accepted)} train-only pseudo windows; wrote {out}")


if __name__ == "__main__":
    main()