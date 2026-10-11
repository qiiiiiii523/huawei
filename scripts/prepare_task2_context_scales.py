"""Prepare Task-2 context scales using NumPy only; never train a network."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")
A = "A_raw_window"
B = "B_detrend_0p2Hz_then_window"
PROTOCOL = "task2-context-scales-v1"
SPECS = (
    ("machine_d6_scales.npz", "ecg_machine_d6", A),
    ("body_A_scales.npz", "body_scale_d6", A),
    ("body_B_scales.npz", "body_scale_d6", B),
)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def index_unique(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate metadata key: {key}")
    return result


def lead_mask(value, count):
    parts = value.split("|")
    if len(parts) != count or any(p not in {"0", "1"} for p in parts):
        raise ValueError("Invalid QC lead mask")
    return np.asarray([p == "1" for p in parts], dtype=bool)


def window_ranges(values, valid):
    """Ignore padding/gaps entirely, including nonfinite invalid placeholders."""
    if values.shape != valid.shape or values.shape[0] != 6:
        raise ValueError("Expected matching [6,T] context and mask")
    ranges = np.full(6, np.nan, dtype=np.float64)
    counts = valid.sum(axis=1).astype(np.int64)
    for lead in range(6):
        if counts[lead]:
            points = values[lead, valid[lead]].astype(np.float64)
            if not np.isfinite(points).all():
                raise ValueError("Nonfinite context marked valid")
            low, high = np.percentile(points, [5, 95])
            ranges[lead] = high - low
    return ranges, counts


def self_test():
    values = np.stack([np.arange(100) * (i + 1) for i in range(6)]).astype(float)
    padded = np.concatenate([values, np.full((6, 50), np.nan)], axis=1)
    valid = np.concatenate([np.ones_like(values, bool), np.zeros((6, 50), bool)], axis=1)
    result, counts = window_ranges(padded, valid)
    np.testing.assert_allclose(result, 89.1 * np.arange(1, 7))
    np.testing.assert_array_equal(counts, np.full(6, 100))
    empty, counts = window_ranges(padded, np.zeros_like(valid))
    assert np.isnan(empty).all() and not counts.any()
    bad = valid.copy()
    bad[:, -1] = True
    try:
        window_ranges(padded, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Valid nonfinite values were accepted")
    constant, _ = window_ranges(np.ones((6, 80)), np.ones((6, 80), bool))
    np.testing.assert_array_equal(np.maximum(constant, 25), np.full(6, 25))
    print("SELF_TEST_OK: masks, finite-value checks, percentiles, and scale floor")


def eligible_contexts(repo, data):
    subjects = index_unique(read_csv(repo / "metadata/subject_split.csv"), "subject_id")
    qc = index_unique(read_csv(repo / "metadata/device_interpretation_qc.csv"), "record_id")
    pairs = index_unique(read_csv(repo / "metadata/pair_manifest_task2.csv"), "pair_id")
    grouped = defaultdict(list)
    for r in read_csv(data / "task2_window_metadata.csv"):
        if r["split"] == "train":
            grouped[r["pair_id"]].append(r)
    selected = {}
    pair_counts, disabled = Counter(), Counter()
    for pair_id, rows in grouped.items():
        r, p = rows[0], pairs[pair_id]
        source = r["input_type"]
        if source not in {"ecg_machine_d6", "body_scale_d6"}:
            raise ValueError("Unexpected Task-2 device")
        if len(rows) != 12 or sorted(int(x["start_sample_500hz"]) for x in rows) != list(range(0, 60000, 5000)):
            raise ValueError("Incomplete training target pair")
        for x in rows:
            if int(x["expected_window_count"]) != 12:
                raise ValueError("Wrong target window count")
            if any(x[k] != r[k] for k in ("subject_id", "input_type", "input_record_id", "target_record_id", "split")):
                raise ValueError("Mixed identities in target pair")
        if subjects[r["subject_id"]]["split"] != "train" or p["split"] != "train":
            raise ValueError("Patient split leakage")
        if any(p[k] != r[k] for k in ("subject_id", "input_type", "input_record_id", "target_record_id")):
            raise ValueError("Pair manifest disagreement")
        target_qc = qc[r["target_record_id"]]
        target_mask = lead_mask(target_qc["target_lead_mask"], 12)
        if target_qc["device_type"] != "ecg_machine_d12":
            raise ValueError("Target QC device mismatch")
        if p["pair_status"] != "paired" or p["target_quality_status"] != "usable" or any(x["quality_status"] != "usable" for x in rows):
            disabled[source + ":target_structure"] += 1
            continue
        if target_qc["d12_direct_supervision_eligible"] != "true" or not target_mask[0]:
            disabled[source + ":target_qc"] += 1
            continue
        pair_counts[source] += 1
        if p["input_quality_status"] != "usable":
            disabled[source + ":context_structure"] += 1
            continue
        mask = np.ones(6, bool)
        if source == "ecg_machine_d6":
            input_qc = qc[r["input_record_id"]]
            if input_qc["device_type"] != "ecg_machine_d6":
                raise ValueError("Machine context QC mismatch")
            if input_qc["d6_context_training_eligible"] != "true":
                disabled[source + ":context_qc"] += 1
                continue
            mask &= lead_mask(input_qc["input_lead_mask"], 6)
        key = source, r["input_record_id"]
        value = r["subject_id"], mask
        if key in selected and (selected[key][0] != value[0] or not np.array_equal(selected[key][1], mask)):
            raise ValueError("One history record has inconsistent patient/QC")
        selected[key] = value
    return selected, subjects, dict(pair_counts), dict(disabled)


def verify_overlay(repo, data, body):
    path = body / "body_scale_ablation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["protocol"] != "independent-body-context-overlay-v1" or manifest["variant"] != B:
        raise ValueError("Wrong body B cache protocol")
    if manifest["filter"]["protocol"] != "body-context-b-hp0p2-order2-sosfiltfilt-valid-runs-v1":
        raise ValueError("Wrong body B filter protocol")
    protected = {
        "target_metadata": data / "task2_window_metadata.csv",
        "context_metadata": data / "task2_context_window_metadata.csv",
        "pair_manifest": repo / "metadata/pair_manifest_task2.csv",
        "subject_split": repo / "metadata/subject_split.csv",
        "device_qc": repo / "metadata/device_interpretation_qc.csv",
    }
    for name, filename in protected.items():
        if sha256(filename) != manifest["canonical_metadata_sha256"][name]:
            raise ValueError(f"Stale body B overlay: {name}")
    for name in ("body_scale_context_window_metadata_B.csv", "body_scale_train_context_B.npy", "body_scale_train_context_valid_mask_B.npy"):
        if sha256(body / name) != manifest["overlay_files_sha256"][name]:
            raise ValueError(f"Body B file hash mismatch: {name}")
    return manifest


def fit_artifact(source, variant, values, masks, rows, selected, subjects, minimum):
    if values.ndim != 3 or values.shape[1:] != (6, 5000) or masks.shape != values.shape or masks.dtype != bool:
        raise ValueError("Invalid train context arrays")
    if [int(r["array_index"]) for r in rows] != list(range(len(values))):
        raise ValueError("Train metadata and arrays disagree")
    ranges, point_counts, used, records = [], [], set(), set()
    skipped = 0
    grouped = defaultdict(list)
    for r in rows:
        if r["split"] != "train" or subjects[r["subject_id"]]["split"] != "train":
            raise ValueError("Context crosses patient split")
        if r["input_type"] != source:
            continue
        key = source, r["input_record_id"]
        if key not in selected:
            skipped += 1
            continue
        subject, lead_quality = selected[key]
        if subject != r["subject_id"]:
            raise ValueError("Context patient mismatch")
        identity = key + (int(r["start_sample_500hz"]),)
        if identity in used:
            raise ValueError("Duplicate physical context window")
        used.add(identity)
        grouped[key].append(r)
        index, length = int(r["array_index"]), int(r["valid_length"])
        if not 1 <= length <= 5000 or masks[index, :, length:].any():
            raise ValueError("Padding marked valid")
        valid = np.asarray(masks[index]) & lead_quality[:, None]
        if r["context_window_valid"] != "true":
            valid[:] = False
        spread, points = window_ranges(values[index], valid)
        if np.isfinite(spread).any():
            ranges.append(spread)
            point_counts.append(points)
            records.add(key)
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: int(r["start_sample_500hz"]))
        if len(ordered) != int(ordered[0]["expected_context_window_count"]) or [int(r["start_sample_500hz"]) for r in ordered] != list(range(0, len(ordered) * 5000, 5000)):
            raise ValueError("Incomplete context record")
    if not ranges:
        raise ValueError("No eligible context windows")
    stacked = np.stack(ranges)
    window_counts = np.isfinite(stacked).sum(axis=0)
    if (window_counts == 0).any():
        raise ValueError("A context lead has no valid training data")
    scale = np.maximum(np.nanmedian(stacked, axis=0), minimum).astype(np.float32)
    if not np.isfinite(scale).all() or (scale < minimum).any():
        raise ValueError("Invalid fitted scale")
    report = {
        "source_type": source, "variant": variant, "lead_order": list(LEADS),
        "records_used": len(records), "windows_used": len(ranges),
        "windows_per_lead": window_counts.tolist(),
        "valid_points_per_lead": np.stack(point_counts).sum(axis=0).tolist(),
        "excluded_device_windows": skipped, "scale_uV": scale.tolist(),
        "dedup": "one physical history record/window per source; no repeated target-pair weighting",
    }
    return scale, report


def with_context_scales(base_preprocessor, path, source_type, variant=A):
    """Attach one context artifact without replacing d12/anchor scales."""
    from ecg12gen.preprocessing import ECGPreprocessor
    with np.load(path, allow_pickle=False) as a:
        if str(a["protocol"].item()) != PROTOCOL or str(a["source_type"].item()) != source_type or str(a["variant"].item()) != variant:
            raise ValueError("Context scale protocol/source/variant mismatch")
        if tuple(a["lead_order"].tolist()) != LEADS:
            raise ValueError("Context scale lead order mismatch")
        scale = a["scale_uV"].astype(np.float32).copy()
    if scale.shape != (6,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid context scale")
    scales = {k: v.copy() for k, v in base_preprocessor.scale_uV_by_source.items()}
    scales[source_type] = scale
    return ECGPreprocessor(base_preprocessor.config, scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--body-b-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    repo = args.repo.resolve()
    data = (args.data_dir or repo.parent / "task2_record_context_v3").resolve()
    body = (args.body_b_dir or repo.parent / "task2_body_scale_ablation").resolve()
    output = (args.output_dir or repo / "results/B5/context_params").resolve()
    names = [s[0] for s in SPECS] + ["context_scales_report.json", "README.md"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Output already contains artifacts; use a new directory, do not overwrite frozen scales")
    preprocessing = (repo / "configs/preprocessing.yaml").read_text(encoding="utf-8-sig")
    if 'method: train_median_p5_p95_range' not in preprocessing or 'minimum_scale_uV: 25.0' not in preprocessing:
        raise ValueError("This script requires the current fixed 25-uV robust-range protocol")
    selected, subjects, pairs, disabled = eligible_contexts(repo, data)
    overlay = verify_overlay(repo, data, body)
    canonical = [r for r in read_csv(data / "task2_context_window_metadata.csv") if r["split"] == "train"]
    canonical.sort(key=lambda r: int(r["array_index"]))
    body_rows = [r for r in read_csv(body / "body_scale_context_window_metadata_B.csv") if r["split"] == "train"]
    body_rows.sort(key=lambda r: int(r["array_index"]))
    fields = ("subject_id", "split", "input_type", "input_record_id", "context_window_index", "start_sample_500hz", "valid_length", "valid_sample_count", "context_window_valid", "expected_context_window_count")
    canonical_body = [r for r in canonical if r["input_type"] == "body_scale_d6"]
    if len(canonical_body) != len(body_rows):
        raise ValueError("Body B record coverage mismatch")
    for r in body_rows:
        original = canonical[int(r["canonical_array_index"])]
        if any(r[k] != original[k] for k in fields):
            raise ValueError("Body B identity/mask metadata mismatch")
    raw = np.load(data / "task2_train_context.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(data / "task2_train_context_valid_mask.npy", mmap_mode="r", allow_pickle=False)
    b_raw = np.load(body / "body_scale_train_context_B.npy", mmap_mode="r", allow_pickle=False)
    b_mask = np.load(body / "body_scale_train_context_valid_mask_B.npy", mmap_mode="r", allow_pickle=False)
    for r in body_rows:
        if not np.array_equal(b_mask[int(r["array_index"])], mask[int(r["canonical_array_index"])]):
            raise ValueError("Body B validity mask differs from canonical A")
    results = {}
    artifacts = []
    for filename, source, variant in SPECS:
        is_b = variant == B
        scale, report = fit_artifact(source, variant, b_raw if is_b else raw, b_mask if is_b else mask, body_rows if is_b else canonical, selected, subjects, 25.0)
        results[filename] = report
        artifacts.append((filename, source, variant, scale))
    input_paths = {
        "subject_split": repo / "metadata/subject_split.csv",
        "pairs": repo / "metadata/pair_manifest_task2.csv",
        "qc": repo / "metadata/device_interpretation_qc.csv",
        "target_metadata": data / "task2_window_metadata.csv",
        "context_metadata": data / "task2_context_window_metadata.csv",
        "train_context": data / "task2_train_context.npy",
        "train_mask": data / "task2_train_context_valid_mask.npy",
        "body_manifest": body / "body_scale_ablation_manifest.json",
        "body_B_metadata": body / "body_scale_context_window_metadata_B.csv",
        "body_B_train_context": body / "body_scale_train_context_B.npy",
        "body_B_train_mask": body / "body_scale_train_context_valid_mask_B.npy",
        "preprocessing_config": repo / "configs/preprocessing.yaml",
    }
    report = {
        "protocol": PROTOCOL, "fit_split": "train", "minimum_scale_uV": 25.0,
        "formula": "per-window valid-point P95-P5, median across eligible unique windows per lead",
        "training_started": False, "validation_waveforms_loaded": False,
        "target_waveforms_loaded": False, "d12_anchor_scales_modified": False,
        "eligible_target_pairs": pairs, "excluded_pairs_or_context": disabled,
        "body_B_filter": overlay["filter"],
        "input_sha256": {k: sha256(p) for k, p in input_paths.items()},
        "artifacts": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, source, variant, scale in artifacts:
        with (output / filename).open("xb") as f:
            np.savez(f, protocol=np.asarray(PROTOCOL), source_type=np.asarray(source), variant=np.asarray(variant), lead_order=np.asarray(LEADS), scale_uV=scale)
        with np.load(output / filename, allow_pickle=False) as a:
            np.testing.assert_array_equal(a["scale_uV"], scale)
        results[filename]["sha256"] = sha256(output / filename)
    (output / "context_scales_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Task 2 context scales\n\n"
        "Prepared from eligible training context only. No neural-network training is required.\n\n"
        "- machine_d6_scales.npz: ecg_machine_d6 / A_raw_window\n"
        "- body_A_scales.npz: body_scale_d6 / A_raw_window\n"
        "- body_B_scales.npz: body_scale_d6 / B_detrend_0p2Hz_then_window\n\n"
        "These are context-only artifacts, not replacements for the original B5 preprocessing_scales.npz.\n"
        "Load the checkpoint's original preprocessor, then attach the matching file with "
        "scripts.prepare_task2_context_scales.with_context_scales(base, path, source_type, variant).\n"
        "The helper preserves d12 and ecg_machine_i values. Context training must audit the base "
        "checkpoint scales separately from these additional scales.\n"
        "Send this whole directory, including context_scales_report.json, to teammates.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "eligible_target_pairs": pairs, "artifacts": results}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
