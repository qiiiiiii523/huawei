"""O16 A/B history caches on identical P predictions; B overlay is read-only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, sha
from .adapters import history_from_sample, _visible_digest
from .body_features import BODY_PROTOCOL, BEAT16_ENCODING, LEGACY_ENCODING, feature_encoding, relation_features
from .body_history import A_POLICY, B_POLICY, history_policy, process_history
from .body_prepare import load_cache, write_ledger
from .evaluate import pairs

PROTOCOL = "B5-P-body-O16-history-AB-v1"
FIXED = ("anchor", "base_ii", "target_ii", "quality")
OVERLAY_WAVE_ATOL_UV = 1e-3
OVERLAY_WAVE_RTOL = 1e-6
OVERLAY_FEATURE_ATOL = 2e-5
OVERLAY_FEATURE_RTOL = 2e-5


def numerical_comparison(actual, reference, *, atol, rtol):
    """Bound cross-platform float32 roundoff; never align, center or rescale."""
    x, y = np.asarray(actual), np.asarray(reference)
    report = {"actual_shape": list(x.shape), "reference_shape": list(y.shape),
              "atol": atol, "rtol": rtol, "exact": False, "within_tolerance": False}
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return report
    difference = np.abs(x.astype(np.float64) - y.astype(np.float64))
    allowed = atol + rtol * np.abs(y.astype(np.float64))
    report.update(exact=bool(np.array_equal(x, y)),
        within_tolerance=bool(np.all(difference <= allowed)),
        max_abs_difference=float(difference.max()) if difference.size else 0.,
        rms_difference=float(np.sqrt(np.mean(difference**2))) if difference.size else 0.,
        outside_tolerance_points=int(np.sum(difference > allowed)), total_points=int(difference.size))
    return report


def verify_overlay_history(raw, filtered, overlay, report_path, identity):
    report = {**identity, "raw_record_id": raw.record_id, "overlay_record_id": overlay.record_id,
              "record_id_equal": raw.record_id == overlay.record_id,
              "raw_overlay_mask_equal": bool(np.array_equal(raw.valid_mask, overlay.valid_mask)),
              "runtime_overlay_mask_equal": bool(np.array_equal(filtered.valid_mask, overlay.valid_mask)),
              "runtime_shape": list(np.shape(filtered.raw_uV)), "overlay_shape": list(np.shape(overlay.raw_uV))}
    structural = (report["record_id_equal"] and report["raw_overlay_mask_equal"]
                  and report["runtime_overlay_mask_equal"] and report["runtime_shape"] == report["overlay_shape"])
    report["structural_match"] = structural
    if structural and filtered.raw_uV is not None:
        mask = filtered.valid_mask
        report["waveform_uV"] = numerical_comparison(filtered.raw_uV[mask], overlay.raw_uV[mask],
                                                    atol=OVERLAY_WAVE_ATOL_UV, rtol=OVERLAY_WAVE_RTOL)
        report["per_lead_max_abs_uV"] = [
            float(np.max(np.abs(filtered.raw_uV[i, mask[i]].astype(float) - overlay.raw_uV[i, mask[i]].astype(float))))
            if mask[i].any() else 0. for i in range(6)]
    elif structural:
        report["waveform_uV"] = {"exact": True, "within_tolerance": True, "total_points": 0}
    report["waveform_passed"] = bool(structural and report["waveform_uV"]["within_tolerance"])
    write(report_path, report)
    if not report["waveform_passed"]:
        metrics = report.get("waveform_uV", {})
        raise ValueError("Runtime B filter cannot reproduce uploaded overlay within numerical tolerance; "
                         f"pair={identity['pair_id']}, subject={identity['subject_id']}; "
                         f"structural_match={structural}, max_abs_uV={metrics.get('max_abs_difference')}, "
                         f"outside_points={metrics.get('outside_tolerance_points')}; details={report_path}")
    return report


def verify_overlay_features(actual, reference, report, report_path):
    report["features"] = {name: numerical_comparison(x, y, atol=OVERLAY_FEATURE_ATOL, rtol=OVERLAY_FEATURE_RTOL)
                          for name, x, y in zip(("beat", "stats"), actual[:2], reference[:2])}
    report["availability_equal"] = bool(actual[2]) == bool(reference[2])
    report["features_passed"] = bool(report["availability_equal"] and
                                     all(r["within_tolerance"] for r in report["features"].values()))
    write(report_path, report)
    if not report["features_passed"]:
        raise ValueError("B O16 feature extraction differs beyond roundoff or availability changed; "
                         f"pair={report['pair_id']}, subject={report['subject_id']}; details={report_path}")


def check_pair(a_root, b_root):
    a_root, b_root = Path(a_root), Path(b_root)
    a, b = load_cache(a_root), load_cache(b_root)
    if (a.get("history_ab_protocol") != PROTOCOL or b.get("history_ab_protocol") != PROTOCOL
            or history_policy(a) != A_POLICY or history_policy(b) != B_POLICY
            or feature_encoding(a) != BEAT16_ENCODING or feature_encoding(b) != BEAT16_ENCODING
            or a["frozen"] != b["frozen"] or a["source_cache_sha256"] != b["source_cache_sha256"]
            or a["overlay_manifest_sha256"] != b["overlay_manifest_sha256"]):
        raise ValueError("Use paired A/B O16 caches with identical frozen P/data/QC")
    left = {(r["split"], r["pair_id"]): r for r in a["records"]}
    right = {(r["split"], r["pair_id"]): r for r in b["records"]}
    if set(left) != set(right) or len(left) != len(a["records"]) or len(right) != len(b["records"]):
        raise ValueError("A/B record population differs or has duplicates")
    excluded = {"sha256", "history_available", "processed_history_sha256"}
    changes = []
    for key, record in left.items():
        other = right[key]
        if ({k: v for k, v in record.items() if k not in excluded}
                != {k: v for k, v in other.items() if k not in excluded}):
            raise ValueError("A/B identity, raw history or quality changed")
        with np.load(a_root / record["file"], allow_pickle=False) as x, np.load(b_root / other["file"], allow_pickle=False) as y:
            if set(x.files) != set(y.files):
                raise ValueError("A/B array fields differ")
            for name in FIXED:
                if x[name].dtype != y[name].dtype or not np.array_equal(x[name], y[name]):
                    raise ValueError("A/B changed " + name)
            for data in (x, y):
                if (data["beat"].shape != (2, 300) or data["stats"].shape != (27,)
                        or not np.isfinite(data["beat"]).all() or not np.isfinite(data["stats"]).all()):
                    raise ValueError("Invalid O16 features")
            changes.append({"split": key[0], "pair_id": key[1],
                            "A_available": bool(x["available"]), "B_available": bool(y["available"]),
                            "beat_changed": not np.array_equal(x["beat"], y["beat"]),
                            "stats_changed": not np.array_equal(x["stats"], y["stats"])})
    return a, b, changes


def prepare(args):
    from baselines.B5.config import load_config
    from baselines.B5.data import common_config
    from ecg12gen.dataset import JointAnchorDataset
    source_root, output, overlay = map(lambda p: Path(p).resolve(), (args.cache, args.output, args.overlay))
    if source_root == output or source_root in output.parents or output in source_root.parents:
        raise ValueError("A/B output must be separate from the source P cache")
    source = load_cache(source_root)
    if feature_encoding(source) != LEGACY_ENCODING or "history_preprocessing" in source:
        raise ValueError("Use freshly prepared P raw-history legacy cache")
    cfg = common_config(load_config(args.config, data_only=True))
    cfg.raw["paths"]["task2_body_scale_ablation"] = str(overlay)
    cfg.raw["paths"]["task2_body_scale_b_metadata"] = str(overlay / "body_scale_context_window_metadata_B.csv")
    expected = {"protocol": BODY_PROTOCOL, "feature_encoding": BEAT16_ENCODING,
        "history_ab_protocol": PROTOCOL, "source_cache": str(source_root),
        "source_cache_sha256": sha(source_root / "cache.json"), "config_sha256": sha(args.config),
        "overlay": str(overlay), "overlay_manifest_sha256": sha(overlay / "body_scale_ablation_manifest.json"),
        "frozen": source["frozen"]}
    manifests = {}
    for name, policy in (("A", A_POLICY), ("B", B_POLICY)):
        root = output / (name + "_cache")
        ledger = root / "cache.json"
        identity = {**expected, "history_preprocessing": policy}
        if ledger.exists() and args.resume:
            manifest = read(ledger)
            if any(manifest.get(k) != v for k, v in identity.items()):
                raise ValueError("Resume changed P/cache/overlay/config")
        else:
            if root.exists() and any(root.iterdir()):
                raise FileExistsError("Use a new A/B output or --resume")
            manifest = {**identity, "completed": False, "records": []}
            write_ledger(ledger, manifest)
        manifests[name] = manifest
    originals = {(r["split"], r["pair_id"]): r for r in source["records"]}
    existing = {name: {(r["split"], r["pair_id"]): r for r in m["records"]} for name, m in manifests.items()}
    visited = set()
    overlay_checks = []
    for split in ("train", "validation"):
        ds_a = JointAnchorDataset(cfg, "task2", split)
        try:
            ds_b = JointAnchorDataset(cfg, "task2", split, body_scale_variant=B_POLICY["variant"])
            try:
                if ds_a._indices != ds_b._indices or ds_a._rows != ds_b._rows:
                    raise ValueError("B changes canonical target ordering")
                rows = [ds_a._rows[i] for i in ds_a._indices]
                for indices in pairs(rows):
                    row = rows[indices[0]]
                    if row["input_type"] != "body_scale_d6":
                        continue
                    key = split, row["pair_id"]
                    if key not in originals or key in visited:
                        raise ValueError("Live body population differs from P cache")
                    visited.add(key)
                    record = originals[key]
                    if any(row[k] != record[k] for k in ("subject_id", "target_record_id")):
                        raise ValueError("Live target identity changed")
                    samples = [ds_a[i] for i in indices]
                    raw = history_from_sample(samples[0])
                    if _visible_digest(raw) != record["history_visible_sha256"]:
                        raise ValueError("Original body history changed")
                    with np.load(source_root / record["file"], allow_pickle=False) as handle:
                        arrays = {k: handle[k].copy() for k in handle.files}
                    if (not np.array_equal(np.concatenate([s.anchor_i_ecg for s in samples], axis=-1)[0], arrays["anchor"])
                            or not np.array_equal(np.concatenate([s.Y_12lead[1] for s in samples]), arrays["target_ii"])
                            or any(not np.array_equal(s.target_quality_mask, arrays["quality"]) for s in samples)):
                        raise ValueError("Live I/target/QC differs from frozen P cache")
                    b_sample = ds_b[indices[0]]
                    for field in ("Y_12lead", "anchor_i_ecg", "target_quality_mask", "input_quality_mask"):
                        if not np.array_equal(getattr(samples[0], field), getattr(b_sample, field)):
                            raise ValueError("B altered field " + field)
                    b_overlay = history_from_sample(b_sample)
                    filtered = process_history(raw, B_POLICY)
                    report_path = output / "overlay_checks" / f"{split}_{row['pair_id']}.json"
                    check = verify_overlay_history(raw, filtered, b_overlay, report_path,
                        {"split": split, "pair_id": row["pair_id"], "subject_id": row["subject_id"]})
                    b_features = relation_features(filtered, encoding=BEAT16_ENCODING)
                    overlay_features = relation_features(b_overlay, encoding=BEAT16_ENCODING)
                    verify_overlay_features(b_features, overlay_features, check, report_path)
                    overlay_checks.append(check)
                    features = {"A": relation_features(raw, encoding=BEAT16_ENCODING), "B": b_features}
                    for name in ("A", "B"):
                        root = output / (name + "_cache")
                        if key in existing[name]:
                            if sha(root / existing[name][key]["file"]) != existing[name][key]["sha256"]:
                                raise ValueError("A/B partial cache was modified")
                            continue
                        values = {**arrays, **dict(zip(("beat", "stats", "available"), features[name]))}
                        dest = root / record["file"]
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        temp = dest.with_suffix(".pending")
                        with temp.open("wb") as h:
                            np.savez_compressed(h, **values)
                        temp.replace(dest)
                        manifests[name]["records"].append({**record, "sha256": sha(dest),
                            "history_available": bool(features[name][2]),
                            "processed_history_sha256": _visible_digest(raw if name == "A" else filtered)})
                        write_ledger(root / "cache.json", manifests[name])
                    print(f"{split}: {row['pair_id']} A/B O16 prepared, P predictions unchanged", flush=True)
            finally:
                ds_b.close()
        finally:
            ds_a.close()
    if visited != set(originals):
        raise ValueError("A/B preparation missed P cached records")
    for name, manifest in manifests.items():
        if len(manifest["records"]) != len(source["records"]):
            raise ValueError("A/B partial ledger has wrong record count")
        manifest["completed"] = True
        write_ledger(output / (name + "_cache") / "cache.json", manifest)
    try:
        a, b, changes = check_pair(output / "A_cache", output / "B_cache")
    except Exception:
        for name, manifest in manifests.items():
            manifest["completed"] = False
            write_ledger(output / (name + "_cache") / "cache.json", manifest)
        raise
    from baselines.B5_best.common import csv_write
    csv_write(output / "history_feature_changes.csv", changes)
    report = {"ready": True, "training_started": False, "P_generation_executed": False,
        "same_I_predictions_targets_QC_splits": True, "encoding": BEAT16_ENCODING,
        "records": len(changes), "availability_changes": sum(r["A_available"] != r["B_available"] for r in changes),
        "A_cache_sha256": sha(output / "A_cache/cache.json"), "B_cache_sha256": sha(output / "B_cache/cache.json"),
        "B_overlay_runtime_features_exactly_matched": all(all(r["exact"] for r in c["features"].values()) for c in overlay_checks),
        "B_overlay_runtime_features_within_tolerance": True,
        "overlay_wave_atol_uV": OVERLAY_WAVE_ATOL_UV, "overlay_wave_rtol": OVERLAY_WAVE_RTOL,
        "overlay_feature_atol": OVERLAY_FEATURE_ATOL, "overlay_feature_rtol": OVERLAY_FEATURE_RTOL,
        "overlay_nonexact_waveform_records": sum(not c["waveform_uV"]["exact"] for c in overlay_checks),
        "B_training_features_source": "runtime shared process_history(raw, B_POLICY); overlay is independently checked"}
    write(output / "prepared.json", report)
    print(report, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--overlay", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
