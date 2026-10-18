"""Re-encode visible body history; copy frozen I/prediction/target/QC unchanged."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, sha, csv_write
from .body_features import (BODY_PROTOCOL, LEGACY_ENCODING, SHAPE_ENCODING, UNCLIPPED_ENCODING, BEAT16_ENCODING,
                            feature_encoding, relation_features)
from .body_prepare import load_cache, write_ledger
from .adapters import history_from_sample, _visible_digest
from .evaluate import pairs

ENCODING_STUDY = "B5-best-body-feature-pair-v1"
FIXED_ARRAYS = ("anchor", "base_ii", "target_ii", "quality", "available")


def finish_report(root, output, manifest):
    """Also repair a stop after the atomic ledger commit but before report export."""
    check_cached_pair(root, output, expected_encoding=feature_encoding(manifest))
    audits = [read(p) for p in sorted((output / "audit").glob("*.json"))]
    if len(audits) != 2 * len(manifest["records"]):
        raise ValueError("Prepared encoding cache is missing its paired clipping audit")
    csv_write(output / "clipping_comparison.csv", audits)
    report = {"completed": True, "training_started": False, "B5_inference_executed": False,
        "same_I_predictions_targets_QC_population": True, "same_feature_dimensions": True,
        "source_cache_sha256": manifest["source_cache_sha256"],
        "feature_encoding": feature_encoding(manifest), "records": len(manifest["records"])}
    write(output / "preflight.json", report)
    print(report, flush=True)


def check_cached_pair(old_root, new_root, *, expected_encoding=SHAPE_ENCODING):
    """Verify the experiment differs ONLY in historical beat/stat features."""
    old_root, new_root = Path(old_root), Path(new_root)
    old, new = load_cache(old_root), load_cache(new_root)
    if (expected_encoding not in (SHAPE_ENCODING, UNCLIPPED_ENCODING, BEAT16_ENCODING)
            or feature_encoding(old) != LEGACY_ENCODING or feature_encoding(new) != expected_encoding):
        raise ValueError("Expected legacy control and shape/amplitude experimental cache")
    if (old["frozen"] != new["frozen"] or new.get("source_cache_sha256") != sha(old_root / "cache.json")
            or new.get("encoding_study") != ENCODING_STUDY):
        raise ValueError("Encoding caches do not share frozen B5/data/QC/source")
    a = {(r["split"], r["pair_id"]): r for r in old["records"]}
    b = {(r["split"], r["pair_id"]): r for r in new["records"]}
    if len(a) != len(old["records"]) or set(a) != set(b) or len(b) != len(new["records"]):
        raise ValueError("Encoding pair population changed or has duplicate pairs")
    for key, row in a.items():
        if {k: v for k, v in row.items() if k != "sha256"} != {k: v for k, v in b[key].items() if k != "sha256"}:
            raise ValueError("Encoding cache changed record metadata/QC/history: " + str(key))
        with np.load(old_root / row["file"], allow_pickle=False) as left, np.load(new_root / b[key]["file"], allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                raise ValueError("Encoding cache changed array fields")
            for name in FIXED_ARRAYS:
                if left[name].dtype != right[name].dtype or not np.array_equal(left[name], right[name]):
                    raise ValueError("Encoding experiment changed " + name)
            for name, shape in (("beat", (2, 300)), ("stats", (27,))):
                if right[name].shape != shape or not np.isfinite(right[name]).all():
                    raise ValueError("Invalid encoded historical feature: " + name)
            if expected_encoding == BEAT16_ENCODING and not np.array_equal(left["stats"], right["stats"]):
                raise ValueError("O16 must retain the original stats8 values exactly")
    return old, new


def check_unclipped_pair(clipped_root, unclipped_root):
    """E0 and E differ only in historical templates, never in the 27 statistics."""
    clipped_root, unclipped_root = Path(clipped_root), Path(unclipped_root)
    clipped, unclipped = load_cache(clipped_root), load_cache(unclipped_root)
    if (feature_encoding(clipped) != SHAPE_ENCODING or feature_encoding(unclipped) != UNCLIPPED_ENCODING
            or clipped["frozen"] != unclipped["frozen"]
            or clipped.get("source_cache_sha256") != unclipped.get("source_cache_sha256")
            or clipped.get("config_sha256") != unclipped.get("config_sha256")):
        raise ValueError("E/E0 must share source data/config and shape/amplitude statistics")
    left = {(r["split"], r["pair_id"]): r for r in clipped["records"]}
    right = {(r["split"], r["pair_id"]): r for r in unclipped["records"]}
    if set(left) != set(right):
        raise ValueError("E/E0 populations differ")
    for key, row in left.items():
        with np.load(clipped_root / row["file"], allow_pickle=False) as a, np.load(unclipped_root / right[key]["file"], allow_pickle=False) as b:
            for name in (*FIXED_ARRAYS, "stats"):
                if a[name].dtype != b[name].dtype or not np.array_equal(a[name], b[name]):
                    raise ValueError("E0 changed more than historical template clipping: " + name)


def run(args):
    from baselines.B5.config import load_config
    from baselines.B5.data import common_config
    from baselines.B5.metadata import subject_key
    from ecg12gen.dataset import JointAnchorDataset
    root, output = Path(args.cache).resolve(), Path(args.output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("New feature cache must be separate from the source cache")
    source = load_cache(root)
    encoding = getattr(args, "encoding", SHAPE_ENCODING)
    if encoding not in (SHAPE_ENCODING, UNCLIPPED_ENCODING, BEAT16_ENCODING):
        raise ValueError("Prepare E, E0 or O16 from the same original legacy cache")
    if feature_encoding(source) != LEGACY_ENCODING:
        raise ValueError("Use the original body_v1/cache as the legacy control")
    expected = {"protocol": BODY_PROTOCOL, "feature_encoding": encoding,
        "encoding_study": ENCODING_STUDY, "source_cache": str(root),
        "source_cache_sha256": sha(root / "cache.json"), "config_sha256": sha(args.config),
        "frozen": source["frozen"]}
    ledger = output / "cache.json"
    if ledger.exists() and args.resume:
        manifest = read(ledger)
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume changed source data/config/encoding")
        if manifest.get("completed"):
            finish_report(root, output, manifest)
            print({"completed": True, "reused": True, "B5_inference_executed": False}, flush=True)
            return
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Use an empty --output or explicit --resume; keep tee logs adjacent")
        manifest = {**expected, "completed": False, "records": []}
        output.mkdir(parents=True, exist_ok=True)
        write_ledger(ledger, manifest)
    existing = {(r["split"], r["pair_id"]): r for r in manifest["records"]}
    lookup = {(r["split"], r["pair_id"]): r for r in source["records"]}
    cfg = common_config(load_config(args.config, data_only=True))
    visited = set()
    for split in ("train", "validation"):
        dataset = JointAnchorDataset(cfg, "task2", split)
        try:
            rows = [dataset._rows[i] for i in dataset._indices]
            for indices in pairs(rows):
                first = rows[indices[0]]
                if first["input_type"] != "body_scale_d6":
                    continue
                key = split, first["pair_id"]
                if key not in lookup or key in visited:
                    raise ValueError("Live data and source cache pair populations differ")
                visited.add(key)
                record = lookup[key]
                if (subject_key(first["subject_id"]) != record["subject_id"]
                        or first["target_record_id"] != record["target_record_id"]):
                    raise ValueError("Source subject/target mapping changed")
                if key in existing:
                    if sha(output / existing[key]["file"]) != existing[key]["sha256"]:
                        raise ValueError("Partial encoded cache changed")
                    continue
                samples = [dataset[i] for i in indices]
                history = history_from_sample(samples[0])
                if _visible_digest(history) != record["history_visible_sha256"]:
                    raise ValueError("Historical body signal/mask changed since original preparation")
                with np.load(root / record["file"], allow_pickle=False) as handle:
                    arrays = {k: handle[k].copy() for k in handle.files}
                anchor = np.concatenate([s.anchor_i_ecg for s in samples], axis=-1)[0]
                target = np.concatenate([s.Y_12lead[1] for s in samples])
                if (not np.array_equal(anchor, arrays["anchor"]) or not np.array_equal(target, arrays["target_ii"])
                        or any(not np.array_equal(s.target_quality_mask, arrays["quality"]) for s in samples)):
                    raise ValueError("Live I/target/QC differ from frozen source cache")
                old_audit, new_audit = {}, {}
                old_features = relation_features(history, audit=old_audit)
                new_features = relation_features(history, audit=new_audit, encoding=encoding)
                for name, value in zip(("beat", "stats", "available"), old_features):
                    if not np.array_equal(arrays[name], value):
                        raise ValueError("Legacy feature extraction failed to reproduce original cache")
                if new_features[2] != bool(arrays["available"]):
                    raise ValueError("Encoding must not change history availability/QC")
                new_label = {SHAPE_ENCODING: "E", UNCLIPPED_ENCODING: "E0", BEAT16_ENCODING: "O16"}[encoding]
                for label, audit in (("O", old_audit), (new_label, new_audit)):
                    uncut, clipped = audit.pop("_unclipped_template", None), audit.pop("_clipped_template", None)
                    write(output / "audit" / f"{split}_{record['pair_id']}_{label}.json",
                        {"split": split, "pair_id": record["pair_id"], "subject_id": record["subject_id"],
                         "encoding_group": label, **audit})
                    if split == "validation" and record["subject_id"] in args.subjects and uncut is not None:
                        csv_write(output / "templates" / f"{record['pair_id']}_{label}.csv", [
                            {"time_s": (i - 100) / 500, "I_before_clip": float(uncut[0, i]),
                             "II_before_clip": float(uncut[1, i]), "I_after_clip": float(clipped[0, i]),
                             "II_after_clip": float(clipped[1, i])} for i in range(300)])
                arrays["beat"], arrays["stats"] = new_features[:2]
                dest = output / record["file"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_suffix(".pending")
                with temp.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                temp.replace(dest)
                manifest["records"].append({**record, "sha256": sha(dest)})
                write_ledger(ledger, manifest)
                print(f"{split}: {record['pair_id']} re-encoded; no B5 generation", flush=True)
        finally:
            dataset.close()
    if visited != set(lookup) or len(manifest["records"]) != len(source["records"]):
        raise ValueError("Encoding preparation missed cached pairs")
    manifest["completed"] = True
    write_ledger(ledger, manifest)
    try:
        finish_report(root, output, manifest)
    except Exception:
        manifest["completed"] = False
        write_ledger(ledger, manifest)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True, help="Existing original body_v1/cache, read-only")
    p.add_argument("--output", required=True)
    p.add_argument("--encoding", choices=(SHAPE_ENCODING, UNCLIPPED_ENCODING, BEAT16_ENCODING), default=SHAPE_ENCODING)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--subjects", nargs="+", default=["HNU11135", "HNU11141", "HNU11119", "HNU11104"])
    run(p.parse_args())


if __name__ == "__main__":
    main()
