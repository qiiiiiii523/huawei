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
                    if (raw.record_id != b_overlay.record_id or not np.array_equal(raw.valid_mask, b_overlay.valid_mask)
                            or not np.array_equal(filtered.valid_mask, b_overlay.valid_mask)
                            or not np.array_equal(filtered.raw_uV[filtered.valid_mask], b_overlay.raw_uV[b_overlay.valid_mask])):
                        raise ValueError("Runtime B filter does not reproduce uploaded overlay on valid points")
                    b_features = relation_features(filtered, encoding=BEAT16_ENCODING)
                    overlay_features = relation_features(b_overlay, encoding=BEAT16_ENCODING)
                    if any(not np.array_equal(x, y) for x, y in zip(b_features, overlay_features)):
                        raise ValueError("B inference/template features do not match B overlay")
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
        "B_overlay_runtime_features_exactly_matched": True}
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
