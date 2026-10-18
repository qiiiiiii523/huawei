"""Freeze latest B5-best body train predictions and reuse the existing validation cache."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, sha
from .adapters import history_from_sample, _visible_digest
from .body_features import BODY_PROTOCOL, relation_features
from .evaluate import load_baseline, pairs


def write_ledger(path, value):
    temp = path.with_name("cache.pending.json")
    write(temp, value)
    temp.replace(path)


def load_cache(path):
    path = Path(path).resolve()
    manifest = read(path / "cache.json")
    if manifest.get("completed") is not True or manifest.get("protocol") != BODY_PROTOCOL:
        raise ValueError("Finish body_prepare before training; cache is incomplete")
    for row in manifest["records"]:
        if sha(path / row["file"]) != row.get("sha256"):
            raise ValueError("Frozen body cache changed: " + row["file"])
    train = [r for r in manifest["records"] if r["split"] == "train"]
    val = [r for r in manifest["records"] if r["split"] == "validation"]
    for key in ("subject_id", "target_record_id"):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError("Train/validation leakage: " + key)
    return manifest


def prepare(args):
    # No optimizer exists in this command. Only the two frozen B5-best networks run.
    from baselines.B5_best.bundle import DEFAULT_PROFILE, load_inference_bundle
    from baselines.B5_best.predict import Predictor
    from baselines.B5.config import load_config
    from baselines.B5.data import common_config, demographics_table, demographics_path
    from baselines.B5.runtime import device_from_name, seed_all
    from baselines.B5.metadata import subject_key
    from ecg12gen.dataset import JointAnchorDataset
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output or --resume; tee logs must stay outside it")
    baseline, validation, fingerprints = load_baseline(args.baseline_evaluation)
    bundle, checkpoint = load_inference_bundle(args.checkpoint, args.profile or str(DEFAULT_PROFILE))
    sampling = {"seed": 42, "solver": "heun", "steps": 16, "samples": 16}
    if (bundle["sampling"] != sampling or bundle.get("slow_beta") != .75
            or baseline["checkpoint_sha256"] != sha(checkpoint)):
        raise ValueError("Validation and training must use the same current B5-best Heun16/K16 seed42 beta=.75 bundle")
    cfg = load_config(args.config, data_only=True)
    common = common_config(cfg)
    inputs = {"demographics": demographics_path(cfg), "split": common.path("subject_split_csv"),
              "qc": common.path("device_interpretation_qc_csv"),
              "pairs": common.path("task2_pair_manifest_csv")}
    folder = common.path("task2_output")
    for name in ("task2_window_metadata.csv", "task2_context_window_metadata.csv"):
        inputs[name] = folder / name
    for split in ("train", "validation"):
        for kind in ("target", "context", "context_valid_mask"):
            name = f"task2_{split}_{kind}.npy"
            inputs[name] = folder / name
    frozen = {"base_checkpoint": str(checkpoint), "base_checkpoint_sha256": sha(checkpoint),
        "baseline_evaluation": str(Path(args.baseline_evaluation).resolve()),
        "baseline_population": fingerprints, "sampling": sampling, "slow_beta": .75,
        "sources": {k: {"path": str(p.resolve()), "sha256": sha(p)} for k, p in inputs.items()}}
    ledger = output / "cache.json"
    if ledger.exists():
        manifest = read(ledger)
        if manifest["frozen"] != frozen or manifest.get("protocol") != BODY_PROTOCOL:
            raise ValueError("Resume changed B5-best/data/QC/sampling; use a new cache directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output contains files but no resumable cache.json")
        manifest = {"protocol": BODY_PROTOCOL, "completed": False, "frozen": frozen, "records": []}
        output.mkdir(parents=True, exist_ok=True)
        write_ledger(ledger, manifest)
    existing = {(r["split"], r["pair_id"]): r for r in manifest["records"]}
    demographics = demographics_table(cfg)
    predictor, generated = None, {}
    datasets = []
    try:
        for split in ("train", "validation"):
            dataset = JointAnchorDataset(common, "task2", split)
            datasets.append(dataset)
            rows = [dataset._rows[i] for i in dataset._indices]
            lookup = {(r["pair_id"], int(r["start_sample_500hz"])): i for i, r in enumerate(validation["task2"]["rows"])}
            for indices in pairs(rows):
                row = rows[indices[0]]
                if row["input_type"] != "body_scale_d6":
                    continue
                key = split, row["pair_id"]
                if key in existing:
                    if sha(output / existing[key]["file"]) != existing[key]["sha256"]:
                        raise ValueError("Resume found changed cache file: " + existing[key]["file"])
                    continue
                samples = [dataset[i] for i in indices]
                anchor = np.concatenate([s.anchor_i_ecg for s in samples], axis=-1).astype(np.float32)
                truth = np.concatenate([s.Y_12lead[1] for s in samples])
                quality = np.asarray(samples[0].target_quality_mask, bool)
                if any(not np.array_equal(s.target_quality_mask, quality) for s in samples):
                    raise ValueError("Target QC changed within a record")
                history = history_from_sample(samples[0])
                beat, stats, available = relation_features(history)
                if split == "validation":
                    ids = [lookup[(row["pair_id"], j * 5000)] for j in range(12)]
                    item = validation["task2"]
                    for local, i in enumerate(ids):
                        if (not np.array_equal(samples[local].Y_12lead, item["target"][i])
                                or not np.array_equal(quality, item["quality"][i])
                                or item["rows"][i]["subject_id"] != row["subject_id"]):
                            raise ValueError("Validation cache/live target/QC/subject mismatch")
                    base = np.concatenate([item["prediction"][i, 1] for i in ids])
                else:
                    # Only the observed I and demographic fields enter the predictor.
                    encoded_meta = demographics.get(row["subject_id"])
                    digest = hashlib.sha256(anchor.tobytes())
                    for name, value in sorted(encoded_meta.items()):
                        digest.update(name.encode()); digest.update(np.asarray(value).tobytes())
                    target_id = row["target_record_id"]
                    if target_id not in generated:
                        if predictor is None:
                            seed_all(42, True)
                            predictor = Predictor(bundle, device_from_name(args.device))
                        base = predictor.predict_encoded(anchor, encoded_meta, target_id, args.batch_size)[1]
                        generated[target_id] = digest.hexdigest(), base
                    else:
                        previous, base = generated[target_id]
                        if previous != digest.hexdigest():
                            raise ValueError("Repeated target has different observed I/demographics")
                name = hashlib.sha256((split + ":" + row["pair_id"]).encode()).hexdigest()[:24] + ".npz"
                relative = split + "/" + name
                dest = output / relative
                dest.parent.mkdir(exist_ok=True)
                # Replace an interrupted record atomically; ledger is written only after completion.
                temp = dest.with_suffix(".pending")
                with temp.open("wb") as handle:
                    np.savez_compressed(handle, anchor=anchor[0], base_ii=base, target_ii=truth,
                        quality=quality, beat=beat, stats=stats, available=np.array(available))
                temp.replace(dest)
                record = {"split": split, "pair_id": row["pair_id"],
                    "subject_id": subject_key(row["subject_id"]), "target_record_id": row["target_record_id"],
                    "history_record_id": history.record_id, "history_visible_sha256": _visible_digest(history),
                    "history_available": bool(available), "I_II_eligible": bool(quality[0] and quality[1]),
                    "file": relative, "sha256": sha(dest)}
                manifest["records"].append(record)
                write_ledger(ledger, manifest)
                print(f"{split}: {row['pair_id']} cached; history={available}, I/II_QC={record['I_II_eligible']}", flush=True)
        counts = {s: sum(r["split"] == s and r["I_II_eligible"] for r in manifest["records"])
                  for s in ("train", "validation")}
        if not all(counts.values()):
            raise ValueError("No eligible body I/II records in a split")
        manifest["completed"] = True
        write_ledger(ledger, manifest)
        try:
            load_cache(output)  # Check subject/target disjointness and every saved file.
        except Exception:
            manifest["completed"] = False
            write_ledger(ledger, manifest)
            raise
        write(output / "preflight.json", {"completed": True, "training_started": False,
            "optimizer_updates": 0, "eligible_body_pairs": counts, "frozen": frozen})
        print({"completed": True, "training_started": False, "eligible_pairs": counts}, flush=True)
    finally:
        for dataset in datasets:
            dataset.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--baseline-evaluation", required=True, help="Original base_inference, not a postprocessed study group")
    p.add_argument("--checkpoint")
    p.add_argument("--profile")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
