"""Cache the frozen B5-final FM+P base output for full Huawei records.

Only the synchronous, visible I lead enters prediction. The remaining target
channels are never passed to the predictor or copied into this cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from baselines.B5.metadata import DemographicsTable
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.predict import Predictor

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
BASE_SHA = "990f5d95312dba0ab3e4bde416ee15d3b464b902fc63601aab1f852d9881f6f8"


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def records(data_root, split):
    groups = defaultdict(list)
    if split == "train":
        for row in csv_rows(data_root / "metadata/d12_strict_pretrain_index.csv"):
            if row["source_split"] == "train":
                groups[row["target_record_id"]].append((row["source_task_id"], int(row["source_array_index"]), row))
    else:
        for task in ("task1", "task2"):
            for row in csv_rows(data_root / f"datasets/{task}/{task}_window_metadata.csv"):
                if row["split"] == "validation":
                    groups[(task, row["pair_id"])].append((task, int(row["array_index"]), row))
    seen = {}
    for group in groups.values():
        group.sort(key=lambda item: int(item[2]["start_sample_500hz"]))
        starts = [int(item[2]["start_sample_500hz"]) for item in group]
        if starts != list(range(0, 60000, 5000)):
            raise ValueError(f"Not a complete physical 120s record: {group[0][2]['target_record_id']}")
        subject = {item[2]["subject_id"] for item in group}
        record = {item[2]["target_record_id"] for item in group}
        if len(subject) != 1 or len(record) != 1:
            raise ValueError("Mixed subject or target record in a group")
        record_id = next(iter(record))
        if record_id in seen:
            continue
        seen[record_id] = True
        yield record_id, next(iter(subject)), group


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=("train", "validation", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0, help="For a small smoke run; 0 means all records")
    parser.add_argument("--inspect", action="store_true", help="Validate record indexing without loading the model")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    splits = ("train", "validation") if args.split == "both" else (args.split,)
    plans = {split: list(records(data_root, split)) for split in splits}
    print({split: len(plan) for split, plan in plans.items()}, flush=True)
    if args.inspect:
        return
    if sha(args.checkpoint) != BASE_SHA:
        raise ValueError("The checkpoint is not the pinned B5-final FM+P base bundle")
    if args.batch_size < 1 or args.limit < 0:
        raise ValueError("batch-size must be positive and limit nonnegative")
    bundle = load_bundle(args.checkpoint)
    seed_all(bundle["sampling"]["seed"], True)
    predictor = Predictor(bundle, device_from_name(args.device))
    demographics = DemographicsTable(data_root / "demographics/userinfobean.csv")
    arrays = {(task, split): np.load(data_root / f"datasets/{task}/{task}_{split}_target.npy", mmap_mode="r")
              for task in ("task1", "task2") for split in splits}
    args.output.mkdir(parents=True, exist_ok=True)
    for split, plan in plans.items():
        selected = plan[:args.limit] if args.limit else plan
        folder = args.output / split
        folder.mkdir(exist_ok=True)
        for number, (record_id, subject_id, group) in enumerate(selected, 1):
            if not record_id.startswith("d12_") or "/" in record_id or "\\" in record_id:
                raise ValueError("Unexpected record ID")
            prediction_path = folder / f"{record_id}_prediction_uV.npy"
            anchor_path = folder / f"{record_id}_anchor_I_uV.npy"
            audit_path = folder / f"{record_id}.json"
            if all(path.is_file() for path in (prediction_path, anchor_path, audit_path)):
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit.get("checkpoint_sha256") == BASE_SHA and audit.get("record_id") == record_id:
                    print(f"{split} {number}/{len(selected)} {record_id} cached", flush=True)
                    continue
                raise ValueError(f"Existing cache metadata mismatch: {audit_path}")
            if any(path.exists() for path in (prediction_path, anchor_path, audit_path)):
                raise FileExistsError(f"Incomplete existing cache for {record_id}; inspect before retry")
            anchor = np.concatenate([arrays[(task, split)][index, 0] for task, index, _ in group]).astype(np.float32)
            if anchor.shape != (60000,) or not np.isfinite(anchor).all():
                raise ValueError(f"Invalid real 120s I for {record_id}")
            prediction = predictor.predict_encoded(anchor, demographics.get(subject_id), record_id, args.batch_size)
            if prediction.shape != (12, 60000) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid prediction for {record_id}")
            np.save(prediction_path, prediction.astype(np.float32), allow_pickle=False)
            np.save(anchor_path, anchor, allow_pickle=False)
            write_json(audit_path, {"split": split, "record_id": record_id, "subject_id": subject_id,
                                    "source_task": group[0][0], "source_window_indices": [item[1] for item in group],
                                    "checkpoint_sha256": BASE_SHA, "record_key": record_id,
                                    "sampling": bundle["sampling"], "slow_beta": bundle.get("slow_beta"),
                                    "shape": [12, 60000], "lead_order": LEADS, "sampling_rate_hz": 500,
                                    "unit": "uV", "target_used_at_inference": False,
                                    "prediction_sha256": sha(prediction_path), "anchor_sha256": sha(anchor_path)})
            print(f"{split} {number}/{len(selected)} {record_id} saved", flush=True)
    write_json(args.output / "run.json", {"completed": args.limit == 0, "split": args.split,
                                          "counts": {split: len(plan) for split, plan in plans.items()},
                                          "checkpoint_sha256": BASE_SHA, "target_used_at_inference": False})


if __name__ == "__main__":
    main()
