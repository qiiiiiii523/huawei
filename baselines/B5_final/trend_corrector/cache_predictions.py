"""Freeze one B5-final 120-second prediction cache for later TCN training.

The final study is generated once by ``python -m baselines.B5_final.validate``.
This command only compacts its selected ``both`` outputs into record-level
prediction files.  Targets remain in the immutable baseline evaluation and are
loaded only by the trainer; no target is present in the inference cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from baselines.B5_best.common import sha
from baselines.B5_best_context.evaluate import pairs


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _key(task: str, pair_id: str) -> str:
    digest = hashlib.sha256(f"{task}:{pair_id}".encode()).hexdigest()[:16]
    return f"{task}_{digest}.npy"


def _split(subject_id: str, validation_fraction: float) -> str:
    """Assign at subject level so no participant leaks across splits."""
    v = int(hashlib.sha256(subject_id.encode()).hexdigest()[:16], 16) / 2**64
    return "validation" if v < validation_fraction else "train"


def build_cache(study: Path, baseline: Path, output: Path, validation_fraction: float, branch_evaluation: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"Cache destination already exists: {output}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    output.mkdir(parents=True)
    index = {"version": 3 if branch_evaluation else 2, "record_shape": [12, 60000], "dtype": "float32", "records": [],
             "branch_mode": "slow" if branch_evaluation else "final"}
    for task in ("task1", "task2"):
        source = baseline / task
        selected = study / "both" / task
        base_rows = _rows(source / "window_metadata.csv")
        selected_rows = _rows(selected / "window_metadata.csv")
        if base_rows != selected_rows:
            raise ValueError(f"{task}: final-study metadata differs from frozen B5-final baseline")
        if any(not row.get("subject_id") for row in base_rows):
            raise ValueError(f"{task}: subject_id is required for leakage-free TCN splitting")
        pred = np.load(selected / "prediction_uV.npy", mmap_mode="r", allow_pickle=False)
        if branch_evaluation:
            branch_folder = branch_evaluation / task
            branch_rows = _rows(branch_folder / "window_metadata.csv")
            if branch_rows != base_rows:
                raise ValueError(f"{task}: branch metadata differs from frozen B5-final baseline")
            fast = np.load(branch_folder / "fast_uV.npy", mmap_mode="r", allow_pickle=False)
            slow = np.load(branch_folder / "slow_uV.npy", mmap_mode="r", allow_pickle=False)
            if fast.shape != pred.shape or slow.shape != pred.shape or fast.dtype != np.float32 or slow.dtype != np.float32:
                raise ValueError(f"{task}: invalid saved B5 branches")
        if pred.shape != (len(base_rows), 12, 5000) or pred.dtype != np.float32:
            raise ValueError(f"{task}: invalid final prediction array {pred.shape}/{pred.dtype}")
        for ordinal, window_indices in enumerate(pairs(base_rows)):
            row = base_rows[window_indices[0]]
            record = np.concatenate([pred[i] for i in window_indices], axis=-1)
            if record.shape != (12, 60000) or not np.isfinite(record).all():
                raise ValueError(f"{task}/{row['pair_id']}: invalid full prediction")
            name = _key(task, row["pair_id"])
            destination = output / "records" / name
            destination.parent.mkdir(exist_ok=True)
            np.save(destination, record, allow_pickle=False)
            item = {
                "task_id": task, "pair_id": row["pair_id"], "record_id": row["target_record_id"],
                "subject_id": row["subject_id"],
                "input_type": row["input_type"], "prediction_file": str(Path("records") / name),
                "window_indices": window_indices, "split": _split(row["subject_id"], validation_fraction),
            }
            if branch_evaluation:
                for label, array in (("fast", fast), ("slow", slow)):
                    branch_record = np.concatenate([array[i] for i in window_indices], axis=-1)
                    if branch_record.shape != (12, 60000) or not np.isfinite(branch_record).all():
                        raise ValueError(f"{task}/{row['pair_id']}: invalid {label} branch")
                    branch_name = f"{label}_{name}"
                    np.save(output / "records" / branch_name, branch_record, allow_pickle=False)
                    item[f"{label}_file"] = str(Path("records") / branch_name)
            index["records"].append(item)
    if not any(x["split"] == "train" for x in index["records"]) or not any(x["split"] == "validation" for x in index["records"]):
        raise ValueError("Split produced an empty train or validation set")
    subject_splits: dict[str, set[str]] = {}
    for row in index["records"]:
        subject_splits.setdefault(row["subject_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in subject_splits.values()):
        raise AssertionError("subject split leakage detected")
    provenance = {
        "completed": True, "protocol": "B5-final-120s-zero-mean-tcn-cache-v2-subject-split",
        "study": str(study.resolve()), "baseline_evaluation": str(baseline.resolve()),
        "study_final_sha256": sha(study / "final.json"),
        "baseline_evaluation_sha256": sha(baseline / "evaluation.json"),
        "validation_fraction": validation_fraction,
        "split_unit": "subject_id",
        "branch_evaluation": None if branch_evaluation is None else str(branch_evaluation.resolve()),
        "counts": {s: sum(r["split"] == s for r in index["records"]) for s in ("train", "validation")},
        "subject_counts": {split: sum(next(iter(splits)) == split for splits in subject_splits.values()) for split in ("train", "validation")},
        "target_in_cache": False,
    }
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "complete.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, help="completed B5_final.validate output (contains both/task*)")
    parser.add_argument("--baseline-evaluation", required=True, help="immutable base_inference used by that study")
    parser.add_argument("--output", required=True)
    parser.add_argument("--branch-evaluation", help="B5-best validation output created with --save-branches; enables slow-branch TCN inputs")
    parser.add_argument("--validation-fraction", type=float, default=.2)
    args = parser.parse_args()
    result = build_cache(Path(args.study), Path(args.baseline_evaluation), Path(args.output), args.validation_fraction,
                         None if args.branch_evaluation is None else Path(args.branch_evaluation))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
