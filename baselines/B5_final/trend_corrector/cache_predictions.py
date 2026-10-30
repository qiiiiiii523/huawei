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


def _split(task: str, pair_id: str, validation_fraction: float) -> str:
    v = int(hashlib.sha256(f"{task}:{pair_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "validation" if v < validation_fraction else "train"


def build_cache(study: Path, baseline: Path, output: Path, validation_fraction: float) -> dict:
    if output.exists():
        raise FileExistsError(f"Cache destination already exists: {output}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    output.mkdir(parents=True)
    index = {"version": 1, "record_shape": [12, 60000], "dtype": "float32", "records": []}
    for task in ("task1", "task2"):
        source = baseline / task
        selected = study / "both" / task
        base_rows = _rows(source / "window_metadata.csv")
        selected_rows = _rows(selected / "window_metadata.csv")
        if base_rows != selected_rows:
            raise ValueError(f"{task}: final-study metadata differs from frozen B5-final baseline")
        pred = np.load(selected / "prediction_uV.npy", mmap_mode="r", allow_pickle=False)
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
            index["records"].append({
                "task_id": task, "pair_id": row["pair_id"], "record_id": row["target_record_id"],
                "input_type": row["input_type"], "prediction_file": str(Path("records") / name),
                "window_indices": window_indices, "split": _split(task, row["pair_id"], validation_fraction),
            })
    if not any(x["split"] == "train" for x in index["records"]) or not any(x["split"] == "validation" for x in index["records"]):
        raise ValueError("Split produced an empty train or validation set")
    provenance = {
        "completed": True, "protocol": "B5-final-120s-zero-mean-tcn-cache-v1",
        "study": str(study.resolve()), "baseline_evaluation": str(baseline.resolve()),
        "study_final_sha256": sha(study / "final.json"),
        "baseline_evaluation_sha256": sha(baseline / "evaluation.json"),
        "validation_fraction": validation_fraction,
        "counts": {s: sum(r["split"] == s for r in index["records"]) for s in ("train", "validation")},
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
    parser.add_argument("--validation-fraction", type=float, default=.2)
    args = parser.parse_args()
    result = build_cache(Path(args.study), Path(args.baseline_evaluation), Path(args.output), args.validation_fraction)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
