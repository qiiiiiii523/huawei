"""CPU-only two-task scoring of SAVED predictions; no model imports or inference.

The caller must declare where actual input reliability comes from. A saved target
I mask is usable only when the actual model input is that same record's I.
"""
from pathlib import Path
import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg12gen.evaluate import (evaluate_record_predictions, write_report,
                               correlation_score_from_summaries, write_correlation_score)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_target(folder):
    path = folder / "target_uV.npy"
    if path.is_file():
        return path
    reference = read_json(folder / "target_reference.json")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = folder / path
    if not path.is_file() or sha(path) != reference["sha256"]:
        raise ValueError("Target reference missing or SHA256 mismatch")
    return path.resolve()


def rescore(evaluation, output, source, input_masks=None):
    evaluation, output = Path(evaluation).resolve(), Path(output).resolve()
    if source != "same_record_target_I" and (not input_masks or set(input_masks) != {"task1", "task2"}):
        raise ValueError("Explicit source requires both task input-mask paths")
    if source == "same_record_target_I" and input_masks:
        raise ValueError("Choose same-record target I OR explicit input masks, not both")
    # Never place reports inside the historical evaluation or overwrite its files.
    if output == evaluation or evaluation in output.parents:
        raise ValueError("Use a separate output directory outside the source evaluation")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; keep tee log beside it")
    output.mkdir(parents=True, exist_ok=True)
    summaries, provenance = {}, {}
    for task in ("task1", "task2"):
        print(f"Scoring saved {task}: {evaluation / task}", flush=True)
        folder = evaluation / task
        paths = {"prediction": folder / "prediction_uV.npy", "target": resolve_target(folder),
                 "quality": folder / "target_quality_mask.npy", "metadata": folder / "window_metadata.csv"}
        if input_masks:
            paths["input_quality"] = Path(input_masks[task]).resolve()
        with paths["metadata"].open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        # Hash before use and verify after use: reports cannot silently describe changed arrays.
        hashes = {key: sha(path) for key, path in paths.items()}
        with ExitStack() as stack:
            arrays = {}
            for key in ("prediction", "target", "quality", "input_quality"):
                if key in paths:
                    value = np.load(paths[key], mmap_mode="r", allow_pickle=False)
                    stack.callback(value._mmap.close)
                    arrays[key] = value
            observed = arrays["quality"][:, 0] if source == "same_record_target_I" else arrays["input_quality"]
            summary, details = evaluate_record_predictions(
                arrays["prediction"], arrays["target"], task, rows,
                target_quality_mask=arrays["quality"], input_quality_mask=observed,
                require_reliable_input=True, input_quality_source=source)
            retained = np.asarray(observed).reshape(-1).astype(bool)
            retained_rows = [row for i, row in enumerate(rows) if retained[i]]
        if any(sha(path) != hashes[key] for key, path in paths.items()):
            raise ValueError("Source files changed during scoring; discard this incomplete output")
        report = output / task
        write_report(report, summary, details, title="Reliable actual-I input, target-lead QC, raw 120-s r")
        with (report / "retained_window_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(retained_rows)
        (report / "overall_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                                allow_nan=False) + "\n", encoding="utf-8")
        summaries[task] = summary
        provenance[task] = {key: {"path": str(path), "sha256": hashes[key]} for key, path in paths.items()}
        print(f"{task}: r={summary['r_missing11']:.8f}, excluded pairs={summary['input_qc_excluded_pairs']}", flush=True)
    score = correlation_score_from_summaries(summaries)
    write_correlation_score(output, score)
    audit = {"completed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "source_evaluation": str(evaluation), "source_files": provenance,
             "input_quality_source": source, "score": score, "training_started": False,
             "model_loaded": False, "new_inference": False, "checkpoint_reselected": False,
             "source_arrays_modified": False}
    (output / "rescore.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "output_dir": str(output), **score}, ensure_ascii=False), flush=True)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-quality-source", required=True,
                        help="same_record_target_I asserts actual synchronous I is from this target record; otherwise describe explicit source")
    parser.add_argument("--task1-input-quality-mask")
    parser.add_argument("--task2-input-quality-mask")
    args = parser.parse_args()
    masks = {name: path for name, path in (("task1", args.task1_input_quality_mask),
                                         ("task2", args.task2_input_quality_mask)) if path}
    rescore(args.evaluation_dir, args.output_dir, args.input_quality_source, masks or None)


if __name__ == "__main__":
    main()
