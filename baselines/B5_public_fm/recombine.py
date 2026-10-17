"""Rescore saved FM predictions with a new portable bundle's slow CNN; no FM regeneration/training."""
import argparse
from pathlib import Path

import numpy as np
import torch

from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.slow import load_slow
from baselines.B5_best.combine import combine
from baselines.B5_best.validate import pair_indices
from baselines.B5.runtime import device_from_name
from baselines.B5.scoring import diagnostic_summary, scores_from_summaries, write_score_report
from ecg12gen.evaluate import evaluate_record_predictions, write_report
from .common import load_study, read, write, sha, csv_write, fresh_directory
from .evaluation import HybridEvaluator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--evaluation", required=True, help="Existing huawei/evaluations/epoch_NNNN")
    p.add_argument("--slow-bundle", required=True, help="Portable bundle containing intended new S50 slow weights and matching scales")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    path, study = load_study(args.study)
    original = read(Path(args.evaluation) / "evaluation.json")
    if not original.get("completed") or original["study_sha256"] != sha(path) or original["sampling"] != study["settings"]["sampling"]:
        raise ValueError("Requires completed evaluation with the same sampling")
    evaluator = HybridEvaluator(path, study, device_from_name(args.device))
    new = load_bundle(args.slow_bundle)
    if new["fm"]["scales_sha256"] != evaluator.fixed["fm"]["scales_sha256"]:
        raise ValueError("New slow bundle has different scales")
    evaluator.slow = load_slow(new["slow"], evaluator.device, new["provenance"]["slow_variant"])
    output = fresh_directory(args.output)
    summaries = {}
    for task, dataset in evaluator.datasets.items():
        source = Path(args.evaluation) / task
        if sha(source / "fm_prediction_uV.npy") != original["FM_prediction_sha256"][task]:
            raise ValueError("Cached FM prediction changed")
        for name, field in (("window_metadata.csv", "metadata_sha256"), ("target_uV.npy", "target_sha256"), ("target_quality_mask.npy", "QC_sha256")):
            if sha(source / name) != original["population"][task][field]:
                raise ValueError("Saved evaluation data changed: " + name)
        import csv
        with (source / "window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        keys = ("pair_id", "target_record_id", "subject_id", "start_sample_500hz", "input_type")
        if len(rows) != len(dataset) or any(any(str(a[k]) != str(b[k]) for k in keys) for a, b in zip(rows, dataset.rows)):
            raise ValueError("Saved prediction order differs from study")
        raw = np.load(source / "fm_prediction_uV.npy", mmap_mode="r", allow_pickle=False)
        target = np.load(source / "target_uV.npy", mmap_mode="r", allow_pickle=False)
        quality = np.load(source / "target_quality_mask.npy", mmap_mode="r", allow_pickle=False)
        folder = output / task
        folder.mkdir()
        prediction = np.lib.format.open_memmap(folder / "prediction_uV.npy", mode="w+", dtype=np.float32, shape=raw.shape)
        for indices in pair_indices(rows):
            first = rows[indices[0]]
            anchor = np.concatenate([dataset.base[i].anchor_i_ecg for i in indices], -1).astype(np.float32)
            learned, _ = evaluator.slow_prediction(anchor, dataset.demographics.get(first["subject_id"]), first["target_record_id"])
            fm = np.concatenate([raw[i] for i in indices], -1)
            hybrid = combine(fm, learned, study["settings"]["slow_beta"])
            for window, index in enumerate(indices):
                prediction[index] = hybrid[:, window * 5000:(window + 1) * 5000]
        prediction.flush()
        overall, details = evaluate_record_predictions(prediction, target, task, rows, target_quality_mask=quality,
            input_quality_mask=quality[:, 0], require_reliable_input=True, input_quality_source="same_record_target_I")
        summaries[task] = diagnostic_summary(overall)
        write_report(folder, summaries[task], details, title="New slow CNN + identical cached FM")
        csv_write(folder / "window_metadata.csv", rows)
        write(folder / "target_reference.json", {"path": str((source / "target_uV.npy").resolve()), "sha256": sha(source / "target_uV.npy")})
        prediction._mmap.close()
        for value in (raw, target, quality):
            value._mmap.close()
    score = scores_from_summaries(summaries)
    write_score_report(output, score)
    write(output / "evaluation.json", {"completed": True, "score": score, "FM_regenerated": False, "training_started": False,
          "source_evaluation": str(Path(args.evaluation).resolve()), "slow_bundle_sha256": sha(args.slow_bundle),
          "fixed_beta": study["settings"]["slow_beta"], "study_sha256": sha(path)})
    print(score, flush=True)


if __name__ == "__main__":
    main()
