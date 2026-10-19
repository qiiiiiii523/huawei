"""Cross three fixed FM candidates with one P slow CNN using verified cached predictions.

Lives outside frozen experiment packages: adding this tool does not invalidate study.json.
No optimizer, FM inference, public waveform reads or modification of source artifacts.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from baselines.B5.checkpoint import atomic_save, load_checkpoint
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5.scoring import SCORE_PROTOCOL, diagnostic_summary, scores_from_summaries, write_score_report
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.combine import combine
from baselines.B5_best.slow import load_slow
from baselines.B5_best.validate import pair_indices
from baselines.B5_public_fm.common import load_study, read, write, sha, state_sha, csv_write, fresh_directory
from baselines.B5_public_fm.evaluation import HybridEvaluator, candidate_bundle
from ecg12gen.evaluate import evaluate_record_predictions, write_report

NAMES = ("control100", "FM_C_best100", "FM_P_best45")
SCORE_KEYS = ("task1_r_missing11", "task2_r_missing11", "main_correlation")


def same_condition_contract(a, b):
    for key in ("architecture_hash", "condition_schema", "preprocessing_version", "scales_sha256"):
        if a[key] != b[key]:
            raise ValueError("P bundle and frozen PublicFM differ: " + key)


def sources(root, review, study_path, fixed):
    """Bind named candidates to pinned epoch, state, source SHA and both cached seeds."""
    result = {}
    result[NAMES[0]] = {"checkpoint": fixed["fm"], "sha256": sha(study_path.parent / "fixed_slow_bundle.pt"),
                        "seed42": root / "baseline", "seed43": review / "stable_baseline"}
    for group, name, epoch, old_name in (("control", NAMES[1], 100, "FM-C_best100"),
                                         ("expanded", NAMES[2], 45, "FM-P_best45")):
        folder = root / group / "huawei"
        done = read(folder / "completed.json")
        if not done.get("completed") or done["study_sha256"] != sha(study_path) or done["best"]["epoch"] != epoch:
            raise ValueError("Completed group or pinned best epoch mismatch: " + group)
        path = folder / done["best"]["path"]
        if sha(path) != done["best"]["sha256"]:
            raise ValueError("Candidate checkpoint changed: " + str(path))
        checkpoint = load_checkpoint(path)
        same_condition_contract(checkpoint, fixed["fm"])
        evaluation = read(folder / "evaluations" / f"epoch_{epoch:04d}" / "evaluation.json")
        if evaluation["score"] != done["best"]["score"] or evaluation["score"] != checkpoint["validation_score"]:
            raise ValueError("Seed42 evaluation does not match the pinned selected checkpoint")
        result[name] = {"checkpoint": checkpoint, "checkpoint_path": path, "sha256": sha(path),
                        "seed42": folder / "evaluations" / f"epoch_{epoch:04d}", "seed43": review / old_name}
    return result


def check_cache(folder, seed, source_sha, study_path, study, population, fixed_slow_sha):
    report = read(folder / "evaluation.json")
    sampling = dict(study["settings"]["sampling"], seed=seed)
    if (not report.get("completed") or report["study_sha256"] != sha(study_path) or
            report["score_protocol"] != SCORE_PROTOCOL or report["sampling"] != sampling or
            report["population"] != population or report["fixed_slow_state_sha256"] != fixed_slow_sha or
            report["slow_beta"] != .75 or report.get("target_used_at_inference") is not False):
        raise ValueError("Cached evaluation protocol/population/slow identity mismatch: " + str(folder))
    if seed == 43:
        if read(folder / "source.json")["source_sha256"] != source_sha:
            raise ValueError("Seed43 cache belongs to another FM checkpoint")
    for task in ("task1", "task2"):
        files = {"fm_prediction_uV.npy": report["FM_prediction_sha256"][task],
                 "window_metadata.csv": population[task]["metadata_sha256"],
                 "target_uV.npy": population[task]["target_sha256"],
                 "target_quality_mask.npy": population[task]["QC_sha256"]}
        for filename, expected in files.items():
            if sha(folder / task / filename) != expected:
                raise ValueError("Cached artifact changed: " + str(folder / task / filename))
    return report


@torch.no_grad()
def rescore(engine, source, output, seed, source_sha, slow_sha, study_sha):
    """Keep original FM samples; replace only the predicted slow component."""
    output = fresh_directory(output)
    summaries = {}
    for task, dataset in engine.datasets.items():
        src = source / task
        with (src / "window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        keys = ("pair_id", "target_record_id", "subject_id", "start_sample_500hz", "input_type")
        if len(rows) != len(dataset) or any(any(str(a[k]) != str(b[k]) for k in keys) for a, b in zip(rows, dataset.rows)):
            raise ValueError("Cached row order differs from current validation data")
        with ExitStack() as stack:
            arrays = {}
            for name in ("fm_prediction_uV", "target_uV", "target_quality_mask"):
                arrays[name] = np.load(src / (name + ".npy"), mmap_mode="r", allow_pickle=False)
                stack.callback(arrays[name]._mmap.close)
            raw, target, quality = (arrays[n] for n in ("fm_prediction_uV", "target_uV", "target_quality_mask"))
            shape = (len(rows), 12, 5000)
            if raw.shape != shape or target.shape != shape or quality.shape != (len(rows), 12):
                raise ValueError("Invalid cached waveform/mask shapes")
            folder = output / task
            folder.mkdir()
            predicted = np.lib.format.open_memmap(folder / "prediction_uV.npy", mode="w+", dtype=np.float32, shape=shape)
            stack.callback(predicted._mmap.close)
            for indices in pair_indices(rows):
                first = rows[indices[0]]
                anchor = np.concatenate([dataset.base[i].anchor_i_ecg for i in indices], -1).astype(np.float32)
                learned, _ = engine.slow_prediction(anchor, dataset.demographics.get(first["subject_id"]), first["target_record_id"])
                complete_fm = np.concatenate([raw[i] for i in indices], -1)
                if not np.isfinite(complete_fm).all():
                    raise ValueError("Nonfinite cached FM waveform")
                hybrid = combine(complete_fm, learned, .75)
                for window, index in enumerate(indices):
                    predicted[index] = hybrid[:, window * 5000:(window + 1) * 5000]
            predicted.flush()
            overall, details = evaluate_record_predictions(predicted, target, task, rows,
                target_quality_mask=quality, input_quality_mask=quality[:, 0], require_reliable_input=True,
                input_quality_source="same_record_target_I")
            summaries[task] = diagnostic_summary(overall)
            write_report(folder, summaries[task], details, title="Fixed FM cache plus current P slow CNN")
            csv_write(folder / "window_metadata.csv", rows)
    score = scores_from_summaries(summaries)
    write_score_report(output, score)
    write(output / "evaluation.json", {"completed": True, "score": score, "sampling": dict(engine.study["settings"]["sampling"], seed=seed),
          "slow_beta": .75, "FM_regenerated": False, "training_started": False, "target_used_at_inference": False,
          "source_evaluation": str(source), "FM_source_sha256": source_sha, "slow_bundle_sha256": slow_sha,
          "slow_state_sha256": engine.new_slow_sha, "study_sha256": study_sha,
          "score_protocol": SCORE_PROTOCOL, "population": read(source / "evaluation.json")["population"]})
    return score


def summarize(rows):
    """Compare within each seed first; mean-of-scores is not a waveform ensemble."""
    result = []
    for seed in (42, 43):
        group = {r["candidate"]: r for r in rows if r["seed"] == seed}
        if set(group) != set(NAMES):
            raise ValueError("Need all three candidates for each seed")
        for name in NAMES:
            row = group[name]
            result.append({**row, "delta_vs_control100_P": row["main_correlation"] - group[NAMES[0]]["main_correlation"],
                           "delta_vs_FM_C_P": row["main_correlation"] - group[NAMES[1]]["main_correlation"]})
    means = [{"candidate": n, **{k: sum(r[k] for r in rows if r["candidate"] == n) / 2 for k in SCORE_KEYS}}
             for n in NAMES]
    return result, means


def run(args):
    root, review, output = Path(args.run).resolve(), Path(args.seed43_review).resolve(), Path(args.output).resolve()
    path, study = load_study(root / "shared/study.json", verify_data=False)
    for key, item in study["huawei_sources"].items():
        print("Checking Huawei source:", key, flush=True)
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Huawei input/QC/metadata changed: " + key)
    if study["settings"]["sampling"] != {"seed": 42, "solver": "heun", "steps": 16, "samples": 16} or study["settings"]["slow_beta"] != .75:
        raise ValueError("Expected original seed42/Heun16/K16/beta0.75 study")
    if not read(review / "completed.json").get("completed"):
        raise ValueError("Seed43 review did not finish")
    fixed = load_bundle(path.parent / "fixed_slow_bundle.pt")
    new = load_bundle(args.slow_bundle)
    same_condition_contract(fixed["fm"], new["fm"])
    if new["fm_state_sha256"] != fixed["fm_state_sha256"] or new["slow_beta"] != .75:
        raise ValueError("Current P bundle must contain the SAME control100 FM and beta0.75")
    candidates = sources(root, review, path, fixed)
    population = read(root / "baseline/evaluation.json")["population"]
    # Fail before expensive CNN inference if any of the six original prediction caches are missing/changed.
    reports = {}
    for name, item in candidates.items():
        for seed in (42, 43):
            reports[name, seed] = check_cache(item[f"seed{seed}"], seed, item["sha256"], path, study, population, fixed["slow_state_sha256"])
    output = fresh_directory(output)
    slow_sha = sha(args.slow_bundle)
    identity = {"training_started": False, "FM_regenerated": False, "study_sha256": sha(path), "slow_bundle": str(Path(args.slow_bundle).resolve()),
                "slow_bundle_sha256": slow_sha, "slow_state_sha256": new["slow_state_sha256"], "seeds": [42, 43],
                "slow_beta": .75, "score_protocol": SCORE_PROTOCOL, "seed43_review": str(review),
                "candidates": {n: {"sha256": x["sha256"], "seed42": str(x["seed42"]), "seed43": str(x["seed43"])} for n, x in candidates.items()}}
    write(output / "cross.json", identity)
    device = device_from_name(args.device)
    seed_all(42, True)
    engine = HybridEvaluator(path, copy.deepcopy(study), device)
    engine.slow = load_slow(new["slow"], device, new["provenance"]["slow_variant"])
    engine.learned_cache.clear()
    engine.new_slow_sha = new["slow_state_sha256"]
    rows = []
    for seed in (42, 43):
        for name, item in candidates.items():
            print(f"Cross: seed={seed}, FM={name}, fixed=P slow CNN", flush=True)
            score = rescore(engine, item[f"seed{seed}"], output / f"seed{seed}" / name, seed, item["sha256"], slow_sha, sha(path))
            old = reports[name, seed]["score"]
            rows.append({"candidate": name, "seed": seed, **{k: score[k] for k in SCORE_KEYS},
                         "old_PCC20_main_r": old["main_correlation"], "delta_new_P_minus_old_PCC20": score["main_correlation"] - old["main_correlation"],
                         "score_protocol": SCORE_PROTOCOL})
            csv_write(output / "partial_summary.csv", rows)
            if seed == 42 and name == NAMES[0]:
                reference = new.get("reference_qc_r")
                same_scope = new.get("reference_score_protocol") == SCORE_PROTOCOL
                matched = (all(abs(score[k] - reference[k]) <= new.get("reference_tolerance", 2e-5) for k in reference)
                           if reference and same_scope else None)
                write(output / "P_reference_check.json", {"reference": reference, "actual": score, "same_scope": same_scope, "reference_matched": matched})
                if matched is False:
                    raise ValueError("Current P baseline not reproduced; inspect P_reference_check.json before comparing")
            if seed == 42:
                bundle = candidate_bundle(item["checkpoint"], new, item.get("checkpoint_path"), score, study["huawei_manifests"], study["huawei_sources"])
                bundle["provenance"].update(cross_experiment="PublicFM_with_current_P", cross_source_slow_bundle_sha256=slow_sha,
                                             cross_FM_source_sha256=item["sha256"], cross_candidate=name)
                bundle_path = output / "bundles" / (name + "__P.pt")
                atomic_save(bundle_path, bundle)
                write(bundle_path.with_suffix(".json"), {"bundle_sha256": sha(bundle_path), "sampling": bundle["sampling"],
                      "slow_state_sha256": bundle["slow_state_sha256"], "FM_state_sha256": bundle["fm_state_sha256"], "score": score})
    table, means = summarize(rows)
    csv_write(output / "summary.csv", table)
    csv_write(output / "seed_mean.csv", means)
    write(output / "completed.json", {**identity, "completed": True, "rows": table, "seed_mean": means,
          "note": "Two inference seeds of fixed training checkpoints; no retraining or seed/epoch reselection"})
    print("All six combinations complete:", output / "summary.csv", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="/home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1")
    p.add_argument("--seed43-review", default="/home/qht/huawei/results/B5-best-exp-PublicFM/public_expansion_v1/seed43_review_20261019_125707_2872275")
    p.add_argument("--slow-bundle", default="/home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4/P/model.pt")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())


if __name__ == "__main__":
    main()
