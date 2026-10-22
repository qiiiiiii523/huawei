"""Three training seeds and paired patient bootstrap for fixed single-template C3 continuation."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import numpy as np
from baselines.B5_best.common import read, write, csv_write, sha

LEADS = ("II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def target_table(path):
    grouped = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["input_type"] != "body_scale_d6" or row["input_I_reliable"].lower() != "true":
                continue
            key = row["subject_id"], row["target_record_id"]
            grouped.setdefault(key, []).append([float(row["r_" + lead])
                if row["r_" + lead] else np.nan for lead in LEADS])
    result = {}
    for key, values in grouped.items():
        x = np.asarray(values)
        count = np.isfinite(x).sum(0)
        result[key] = np.divide(np.nansum(x, axis=0), count,
            out=np.full(11, np.nan), where=count > 0)
    return result


def paired_bootstrap(reference, candidate, draws=5000, seed=20261021):
    """Sample patients, keep their targets/pairs together, then official lead means / 4."""
    if set(reference) != set(candidate) or not reference:
        raise ValueError("Bootstrap requires the exact paired target population")
    keys = sorted(reference)
    left = np.stack([reference[k] for k in keys])
    right = np.stack([candidate[k] for k in keys])
    if not np.array_equal(np.isfinite(left), np.isfinite(right)):
        raise ValueError("Candidate changed defined target/lead cohort")
    mask = np.isfinite(left)
    if not mask.any(axis=0).all():
        raise ValueError("No defined targets for one or more leads")
    delta = np.where(mask, right - left, 0)
    subjects = sorted({k[0] for k in keys})
    index = np.array([subjects.index(k[0]) for k in keys])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), size=draws)
    weights = counts[:, index]
    denominator = weights @ mask.astype(float)
    valid = (denominator > 0).all(axis=1)
    samples = ((weights[valid] @ delta) / denominator[valid]).mean(axis=1) / 4
    point = float((delta.sum(0) / mask.sum(0)).mean() / 4)
    per_subject = []
    for j in range(len(subjects)):
        eligible = mask[index == j]
        diff = delta[index == j]
        per_subject.append(float(diff.sum() / eligible.sum()))
    return {"delta_mean_r": point, "CI95_low": float(np.quantile(samples, .025)),
        "CI95_high": float(np.quantile(samples, .975)), "bootstrap_draws": draws,
        "valid_bootstrap_draws": int(valid.sum()), "subjects": len(subjects),
        "subjects_improved": int((np.array(per_subject) > 1e-10).sum()),
        "subjects_worsened": int((np.array(per_subject) < -1e-10).sum())}, samples


def summarize(args):
    history, out = Path(args.history_run).resolve(), Path(args.output).resolve()
    features = read(history / "features/features.json")
    # Do not assume the initializer's epoch number; read its selected epoch explicitly.
    completed = read(Path(features["initial_checkpoint"]).parent / "completed.json")
    frozen = Path(features["initial_checkpoint"]).parent / f"validation/epoch_{completed['best_epoch']:03d}/pair_metrics.csv"
    reference = target_table(frozen)
    groups = {42: history / "single", 43: out / "seed43", 44: out / "seed44"}
    base_config = read(groups[42] / "config.json")
    rows, bootstrap = [], {"best": [], "last": []}
    for seed, folder in groups.items():
        config = read(folder / "config.json")
        if any(config[k] != v for k, v in base_config.items() if k != "seed") or config["seed"] != seed:
            raise ValueError("Seed runs changed another training factor")
        result = read(folder / "completed.json")
        for checkpoint in ("best", "last"):
            path = folder / "controls" / checkpoint / "correct/pair_metrics.csv"
            stats, samples = paired_bootstrap(reference, target_table(path))
            expected = result["best_mean_r"] if checkpoint == "best" else result["last_mean_r"]
            if abs(stats["delta_mean_r"] - (expected - result["fixed_C3_mean_r"])) > 1e-10:
                raise ValueError("Patient-bootstrap point estimate differs from official score")
            rows.append({"seed": seed, "checkpoint": checkpoint,
                "epoch": result["best_epoch"] if checkpoint == "best" else 10,
                "mean_r": expected, **stats})
            bootstrap[checkpoint].append(samples)
    aggregate = {}
    for checkpoint in ("best", "last"):
        selected = [r for r in rows if r["checkpoint"] == checkpoint]
        boot = np.stack(bootstrap[checkpoint]).mean(axis=0)
        scores = np.array([r["mean_r"] for r in selected])
        gains = np.array([r["delta_mean_r"] for r in selected])
        aggregate[checkpoint] = {"seeds": [42, 43, 44], "mean_r": float(scores.mean()),
            "std_r": float(scores.std(ddof=1)), "min_r": float(scores.min()), "max_r": float(scores.max()),
            "mean_gain_vs_fixed_C3": float(gains.mean()), "positive_gain_seeds": int((gains > 0).sum()),
            "paired_patient_CI95_low_for_seed_mean_gain": float(np.quantile(boot, .025)),
            "paired_patient_CI95_high_for_seed_mean_gain": float(np.quantile(boot, .975))}
    csv_write(out / "summary.csv", rows)
    write(out / "completed.json", {"completed": True, "aggregate": aggregate, "per_seed": rows,
        "scope": "Continuation seed stability conditional on the SAME trained C3 checkpoint; not full-pipeline retraining stability.",
        "bootstrap": "Paired patients; repeated pairs within target averaged; lead QC then equal 11-lead mean; body gain /4.",
        "bootstrap_limit": "Conditional validation uncertainty; does not correct repeated model selection or replace independent validation.",
        "source_checkpoint_sha256": features["initial_checkpoint_sha256"],
        "feature_manifest_sha256": sha(history / "features/features.json"),
        "historical_reference_recheck_deferred": True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    out, history = Path(args.output).resolve(), Path(args.history_run).resolve()
    if not args.summarize_only:
        if out.exists():
            raise FileExistsError("Use a new stability output")
        if not read(history / "completed.json")["completed"]:
            raise ValueError("Original history study not completed")
        out.mkdir(parents=True)
        write(out / "protocol.json", {"training_seeds": [42, 43, 44],
            "seed42_reused": str(history / "single"), "new_training_seeds": [43, 44],
            "selection": "Report ALL seeds; both best and fixed-last; never pick best seed",
            "fixed": "C3 initializer, single template, features, lr2e-4, 10epochs, batch16, FM noise42",
            "conditional_on_fixed_C3_checkpoint": True})
        for seed in (43, 44):
            subprocess.run([sys.executable, "-u", "-m", "baselines.B5_final.body_history_study", "train",
                "--features", str(history / "features"), "--output", str(out / f"seed{seed}"),
                "--representation", "single", "--seed", str(seed), "--device", args.device], check=True)
    summarize(args)


if __name__ == "__main__":
    main()
