"""Frozen body output-scope ablation; no training and no target-conditioned transforms."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from baselines.B5_best_context.fir import B, C

# All groups keep machine FIR25 and Task1 fixed. First two are projection controls.
CASES = (
    ("R0_no_projection_no_body", False, "none"),
    ("R1_projection_no_body", True, "none"),
    ("C0_no_projection_II", False, "II"),
    ("C0_five_no_projection", False, "five"),
    ("C2_projection_II", True, "II"),
    ("C3_projection_five", True, "five"),
)


def transform(base, observed_i, corrected_ii, *, projection, scope):
    """Reuse ORIGINAL-base II correction, never feed projected II to the adapter."""
    base, anchor = np.asarray(base), np.asarray(observed_i)
    corrected_ii = np.asarray(corrected_ii)
    if (base.ndim != 2 or base.shape[0] != 12 or base.dtype.kind != "f"
            or anchor.shape not in ((base.shape[1],), (1, base.shape[1]))
            or corrected_ii.shape != (base.shape[1],)
            or not all(np.isfinite(v).all() for v in (base, anchor, corrected_ii))):
        raise ValueError("Expected finite floating base[12,T], visible I[T], corrected II[T]")
    if scope not in ("none", "II", "five"):
        raise ValueError("scope must be none, II or five")
    out = base.astype(np.float64, copy=True)
    if projection:
        anchor = anchor.reshape(1, -1)
        q = (B * (out[1:6] - C * anchor)).sum(axis=0) / float((B * B).sum())
        # Match the existing projection's float32 boundary before adding body.
        out[1:6] = (B * q + C * anchor).astype(base.dtype)
    delta = corrected_ii.astype(np.float64) - base[1].astype(np.float64)
    if scope == "II":
        if not projection:
            out[1] = corrected_ii  # Exact deployed C0, including float rounding.
        else:
            out[1] += delta
    elif scope == "five":
        out[1:6] += B * delta
    return out.astype(base.dtype)


def main():
    from baselines.B5_final.profile import (BASE_SHA, add_asset_arguments,
        assets_from_args, check_payloads)
    from baselines.B5_final.validate import check_baseline, reference_check
    from baselines.B5_best.common import write, csv_write
    from baselines.B5.runtime import device_from_name
    from baselines.B5_best.bundle import load_bundle
    from baselines.B5_best_context.body_model import BodyAdapter
    from baselines.B5_best_context.adapters import ContextRouter
    from baselines.B5_best_context.evaluate import load_baseline, HistoryProvider, _score, _record_metrics
    from baselines.B5.scoring import scores_from_summaries, write_score_report
    from ecg12gen.evaluate import write_report

    p = argparse.ArgumentParser(description=__doc__)
    add_asset_arguments(p)
    p.add_argument("--config", help="Validation config with actual server data paths")
    p.add_argument("--baseline-evaluation", required=True, help="Plain FM+P cache, never context predictions")
    p.add_argument("--output", required=True, help="New directory; six full prediction caches are written")
    p.add_argument("--device", default="cuda")
    p.add_argument("--verify-reference", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a new output directory")
    assets = assets_from_args(args)
    loaded = load_baseline(args.baseline_evaluation)
    provider = None
    try:
        baseline, data, population = loaded
        check_baseline(baseline)
        body = BodyAdapter(assets.body, BASE_SHA, device_from_name(args.device), expected_group="context")
        check_payloads(load_bundle(assets.base), body.payload)
        provider = HistoryProvider(args.config or assets.validation_config, data["task2"])
        machine = ContextRouter(machine_mode="fir25")
        out.mkdir(parents=True)
        write(out / "protocol.json", {**assets.report(), "cases": CASES,
            "baseline_evaluation": str(Path(args.baseline_evaluation).resolve()),
            "actual_validation_config": str(Path(args.config or assets.validation_config).resolve()),
            "body_input": "original base II, visible I, raw personal history",
            "projection": "existing fixed-I least-squares projection using all five predicted limb leads",
            "propagation_coefficients": B[:, 0].tolist(), "population": population})
        arrays, metrics, statuses = {}, {}, {name: [] for name, _, _ in CASES}
        for name, _, _ in CASES:
            arrays[name], metrics[name] = {}, []
            for task, item in data.items():
                folder = out / name / task
                folder.mkdir(parents=True)
                arrays[name][task] = np.lib.format.open_memmap(folder / "prediction_uV.npy",
                    mode="w+", dtype=np.float32, shape=item["prediction"].shape)
        forward_calls = body_records = 0
        for task, item in data.items():
            for indices in item["pairs"]:
                row = item["rows"][indices[0]]
                base = np.concatenate([item["prediction"][i] for i in indices], axis=-1)
                # Only target I is exposed here: it is the protocol's observed input.
                anchor = np.concatenate([item["target"][i, :1] for i in indices], axis=-1)
                is_body = task == "task2" and row["input_type"] == "body_scale_d6"
                if is_body:
                    corrected, status = body(base.copy(), anchor, provider(row))
                    forward_calls += 1
                    body_records += 1
                    if not np.array_equal(corrected[[0, *range(2, 12)]], base[[0, *range(2, 12)]]):
                        raise AssertionError("Original body adapter changed another lead")
                else:
                    history = provider(row) if task == "task2" else None
                    fixed, status = machine.apply(base, anchor, task_id=task,
                        input_type=row["input_type"], history=history)
                for name, projection, scope in CASES:
                    result = (transform(base, anchor, corrected[1], projection=projection, scope=scope)
                              if is_body else fixed)
                    preserved = [0, *range(6, 12)]
                    if not np.array_equal(result[preserved], base[preserved]):
                        raise AssertionError("I/chest lead changed")
                    if task == "task1" and not np.array_equal(result, base):
                        raise AssertionError("Task1 changed")
                    for j, index in enumerate(indices):
                        arrays[name][task][index] = result[:, j * 5000:(j + 1) * 5000]
                    truth = np.concatenate([item["target"][i] for i in indices], axis=-1)
                    metrics[name].append(_record_metrics(result, truth, item["quality"][indices[0]], row, task))
                    statuses[name].append({"task_id": task, "pair_id": row["pair_id"],
                        "input_type": row["input_type"], "body_projection": bool(is_body and projection),
                        "body_scope": scope if is_body else "fixed_route", **status})
        if not body_records:
            raise ValueError("No body records in validation population")
        summaries = []
        for name, _, _ in CASES:
            tasks = {}
            for task, item in data.items():
                prediction = arrays[name][task]
                prediction.flush()
                overall, details = _score(prediction, item, task)
                tasks[task] = overall
                folder = out / name / task
                write_report(folder, overall, details, title=f"Body scope {name} {task}")
                csv_write(folder / "window_metadata.csv", item["rows"])
            score = scores_from_summaries(tasks)
            write_score_report(out / name, score)
            csv_write(out / name / "pair_metrics.csv", metrics[name])
            keys = list(dict.fromkeys(k for r in statuses[name] for k in r))
            csv_write(out / name / "postprocess_records.csv",
                [{k: r.get(k) for k in keys} for r in statuses[name]])
            summaries.append({"group": name, "task1_r": score["task1_r_missing11"],
                "machine_r": tasks["task2"]["ecg_machine_d6_r_missing11"],
                "body_r": tasks["task2"]["body_scale_d6_r_missing11"],
                "task2_r": score["task2_r_missing11"], "mean_r": score["main_correlation"]})
        by_name = {r["group"]: r for r in summaries}
        c0 = by_name["C0_no_projection_II"]
        for row in summaries:
            row["delta_mean_r_vs_C0"] = row["mean_r"] - c0["mean_r"]
            for key in ("task1_r", "machine_r"):
                if row[key] != c0[key]:
                    raise AssertionError("Fixed route scores differ")
        reference = reference_check(assets.profile, population, c0)
        comparisons = [("projection_without_body", CASES[1][0], CASES[0][0]),
            ("projection_with_II", CASES[4][0], CASES[2][0]),
            ("projection_with_five", CASES[5][0], CASES[3][0]),
            ("expand_without_projection", CASES[3][0], CASES[2][0]),
            ("expand_with_projection", CASES[5][0], CASES[4][0])]
        contrasts = [{"comparison": label, "candidate": a, "control": b,
            **{"delta_" + key: by_name[a][key] - by_name[b][key]
               for key in ("body_r", "task2_r", "mean_r")}} for label, a, b in comparisons]
        csv_write(out / "summary.csv", summaries)
        csv_write(out / "contrasts.csv", contrasts)
        write(out / "final.json", {"completed": True, "training_started": False,
            "summary": summaries, "contrasts": contrasts, "reference_check": reference,
            "body_forward_calls": forward_calls, "body_records": body_records,
            "one_body_forward_per_record_shared_across_all_cases": True})
        if args.verify_reference and not reference["matched"]:
            raise ValueError("C0 failed original-population reference check; reports retained")
        print("Completed: " + str(out / "summary.csv"), flush=True)
    finally:
        for task_arrays in locals().get("arrays", {}).values():
            for arr in task_arrays.values():
                arr._mmap.close()
        if provider is not None:
            provider.close()
        for item in loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()


if __name__ == "__main__":
    main()
