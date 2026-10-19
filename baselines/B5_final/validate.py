"""Four context routes on one base prediction cache, with the final adapted body."""
from __future__ import annotations

import argparse
from pathlib import Path

from baselines.B5_best.common import write
from .profile import BASE_SHA, SAMPLING, add_asset_arguments, assets_from_args, check_payloads


def cases():
    return [("baseline", "none", None, None), ("d6_only", "fir25", None, None),
            ("body_only", "none", "body", None), ("both", "fir25", "body", None)]


def reference_check(profile, population, final_row):
    ref = profile["reference"]
    same_population = all(population[t][k] == v for t, fields in ref["population"].items()
                          for k, v in fields.items())
    differences = {k: final_row[k] - ref[k] for k in ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")}
    return {"same_target_population_and_QC": same_population,
            "matched": same_population and all(abs(v) <= ref["tolerance"] for v in differences.values()),
            "deltas_from_reported_reference": differences, "tolerance": ref["tolerance"]}


def check_baseline(report):
    if (report.get("checkpoint_sha256") != BASE_SHA or report.get("sampling") != SAMPLING
            or report.get("slow_beta") != .75):
        raise ValueError("Use the exact newFM-best45+P baseline, Heun16/K16/seed42/beta.75")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_asset_arguments(p)
    p.add_argument("--config")
    p.add_argument("--baseline-evaluation", help="Completed plain FM+P prediction cache, never a context output")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--verify-reference", action="store_true", help="Require original validation population and reported final r")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a new study directory; keep shell logs outside it")
    assets = assets_from_args(args)
    from baselines.B5.runtime import device_from_name
    from baselines.B5_best.bundle import load_bundle
    from baselines.B5_best_context.body_model import BodyAdapter
    from baselines.B5_best_context.body_alpha_core import FixedAlphaAdapter, SharedCorrection
    from baselines.B5_best_context.body_study import run_study
    from baselines.B5_best_context.evaluate import load_baseline, HistoryProvider
    baseline = args.baseline_evaluation or assets.baseline_evaluation
    loaded = load_baseline(baseline)
    check_baseline(loaded[0])
    body = BodyAdapter(assets.body, BASE_SHA, device_from_name(args.device), expected_group="context")
    check_payloads(load_bundle(assets.base), body.payload)
    provider = HistoryProvider(args.config or assets.validation_config, loaded[1]["task2"])
    try:
        rows = run_study(baseline, out, provider, None, None, loaded=loaded,
            case_specs=cases(), adapter_map={"body": FixedAlphaAdapter(SharedCorrection(body), 1.)},
            notes=["B5-final: newFM-best45+P, adapted body seed43-best3, d6 FIR25, alpha1.",
                   "Same base arrays/data/QC; body history is RAW, then filtered once by saved B policy.",
                   "No training, no target calibration, no per-record context coefficient search."])
        reference = reference_check(assets.profile, loaded[2], rows[-1])
        write(out / "final.json", {**assets.report(), "completed": True,
            "score_protocol": loaded[0]["score_protocol"], "baseline_evaluation": str(Path(baseline).resolve()),
            "population": loaded[2], "reference_check": reference, "summary": rows})
        if args.verify_reference and not reference["matched"]:
            raise ValueError("Final reference did not reproduce; retain final.json and check data/context/numerics")
    finally:
        provider.close()
        for item in loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()
    print("Completed: " + str(out / "summary.csv"), flush=True)


if __name__ == "__main__":
    main()
