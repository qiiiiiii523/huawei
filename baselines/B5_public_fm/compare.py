"""Summarize both completed arms and the stable baseline under one score population."""
import argparse
from pathlib import Path

from .common import read, write, csv_write, load_study, sha


def summarize(study_path, control, expanded, baseline, output):
    _, study = load_study(study_path, verify_data=False)
    reports = [("stable_baseline", read(Path(baseline) / "evaluation.json"))]
    for name, root in (("FM-C", control), ("FM-P", expanded)):
        root = Path(root) / "huawei"
        completed = read(root / "completed.json")
        if not completed.get("completed") or completed["study_sha256"] != sha(study_path):
            raise ValueError("Both arms must complete before comparison")
        for candidate, epoch in (("best", completed["best"]["epoch"]), ("last", completed["epochs"])):
            report = read(root / "evaluations" / f"epoch_{epoch:04d}" / "evaluation.json")
            report["epoch"] = epoch
            reports.append((name + "_" + candidate, report))
    reference = reports[0][1]
    rows = []
    for name, report in reports:
        if (not report.get("completed") or report["study_sha256"] != sha(study_path) or report["population"] != reference["population"] or
                report["score_protocol"] != reference["score_protocol"] or report["sampling"] != reference["sampling"] or
                report["fixed_slow_state_sha256"] != reference["fixed_slow_state_sha256"] or report["slow_beta"] != reference["slow_beta"]):
            raise ValueError("Cannot compare different data/QC/slow/sampling protocols")
        rows.append({"candidate": name, "epoch": report.get("epoch", ""), **report["score"],
                     "delta_vs_stable": report["score"]["main_correlation"] - reference["score"]["main_correlation"]})
    output = Path(output)
    csv_write(output / "summary.csv", rows)
    c = next(r for r in rows if r["candidate"] == "FM-C_best")
    p = next(r for r in rows if r["candidate"] == "FM-P_best")
    write(output / "comparison.json", {"completed": True, "rows": rows, "expanded_minus_control": p["main_correlation"] - c["main_correlation"],
          "score_protocol": study["score_protocol"], "note": "One paired training seed, not a stability or significance claim"})
    print((output / "summary.csv").read_text(encoding="utf-8"), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--expanded", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    summarize(args.study, args.control, args.expanded, args.baseline, args.output)


if __name__ == "__main__":
    main()
