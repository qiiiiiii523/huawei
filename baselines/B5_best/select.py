"""Rank full hybrid outputs by mean r; never use FM-alone r or old bonus."""
import argparse
from pathlib import Path

from . import SCORE_PROTOCOL
from .common import read, write, csv_write
from .scoring import correlation_score
from ecg12gen.evaluate import RELIABLE_INPUT_MEAN_R_SCORE_PROTOCOL, correlation_score as public_correlation_score


def select(evaluations, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty ranking output directory")
    reports = [read(Path(p) / "evaluation.json") for p in evaluations]
    if not reports:
        raise ValueError("No evaluations supplied")
    reference = reports[0]
    rows = []
    for path, report in zip(evaluations, reports):
        protocol = report.get("score_protocol")
        if not report.get("completed") or protocol not in {SCORE_PROTOCOL, RELIABLE_INPUT_MEAN_R_SCORE_PROTOCOL}:
            raise ValueError("Only completed B5-best full-output evaluations can be ranked")
        if (protocol != reference.get("score_protocol") or report["population"] != reference["population"] or
                report["sampling"] != reference["sampling"]):
            raise ValueError("Cannot rank different data/QC populations or sampling rules")
        score = public_correlation_score(report["score"]["task1_r_missing11"],
                                         report["score"]["task2_r_missing11"], score_protocol=protocol)
        if score != report["score"]:
            raise ValueError("Inconsistent mean-r score")
        rows.append({"evaluation": str(Path(path).resolve()), "checkpoint_sha256": report["checkpoint_sha256"], **score})
    rows.sort(key=lambda r: -r["main_correlation"])
    csv_write(output / "ranking.csv", rows)
    write(output / "selected.json", {"score_protocol": reference["score_protocol"], "selected": rows[0],
          "scope": "supplied_saved_complete_hybrid_candidates", "checkpoint_files_modified": False})
    print(rows[0], flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluations", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    select(args.evaluations, args.output)


if __name__ == "__main__":
    main()
