"""Combine task reports for best_r selection, without RMSE bonus."""
from pathlib import Path
import argparse
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg12gen.evaluate import correlation_score_from_summaries, write_correlation_score


def read_row(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("Expected exactly one task summary row: " + str(path))
    return rows[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task1-overall", required=True)
    parser.add_argument("--task2-overall", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--legacy-target-qc-only", action="store_true",
                        help="Explicit historical comparison: QC but no reliable-input eligibility; separate protocol")
    args = parser.parse_args()
    summaries = {"task1": read_row(args.task1_overall), "task2": read_row(args.task2_overall)}
    score = correlation_score_from_summaries(summaries, require_reliable_input=not args.legacy_target_qc_only)
    print("Wrote:", *write_correlation_score(args.output_dir, score), sep="\n")


if __name__ == "__main__":
    main()
