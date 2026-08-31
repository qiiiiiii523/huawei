"""汇总任务一、任务二 V0 报告，计算正式比赛主分和 RMSE 加分。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.evaluate import competition_score, write_competition_score


def _read_first_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one overall-metrics row: {path}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 r1、r2 和任务二 RMSE 加分。")
    parser.add_argument("--task1-overall", required=True, help="任务一 overall_metrics.csv")
    parser.add_argument("--task2-overall", required=True, help="任务二 overall_metrics.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    task1 = _read_first_row(Path(args.task1_overall))
    task2 = _read_first_row(Path(args.task2_overall))
    if task1.get("task_id") != "task1" or task2.get("task_id") != "task2":
        raise ValueError("Input reports must be task1 and task2 V0 overall reports respectively")
    summary = competition_score(
        r1=float(task1["task1_r1"]),
        r2=float(task2["task2_r2"]),
        missing_lead_rmse_uV=float(task2["task2_missing_lead_mean_rmse_uV"]),
    )
    csv_path, markdown_path = write_competition_score(args.output_dir, summary)
    print("Wrote:", csv_path, markdown_path, sep="\n")


if __name__ == "__main__":
    main()
