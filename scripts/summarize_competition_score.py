"""旧60分协议审计入口；新best_r选模请使用summarize_mean_r.py。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.evaluate import (competition_score, write_competition_score,
                             TASK2_DEVICE_AGGREGATION, TARGET_RECORD_AGGREGATION,
                             COMPETITION_SCORE_PROTOCOL, QC_COMPETITION_SCORE_PROTOCOL,
                             QC_TASK2_DEVICE_AGGREGATION, TARGET_LEAD_QC_PROTOCOL)


def _read_first_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one overall-metrics row: {path}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="旧协议复核：r1/r2和RMSE加分；新选模使用summarize_mean_r.py。")
    parser.add_argument("--task1-overall", required=True, help="任务一 overall_metrics.csv")
    parser.add_argument("--task2-overall", required=True, help="任务二 overall_metrics.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    task1 = _read_first_row(Path(args.task1_overall))
    task2 = _read_first_row(Path(args.task2_overall))
    if task1.get("task_id") != "task1" or task2.get("task_id") != "task2":
        raise ValueError("Input reports must be task1 and task2 V0 overall reports respectively")
    if any(task.get("target_record_aggregation") != TARGET_RECORD_AGGREGATION for task in (task1, task2)):
        raise ValueError("Regenerate both reports using target-equal scoring; do not mix old pair-weighted reports")
    if task2.get("task2_rmse_scored_leads") != "V1,V2,V3,V4,V5,V6":
        raise ValueError("Task2 report must score V1--V6 RMSE; regenerate old 11-lead RMSE reports")
    qc = [task.get("target_lead_qc_protocol") for task in (task1, task2)]
    if qc not in ([None, None], [TARGET_LEAD_QC_PROTOCOL, TARGET_LEAD_QC_PROTOCOL]):
        raise ValueError("Do not mix full-lead and QC-filtered reports")
    use_qc = qc[0] is not None
    protocol = QC_COMPETITION_SCORE_PROTOCOL if use_qc else COMPETITION_SCORE_PROTOCOL
    if use_qc and any(task.get("competition_score_protocol") != protocol or
                      str(task.get("quality_mask_applied")).lower() != "true" for task in (task1, task2)):
        raise ValueError("QC reports must declare matching score protocol and applied quality mask")
    expected_device = QC_TASK2_DEVICE_AGGREGATION if use_qc else TASK2_DEVICE_AGGREGATION
    if task2.get("task2_device_aggregation") != expected_device:
        raise ValueError("Regenerate Task2 report: device-equal r and separately converted RMSE bonuses required")
    summary = competition_score(
        r1=float(task1["r_missing11"]),
        r2=float(task2["r_missing11"]),
        machine_rmse_uV=float(task2["ecg_machine_d6_task2_missing_lead_mean_rmse_uV"]),
        body_rmse_uV=float(task2["body_scale_d6_task2_missing_lead_mean_rmse_uV"]),
        score_protocol=protocol,
    )
    csv_path, markdown_path = write_competition_score(args.output_dir, summary)
    print("Wrote:", csv_path, markdown_path, sep="\n")


if __name__ == "__main__":
    main()
