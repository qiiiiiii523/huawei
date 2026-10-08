"""Raw-uV record evaluation. Only r_missing11; no preprocessing or I replacement."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
import numpy as np
from .contracts import D12_LEADS, ContractError

MISSING_11_LEAD_INDICES = np.arange(1, 12)
TASK2_RMSE_LEADS = D12_LEADS[6:12]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(np.float64, copy=False), y.astype(np.float64, copy=False)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator > 0 else float("nan")


def _record_groups(metadata_rows: list[dict[str, str]], expected_n: int) -> list[list[int]]:
    if len(metadata_rows) != expected_n:
        raise ContractError("Record metadata count does not match prediction rows")
    grouped: dict[str, list[tuple[int, int]]] = {}
    for index, row in enumerate(metadata_rows):
        sample_id = row.get("pair_id") or row.get("target_record_id")
        if not sample_id:
            raise ContractError("Record evaluation requires pair_id or target_record_id")
        try:
            start = int(row["start_sample_500hz"])
        except (KeyError, ValueError, TypeError) as error:
            raise ContractError("Record evaluation requires integer start_sample_500hz") from error
        grouped.setdefault(sample_id, []).append((start, index))
    groups = []
    for sample_id in sorted(grouped):
        ordered = sorted(grouped[sample_id])
        starts = [start for start, _ in ordered]
        if len(starts) != len(set(starts)):
            raise ContractError(f"Duplicate window start within validation sample {sample_id}")
        if starts[0] != 0 or any(b - a != 5000 for a, b in zip(starts, starts[1:])):
            raise ContractError(f"Validation sample {sample_id} is not contiguous: expected starts 0,5000,10000,...")
        if len({metadata_rows[i].get("target_record_id") for _, i in ordered}) > 1:
            raise ContractError(f"Validation sample {sample_id} mixes target records")
        try:
            expected = {int(metadata_rows[i]["expected_window_count"])
                        for _, i in ordered if metadata_rows[i].get("expected_window_count")}
        except (ValueError, TypeError) as error:
            raise ContractError("expected_window_count must be an integer") from error
        if len(expected) > 1 or (expected and len(ordered) != next(iter(expected))):
            raise ContractError(f"Validation sample {sample_id} has an inconsistent or incomplete window count")
        groups.append([i for _, i in ordered])
    return groups


def evaluate_record_predictions(prediction: np.ndarray, target: np.ndarray, task_id: str,
                                metadata_rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Stitch raw-uV windows, compute record r per lead, then average II--V6.

    RMSE remains point-weighted per lead, followed by a lead average.
    Task-2 bonus uses V1--V6 only; r1/r2 retain record-level r_missing11.
    Undefined constant-record correlations are excluded with explicit counts.
    """
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (12, 5000):
        raise ContractError("Record evaluation requires matching raw-uV [N,12,5000] arrays")
    if not len(prediction) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ContractError("Record evaluation requires non-empty finite predictions and targets")
    if task_id not in {"task1", "task2"}:
        raise ContractError("task_id must be task1 or task2")
    groups = _record_groups(metadata_rows, len(prediction))
    records = [(np.concatenate([prediction[i] for i in indices], axis=1),
                np.concatenate([target[i] for i in indices], axis=1)) for indices in groups]
    mean_correlations, details = [], []
    invalid_count = 0
    for lead in MISSING_11_LEAD_INDICES:
        correlations = np.asarray([_pearson(p[lead], t[lead]) for p, t in records])
        valid = np.isfinite(correlations)
        if not valid.any():
            raise ContractError(f"No defined record Pearson r for {D12_LEADS[lead]}")
        invalid_count += int((~valid).sum())
        mean_correlations.append(float(correlations[valid].mean()))
        squared_error = sum(float(np.sum((p[lead].astype(np.float64) - t[lead].astype(np.float64)) ** 2))
                            for p, t in records)
        point_count = sum(p.shape[1] for p, _ in records)
        details.append({"lead": D12_LEADS[lead], "rmse_uV": float(np.sqrt(squared_error / point_count)),
                        "n_records": len(records), "n_undefined_record_correlations": int((~valid).sum()),
                        "n_validation_points": point_count})
    rmse = float(np.mean([row["rmse_uV"] for row in details]))
    overall = {
        "split": "validation", "task_id": task_id, "n_windows": len(prediction), "n_records": len(records),
        "evaluation_view": "raw_uV", "prediction_unit": "uV",
        "evaluation_aggregation": "record_macro_after_chronological_window_stitch",
        "rmse_aggregation": "point_weighted_per_lead_then_lead_mean",
        "evaluation_input_contract": "joint_anchor_test_like", "checkpoint_selection_metric": "r_missing11",
        "scored_leads": ",".join(D12_LEADS[i] for i in MISSING_11_LEAD_INDICES),
        "r_missing11": float(np.mean(mean_correlations)), "missing11_mean_rmse_uV": rmse,
        "n_undefined_record_lead_correlations": invalid_count,
    }
    if task_id == "task2":
        overall["task2_missing_lead_mean_rmse_uV"] = float(np.mean(
            [row["rmse_uV"] for row in details if row["lead"] in TASK2_RMSE_LEADS]))
        overall["task2_rmse_scored_leads"] = ",".join(TASK2_RMSE_LEADS)
    return overall, details


def evaluate_joint_anchor_predictions(prediction_raw: np.ndarray, target: np.ndarray,
                                      anchor_i_ecg: np.ndarray, task_id: str,
                                      metadata_rows: list[dict[str, str]],
                                      target_quality_mask: np.ndarray | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """P0/P1 adapter returns (summary, RMSE details), without altering predictions.

    A supplied training QC mask is validated but does not delete scored leads.
    """
    prediction, anchor = np.asarray(prediction_raw), np.asarray(anchor_i_ecg)
    if prediction.ndim != 3 or anchor.shape != (len(prediction), 1, 5000) or not np.isfinite(anchor).all():
        raise ContractError("anchor_i_ecg must be finite [N,1,5000]")
    if target_quality_mask is not None and np.asarray(target_quality_mask).shape != prediction.shape[:2]:
        raise ContractError("target_quality_mask must have shape [N,12]")
    return evaluate_record_predictions(prediction, target, task_id, metadata_rows)


def competition_score(r1: float, r2: float, missing_lead_rmse_uV: float) -> dict[str, float]:
    if not np.isfinite([r1, r2, missing_lead_rmse_uV]).all() or missing_lead_rmse_uV < 0:
        raise ContractError("Scores must be finite and RMSE non-negative")
    main_score = 0.5 * r1 + 0.5 * r2
    bonus = 10.0 if missing_lead_rmse_uV <= 70.0 else 700.0 / missing_lead_rmse_uV
    return {"task1_r_missing11": float(r1), "task2_r_missing11": float(r2),
            "task2_missing_lead_mean_rmse_uV": float(missing_lead_rmse_uV),
            "main_score": main_score, "task2_rmse_bonus_score": bonus,
            "competition_total_score": main_score + bonus}


def write_competition_score(output_dir: str | Path, summary: dict[str, float]) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path, markdown = output / "competition_score.csv", output / "competition_score.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    lines = ["# 比赛总分", "", "| 指标 | 数值 |", "|---|---:|"]
    lines.extend(f"| {key} | {value:.6f} |" for key, value in summary.items())
    lines += ["", "主分 = 两任务 r_missing11 的平均；Task 2 V1–V6 平均 RMSE 加分 = 10（≤70 μV），否则为 700 / RMSE。"]
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown


def write_report(output_dir: str | Path, overall: dict[str, Any], details: list[dict[str, Any]],
                 title: str = "Record-level raw-uV r_missing11") -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    overall_csv, lead_csv, markdown = output / "overall_metrics.csv", output / "lead_metrics.csv", output / "report.md"
    for path, rows in ((overall_csv, [overall]), (lead_csv, details)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [f"# {title}", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in overall.items())
    lines += ["", "输入为已还原的 μV。评估不减 median、不乘 scale、不滤波、不替换 I。",
              "", "| Lead | RMSE (μV) |", "|---|---:|"]
    lines.extend(f"| {row['lead']} | {row['rmse_uV']:.6f} |" for row in details)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall_csv, lead_csv, markdown


def _validation_metadata_rows(path: Path, expected_n: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "validation"]
    if len(rows) != expected_n:
        raise ContractError("Validation metadata count does not match prediction rows")
    try:
        rows = sorted(rows, key=lambda row: int(row["array_index"]))
    except (KeyError, ValueError) as error:
        raise ContractError("Validation metadata requires integer array_index") from error
    if [int(row["array_index"]) for row in rows] != list(range(expected_n)):
        raise ContractError("Validation array_index must be contiguous from zero")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw-uV predictions by record; only r_missing11.")
    parser.add_argument("--prediction", required=True, help="Raw μV [N,12,5000] NPY, already multiplied by frozen d12 scale")
    parser.add_argument("--target", required=True, help="Original raw μV validation targets")
    parser.add_argument("--metadata", required=True, help="Window metadata CSV")
    parser.add_argument("--task-id", required=True, choices=("task1", "task2"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    prediction, target = np.load(args.prediction, mmap_mode="r"), np.load(args.target, mmap_mode="r")
    rows = _validation_metadata_rows(Path(args.metadata), len(prediction))
    overall, details = evaluate_record_predictions(prediction, target, args.task_id, rows)
    print("Wrote:", *write_report(args.output_dir, overall, details), sep="\n")


if __name__ == "__main__":
    main()
