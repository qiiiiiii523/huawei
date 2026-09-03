"""V0: fixed validation-only ECG evaluation and CSV/Markdown reporting."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any
import numpy as np
from .contracts import D12_LEADS, ContractError, canonical_lead_mask

TASK2_GENERATED_LEAD_INDICES = np.arange(6, 12)
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(np.float64, copy=False), y.astype(np.float64, copy=False)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator > 0 else float("nan")

def evaluate_predictions(prediction: np.ndarray, target: np.ndarray, task_id: str,
                         lead_mask: np.ndarray | None = None) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    """Run V0 on validation arrays.

    Per-lead measures flatten all validation windows and points. r1/r2 are the
    unweighted mean of twelve per-lead Pearson correlations. Task-2 missing RMSE
    averages the absent input leads (normally V1--V6).
    """
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 12 or prediction.shape[2] != 5000:
        raise ContractError("prediction and target must both have shape [N, 12, 5000]")
    if task_id not in {"task1", "task2"}:
        raise ContractError("task_id must be task1 or task2")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ContractError("V0 evaluation requires finite prediction and target values")
    if lead_mask is None:
        mask = np.broadcast_to(canonical_lead_mask(1 if task_id == "task1" else 6), (prediction.shape[0], 12))
    else:
        mask = np.asarray(lead_mask, dtype=bool)
        if mask.shape == (12,):
            mask = np.broadcast_to(mask, (prediction.shape[0], 12))
        if mask.shape != (prediction.shape[0], 12):
            raise ContractError("lead_mask must have shape [12] or [N, 12]")
    details: list[dict[str, float | str]] = []
    for lead_index, lead_name in enumerate(D12_LEADS):
        pred, truth = prediction[:, lead_index, :].reshape(-1), target[:, lead_index, :].reshape(-1)
        details.append({"lead": lead_name, "pearson_r": _pearson(pred, truth),
                        "rmse_uV": float(np.sqrt(np.mean((pred.astype(np.float64) - truth.astype(np.float64)) ** 2))),
                        "input_present": bool(mask[:, lead_index].all()), "n_validation_points": int(pred.size)})
    correlations = np.asarray([float(row["pearson_r"]) for row in details])
    rmses = np.asarray([float(row["rmse_uV"]) for row in details])
    missing_leads = ~mask.all(axis=0)
    mean_r = float(np.nanmean(correlations))
    overall: dict[str, float | str] = {
        "split": "validation", "task_id": task_id, "n_windows": int(prediction.shape[0]),
        "twelve_lead_mean_pearson_r": mean_r, "twelve_lead_mean_rmse_uV": float(np.mean(rmses)),
        "task1_r1": mean_r if task_id == "task1" else float("nan"),
        "task2_r2": mean_r if task_id == "task2" else float("nan"),
        "task2_missing_lead_mean_rmse_uV": float(np.mean(rmses[missing_leads])) if task_id == "task2" else float("nan")}
    return overall, details
def _center_per_window_uV(values: np.ndarray) -> np.ndarray:
    """Remove each [lead, window] median for a morphology-only diagnostic."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (12, 5000):
        raise ContractError("Centered diagnostic requires [N, 12, 5000] arrays in μV")
    return array - np.median(array, axis=2, keepdims=True)


def evaluate_centered_diagnostic(prediction_uV: np.ndarray, target_uV: np.ndarray, task_id: str,
                                 lead_mask: np.ndarray | None = None) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    """Evaluate morphology after independent per-window median centering.

    This is a diagnostic view only. Official r1/r2/RMSE must always be
    computed by ``evaluate_predictions`` on raw-μV predictions and raw-μV
    targets. Callers must invert the frozen d12 scale before using it.
    """
    centered_prediction = _center_per_window_uV(prediction_uV)
    centered_target = _center_per_window_uV(target_uV)
    overall, details = evaluate_predictions(centered_prediction, centered_target, task_id, lead_mask)
    overall = {**overall, "evaluation_view": "centered_diagnostic_not_official"}
    return overall, details

def _summary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Summarize a task-2 subset without changing the official V0 metric."""
    _, details = evaluate_predictions(prediction, target, "task2")
    correlations = np.asarray([float(row["pearson_r"]) for row in details])
    rmses = np.asarray([float(row["rmse_uV"]) for row in details])
    generated_r = correlations[TASK2_GENERATED_LEAD_INDICES]
    generated_rmse = rmses[TASK2_GENERATED_LEAD_INDICES]
    return {
        "twelve_lead_mean_pearson_r": float(np.nanmean(correlations)),
        "twelve_lead_mean_rmse_uV": float(np.mean(rmses)),
        "generated_v1_v6_mean_pearson_r": float(np.nanmean(generated_r)),
        "generated_v1_v6_mean_rmse_uV": float(np.mean(generated_rmse)),
    }

def _mean_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "twelve_lead_mean_pearson_r", "twelve_lead_mean_rmse_uV",
        "generated_v1_v6_mean_pearson_r", "generated_v1_v6_mean_rmse_uV",
    )
    return {key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys}

def evaluate_task2_diagnostics(prediction: np.ndarray, target: np.ndarray,
                               metadata_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return subject-macro and device-stratified task-2 diagnostic metrics.

    The official V0 result remains point-weighted over all validation windows.
    These supplementary rows weight every subject equally and expose V1--V6,
    which are the leads task 2 actually needs to generate.
    """
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (12, 5000):
        raise ContractError("Task-2 diagnostic inputs must both have shape [N, 12, 5000]")
    if len(metadata_rows) != len(prediction):
        raise ContractError("Task-2 diagnostic metadata count does not match prediction rows")
    required_fields = {"subject_id", "input_type"}
    if any(not required_fields.issubset(row) or not row["subject_id"] or not row["input_type"] for row in metadata_rows):
        raise ContractError("Task-2 diagnostics require non-empty subject_id and input_type metadata")

    indexed_rows = list(enumerate(metadata_rows))
    subject_rows: list[dict[str, Any]] = []
    for subject_id in sorted({row["subject_id"] for row in metadata_rows}):
        indices = [index for index, row in indexed_rows if row["subject_id"] == subject_id]
        subject_rows.append({"subject_id": subject_id, "n_windows": len(indices),
                             **_summary_metrics(prediction[indices], target[indices])})

    subject_device_rows: list[dict[str, Any]] = []
    for subject_id, input_type in sorted({(row["subject_id"], row["input_type"]) for row in metadata_rows}):
        indices = [index for index, row in indexed_rows
                   if row["subject_id"] == subject_id and row["input_type"] == input_type]
        subject_device_rows.append({"subject_id": subject_id, "input_type": input_type, "n_windows": len(indices),
                                    **_summary_metrics(prediction[indices], target[indices])})

    device_rows: list[dict[str, Any]] = [{
        "scope": "all_task2", "input_type": "all", "aggregation": "subject_macro",
        "n_subjects": len(subject_rows), "n_windows": len(prediction), **_mean_metric_rows(subject_rows),
    }]
    for input_type in sorted({row["input_type"] for row in metadata_rows}):
        indices = [index for index, row in indexed_rows if row["input_type"] == input_type]
        per_subject = [row for row in subject_device_rows if row["input_type"] == input_type]
        device_rows.append({"scope": "device", "input_type": input_type, "aggregation": "pooled_windows",
                            "n_subjects": len(per_subject), "n_windows": len(indices),
                            **_summary_metrics(prediction[indices], target[indices])})
        device_rows.append({"scope": "device", "input_type": input_type, "aggregation": "subject_macro",
                            "n_subjects": len(per_subject), "n_windows": len(indices), **_mean_metric_rows(per_subject)})
    return subject_rows, device_rows

def competition_score(r1: float, r2: float, missing_lead_rmse_uV: float) -> dict[str, float]:
    """Compute the official combined competition score from both V0 tasks.

    Main score = 0.5 * r1 + 0.5 * r2.
    Task-2 bonus is 10 when missing-lead RMSE <= 70 μV, otherwise 700 / RMSE.
    """
    values = np.asarray([r1, r2, missing_lead_rmse_uV], dtype=np.float64)
    if not np.isfinite(values).all() or missing_lead_rmse_uV < 0:
        raise ContractError("r1, r2, and missing-lead RMSE must be finite; RMSE must be non-negative")
    main_score = 0.5 * r1 + 0.5 * r2
    bonus_score = 10.0 if missing_lead_rmse_uV <= 70.0 else 700.0 / missing_lead_rmse_uV
    return {
        "task1_r1": float(r1),
        "task2_r2": float(r2),
        "task2_missing_lead_mean_rmse_uV": float(missing_lead_rmse_uV),
        "main_score": float(main_score),
        "task2_rmse_bonus_score": float(bonus_score),
        "competition_total_score": float(main_score + bonus_score),
    }


def write_competition_score(output_dir: str | Path, summary: dict[str, float]) -> tuple[Path, Path]:
    """Write the cross-task official-score summary as CSV and Markdown."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path, markdown_path = output / "competition_score.csv", output / "competition_score.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    lines = [
        "# 比赛总分汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        *[f"| {key} | {value:.6f} |" for key, value in summary.items()],
        "",
        "主分 = 0.5 × r1 + 0.5 × r2。",
        "",
        "加分：缺失导联平均 RMSE ≤ 70 μV 时为 10 分，否则为 700 / RMSE。",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path

def write_report(output_dir: str | Path, overall: dict[str, Any], details: list[dict[str, Any]], title: str = "V0 validation report") -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    overall_csv, lead_csv, markdown = output / "overall_metrics.csv", output / "lead_metrics.csv", output / "report.md"
    with overall_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall)); writer.writeheader(); writer.writerow(overall)
    with lead_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    lines = [f"# {title}", "", "## Overall", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in overall.items())
    lines += ["", "## Per-lead metrics", "", "| Lead | Pearson r | RMSE (uV) | Input present |", "|---|---:|---:|:---:|"]
    lines.extend(f"| {row['lead']} | {row['pearson_r']:.6f} | {row['rmse_uV']:.6f} | {row['input_present']} |" for row in details)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall_csv, lead_csv, markdown

def write_task2_diagnostics(output_dir: str | Path, subject_rows: list[dict[str, Any]],
                            device_rows: list[dict[str, Any]], report_path: str | Path) -> tuple[Path, Path]:
    """Write supplementary task-2 diagnostics without changing V0 primary files."""
    output = Path(output_dir)
    subject_csv = output / "task2_subject_metrics.csv"
    device_csv = output / "task2_device_metrics.csv"
    with subject_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0]))
        writer.writeheader()
        writer.writerows(subject_rows)
    with device_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(device_rows[0]))
        writer.writeheader()
        writer.writerows(device_rows)
    lines = [
        "", "## Task-2 supplementary diagnostics", "",
        "These rows do not replace the official point-weighted V0 `task2_r2`.",
        "They report subject-macro metrics and V1--V6, the leads task 2 must generate.",
        "", "| Scope | Input device | Aggregation | Subjects | Windows | 12-lead r | V1--V6 r | V1--V6 RMSE (uV) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['scope']} | {row['input_type']} | {row['aggregation']} | {row['n_subjects']} | {row['n_windows']} | "
        f"{row['twelve_lead_mean_pearson_r']:.6f} | {row['generated_v1_v6_mean_pearson_r']:.6f} | "
        f"{row['generated_v1_v6_mean_rmse_uV']:.6f} |"
        for row in device_rows
    )
    with Path(report_path).open("a", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
    return subject_csv, device_csv

def _validation_metadata_rows(path: Path, expected_n: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "validation"]
    if len(rows) != expected_n:
        raise ContractError("Validation metadata count does not match prediction rows")
    try:
        rows = sorted(rows, key=lambda row: int(row["array_index"]))
    except (KeyError, ValueError) as error:
        raise ContractError("Validation metadata requires integer array_index values") from error
    if [int(row["array_index"]) for row in rows] != list(range(expected_n)):
        raise ContractError("Validation metadata array_index values must be contiguous from zero")
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed V0 validation protocol.")
    parser.add_argument("--prediction", required=True, help="[N,12,5000] NPY model prediction")
    parser.add_argument("--target", required=True, help="[N,12,5000] validation target NPY")
    parser.add_argument("--metadata", required=True, help="Window metadata CSV; validation rows are required")
    parser.add_argument("--task-id", required=True, choices=("task1", "task2"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-centered-diagnostic", action="store_true", help="Also write a non-official per-window-centered morphology report.")
    args = parser.parse_args()
    prediction, target = np.load(args.prediction, mmap_mode="r"), np.load(args.target, mmap_mode="r")
    metadata_rows = _validation_metadata_rows(Path(args.metadata), len(prediction))
    overall, details = evaluate_predictions(prediction, target, args.task_id)
    paths = list(write_report(args.output_dir, overall, details))
    if args.task_id == "task2":
        subject_rows, device_rows = evaluate_task2_diagnostics(prediction, target, metadata_rows)
        paths.extend(write_task2_diagnostics(args.output_dir, subject_rows, device_rows, paths[-1]))
    if args.write_centered_diagnostic:
        centered_prediction = _center_per_window_uV(prediction)
        centered_target = _center_per_window_uV(target)
        centered_overall, centered_details = evaluate_centered_diagnostic(centered_prediction, centered_target, args.task_id)
        centered_dir = Path(args.output_dir) / "centered_diagnostic"
        centered_paths = list(write_report(centered_dir, centered_overall, centered_details, title="Centered morphology diagnostic (not official)"))
        if args.task_id == "task2":
            subject_rows, device_rows = evaluate_task2_diagnostics(centered_prediction, centered_target, metadata_rows)
            centered_paths.extend(write_task2_diagnostics(centered_dir, subject_rows, device_rows, centered_paths[-1]))
        paths.extend(centered_paths)
    print("Wrote:", *paths, sep="\n")

if __name__ == "__main__":
    main()
