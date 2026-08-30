"""V0: fixed validation-only ECG evaluation and CSV/Markdown reporting."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any
import numpy as np
from .contracts import D12_LEADS, ContractError, canonical_lead_mask

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

def write_report(output_dir: str | Path, overall: dict[str, Any], details: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    overall_csv, lead_csv, markdown = output / "overall_metrics.csv", output / "lead_metrics.csv", output / "report.md"
    with overall_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall)); writer.writeheader(); writer.writerow(overall)
    with lead_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    lines = ["# V0 validation report", "", "## Overall", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in overall.items())
    lines += ["", "## Per-lead metrics", "", "| Lead | Pearson r | RMSE (uV) | Input present |", "|---|---:|---:|:---:|"]
    lines.extend(f"| {row['lead']} | {row['pearson_r']:.6f} | {row['rmse_uV']:.6f} | {row['input_present']} |" for row in details)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall_csv, lead_csv, markdown

def _validation_metadata_ok(path: Path, expected_n: int) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "validation"]
    if len(rows) != expected_n:
        raise ContractError("Validation metadata count does not match prediction rows")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed V0 validation protocol.")
    parser.add_argument("--prediction", required=True, help="[N,12,5000] NPY model prediction")
    parser.add_argument("--target", required=True, help="[N,12,5000] validation target NPY")
    parser.add_argument("--metadata", required=True, help="Window metadata CSV; validation rows are required")
    parser.add_argument("--task-id", required=True, choices=("task1", "task2"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    prediction, target = np.load(args.prediction, mmap_mode="r"), np.load(args.target, mmap_mode="r")
    _validation_metadata_ok(Path(args.metadata), len(prediction))
    overall, details = evaluate_predictions(prediction, target, args.task_id)
    paths = write_report(args.output_dir, overall, details)
    print("Wrote:", *paths, sep="\n")

if __name__ == "__main__":
    main()
