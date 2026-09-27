"""Compare B4 and M1 with main's shared validation evaluator."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.contracts import D12_LEADS
from ecg12gen.evaluate import evaluate_joint_anchor_predictions


def _evaluate(name: str, path: str, target: np.ndarray, anchor: np.ndarray, task_id: str) -> tuple[dict, list[dict]]:
    prediction = np.asarray(np.load(path), dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError(f"{name} prediction shape {prediction.shape} does not match target {target.shape}")
    summary, details, _, _ = evaluate_joint_anchor_predictions(prediction, target, anchor, task_id)
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b4-prediction", required=True)
    parser.add_argument("--m1-prediction", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    target = np.asarray(np.load(args.target), dtype=np.float32)
    anchor = np.asarray(np.load(args.anchor), dtype=np.float32)
    if target.ndim != 3 or target.shape[1:] != (12, 5000):
        raise ValueError("target must have shape [N,12,5000]")
    if anchor.shape != (len(target), 1, 5000):
        raise ValueError("anchor must have shape [N,1,5000]")
    results = {
        "B4": _evaluate("B4", args.b4_prediction, target, anchor, args.task_id),
        "M1": _evaluate("M1", args.m1_prediction, target, anchor, args.task_id),
    }
    rows = []
    for model_name, (summary, details) in results.items():
        row = {
            "model": model_name,
            "n_windows": len(target),
            "r_missing11": float(summary["r_missing11"]),
            "rmse_missing11_uV": float(summary["missing11_mean_rmse_uV"]),
        }
        for lead_index in range(6, 12):
            lead = D12_LEADS[lead_index]
            row[f"{lead}_r"] = float(details[lead_index]["pearson_r"])
            row[f"{lead}_rmse_uV"] = float(details[lead_index]["rmse_uV"])
        rows.append(row)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "b4_vs_m1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "b4_vs_m1.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    headers = ["Model", "r_missing11", "RMSE missing11 (uV)", *[f"{lead} r / RMSE" for lead in D12_LEADS[6:]]]
    lines = [
        "# B4 vs M1 on identical validation arrays",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        values = [
            row["model"],
            f"{row['r_missing11']:.6f}",
            f"{row['rmse_missing11_uV']:.6f}",
            *[f"{row[f'{lead}_r']:.6f} / {row[f'{lead}_rmse_uV']:.6f}" for lead in D12_LEADS[6:]],
        ]
        lines.append("| " + " | ".join(values) + " |")
    (output / "b4_vs_m1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote B4/M1 comparison to {output}")


if __name__ == "__main__":
    main()
