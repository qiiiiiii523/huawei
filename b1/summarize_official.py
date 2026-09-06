"""Create compact team-facing B1 official-v1 result tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from b1.official_v1 import DEFAULT_OUTPUT_ROOT, read_matrix


def _device_rows(path: Path, evaluation: str) -> list[dict[str, str]]:
    csv_path = path / "evaluation" / evaluation / "task2_device_metrics.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = DEFAULT_OUTPUT_ROOT
    output = root / "delivery"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for spec in read_matrix():
        run = root / "experiments" / spec.id
        summary_path = run / "run_summary.json"
        if not summary_path.exists():
            incomplete.append(spec.id); continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            incomplete.append(spec.id); continue
        metrics = summary["best_metrics"]
        primary = "task1_r1" if spec.task == "task1" else "task2_r2"
        rows.append({
            "experiment": spec.id, "task": spec.task, "route": spec.route,
            "data_variant": spec.data_variant, "loss_variant": spec.loss_variant or "fixed_rpeak_pseudo",
            "best_epoch": summary["best_epoch"], "E1_official_r": metrics["E1"][primary],
            "E1_mean_rmse_uV": metrics["E1"]["twelve_lead_mean_rmse_uV"],
            "E2_copy_r": metrics["E2"][primary], "E2_copy_mean_rmse_uV": metrics["E2"]["twelve_lead_mean_rmse_uV"],
            "centered_diagnostic_r": metrics["centered"][primary],
            "task2_missing_v1_v6_rmse_uV": metrics["E1"].get("task2_missing_lead_mean_rmse_uV")})
        if spec.task == "task2":
            for view in ("E1_raw", "E2_copy_at_eval"):
                for item in _device_rows(run, view):
                    if item["scope"] == "device" and item["aggregation"] == "subject_macro":
                        device_rows.append({"experiment": spec.id, "evaluation": view, **item})
    if incomplete:
        raise SystemExit(f"Cannot create final delivery; incomplete experiments: {', '.join(incomplete)}")
    for filename, values in (("B1_OFFICIAL_V1_SUMMARY.csv", rows), ("B1_TASK2_DEVICE_SUMMARY.csv", device_rows)):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    best_task1 = max((row for row in rows if row["task"] == "task1"), key=lambda row: float(row["E1_official_r"]))
    best_task2 = max((row for row in rows if row["task"] == "task2"), key=lambda row: float(row["E1_official_r"]))
    report = ["# B1 official-v1 delivery summary", "",
              "All 14 runs use the shared main protocol, seed 42, 100 epochs per stage, and official raw-uV V0 checkpoint selection.", "",
              f"- Best task1: `{best_task1['experiment']}`, r1={float(best_task1['E1_official_r']):.6f}",
              f"- Best task2: `{best_task2['experiment']}`, r2={float(best_task2['E1_official_r']):.6f}",
              "- E2 is copy-at-eval and does not retrain the model.",
              "- Centered values are diagnostic and do not replace official V0.", "",
              "See `B1_OFFICIAL_V1_SUMMARY.csv` and `B1_TASK2_DEVICE_SUMMARY.csv` for complete results."]
    (output / "B1_OFFICIAL_V1_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
