"""Create the compact L0/L1 comparison table from completed run summaries."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from b1.data import ROOT


def main():
    out = ROOT / "results" / "b1"
    rows = []
    for variant in ("L0", "L1"):
        run_dir = out / f"B1_T1_A_{variant}"
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        best = max(summary["history"], key=lambda x: x["validation_task1_r1"])
        best_copy = max(summary["history"], key=lambda x: x["copy_at_eval_task1_r1"])
        best_eval = json.loads((run_dir / "evaluation" / f"epoch_{best['epoch']:03d}" / "summary.json").read_text(encoding="utf-8"))
        best_copy_eval = json.loads((run_dir / "evaluation" / f"epoch_{best_copy['epoch']:03d}" / "summary.json").read_text(encoding="utf-8"))
        summary["best_task1_r1"] = best["validation_task1_r1"]
        summary["best_raw_epoch"] = best["epoch"]
        summary["best_copy_at_eval_task1_r1"] = best_copy["copy_at_eval_task1_r1"]
        summary["best_copy_epoch"] = best_copy["epoch"]
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append({"experiment": f"B1_T1_A_{variant}", "epochs": len(summary["history"]),
                     "best_raw_epoch": best["epoch"], "best_raw_task1_r1": best["validation_task1_r1"],
                     "best_raw_rmse_uV": best_eval["raw"]["twelve_lead_mean_rmse_uV"],
                     "copy_at_best_raw_r1": best_eval["copy_at_eval"]["task1_r1"],
                     "copy_at_best_raw_rmse_uV": best_eval["copy_at_eval"]["twelve_lead_mean_rmse_uV"],
                     "best_copy_epoch": best_copy["epoch"], "best_copy_task1_r1": best_copy["copy_at_eval_task1_r1"],
                     "best_copy_rmse_uV": best_copy_eval["copy_at_eval"]["twelve_lead_mean_rmse_uV"],
                     "final_raw_task1_r1": summary["history"][-1]["validation_task1_r1"],
                     "final_copy_task1_r1": summary["history"][-1]["copy_at_eval_task1_r1"]})
    path = out / "B1_T1_A_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
