"""Only the unchanged QC record-level r1/r2 enter selection; no RMSE bonus."""
from __future__ import annotations

import math
from pathlib import Path

from ecg12gen.evaluate import (QC_COMPETITION_SCORE_PROTOCOL, TARGET_LEAD_QC_PROTOCOL,
                              QC_TASK2_DEVICE_AGGREGATION)
from . import SCORE_PROTOCOL
from .common import csv_write, write


def correlation_score(r1, r2):
    values = (float(r1), float(r2))
    if not all(math.isfinite(v) and -1.000001 <= v <= 1.000001 for v in values):
        raise FloatingPointError("Task correlations must be finite and within [-1,1]")
    return {"task1_r_missing11": values[0], "task2_r_missing11": values[1],
            "main_correlation": sum(values) / 2, "score_protocol": SCORE_PROTOCOL,
            "selection_metric": "main_correlation"}


def scores_from_summaries(summaries):
    if not {"task1", "task2"}.issubset(summaries):
        raise ValueError("Mean-r selection requires both task1 and task2")
    for name in ("task1", "task2"):
        s = summaries[name]
        if (s.get("competition_score_protocol") != QC_COMPETITION_SCORE_PROTOCOL or
                s.get("target_lead_qc_protocol") != TARGET_LEAD_QC_PROTOCOL or
                s.get("quality_mask_applied") is not True):
            raise ValueError("Both tasks require the unchanged explicit target-lead QC protocol")
    if summaries["task2"].get("task2_device_aggregation") != QC_TASK2_DEVICE_AGGREGATION:
        raise ValueError("Task2 must retain equal device aggregation")
    return correlation_score(summaries["task1"]["r_missing11"], summaries["task2"]["r_missing11"])


def write_score_report(output, score):
    if score.get("score_protocol") != SCORE_PROTOCOL:
        raise ValueError("Cannot write legacy total/bonus as the new mean-r score")
    output = Path(output)
    csv_write(output / "correlation_score.csv", [score])
    write(output / "correlation_score.json", score)
    text = ("# B5-best 两任务相关性\n\n"
            "选模仅使用 (Task1 r + Task2 r)/2；RMSE不算分。\n\n"
            "|Task1 r|Task2 r|平均r|\n|---:|---:|---:|\n"
            f"|{score['task1_r_missing11']:.8f}|{score['task2_r_missing11']:.8f}|{score['main_correlation']:.8f}|\n")
    (output / "correlation_score.md").write_text(text, encoding="utf-8")


def diagnostic_summary(summary):
    """Keep error diagnostics but never present the retired bonus as a score."""
    result = {k: v for k, v in summary.items() if "bonus" not in k and k != "competition_total_score"}
    result.update(score_protocol=SCORE_PROTOCOL, checkpoint_selection_metric="main_correlation")
    # competition_score_protocol describes the inherited QC aggregation, not a new total.
    return result
