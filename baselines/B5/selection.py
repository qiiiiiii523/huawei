"""Independent checkpoint records for total points, correlation, and bonus."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from ecg12gen.evaluate import competition_score, COMPETITION_SCORE_PROTOCOL

SELECTION_PROTOCOL = "b5-total-correlation-bonus-v1"
METRIC_FILES = {"competition_total_score": "best.pt", "main_correlation": "best_r.pt",
                "task2_rmse_bonus_score": "best_bonus.pt"}


def empty_best_metrics() -> dict[str, Any]:
    return {key: None for key in METRIC_FILES}


def scores_from_summaries(summaries: dict[str, Any]) -> dict[str, Any]:
    if not {"task1", "task2"}.issubset(summaries):
        raise ValueError("Four-checkpoint selection requires both task1 and task2 validation")
    task2 = summaries["task2"]
    return competition_score(float(summaries["task1"]["r_missing11"]), float(task2["r_missing11"]),
        float(task2["ecg_machine_d6_task2_missing_lead_mean_rmse_uV"]),
        float(task2["body_scale_d6_task2_missing_lead_mean_rmse_uV"]))


def update_best_metrics(best: dict[str, Any], scores: dict[str, Any], epoch: int) -> tuple[dict[str, Any], list[str]]:
    if set(best) != set(METRIC_FILES) or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Invalid checkpoint selection state or epoch")
    values = {key: float(scores[key]) for key in METRIC_FILES}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("Nonfinite validation selection metric")
    result = copy.deepcopy(best)
    improved = []
    for key, value in values.items():
        if result[key] is None or value > result[key]["value"]:
            result[key] = {"value": value, "epoch": epoch}
            improved.append(key)
    return result, improved


def restore_best_metrics(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if (checkpoint.get("selection_protocol") != SELECTION_PROTOCOL or
            checkpoint.get("competition_score_protocol") != COMPETITION_SCORE_PROTOCOL or
            checkpoint.get("selection_metric") != "competition_total_score"):
        raise ValueError("Legacy score/selection state cannot be resumed; initialize weights in a new output directory")
    if checkpoint.get("checkpoint_role") != "last":
        raise ValueError("Exact resume requires last.pt; selected checkpoints can initialize a new run")
    best = checkpoint.get("best_metrics")
    if not isinstance(best, dict) or set(best) != set(METRIC_FILES):
        raise ValueError("Checkpoint lacks complete total/correlation/bonus best state")
    for entry in best.values():
        if entry is not None and (not isinstance(entry, dict) or not math.isfinite(float(entry.get("value", float('nan'))))
                or isinstance(entry.get("epoch"), bool) or not isinstance(entry.get("epoch"), int) or entry["epoch"] < 0):
            raise ValueError("Invalid best metric entry")
    expected = None if best["competition_total_score"] is None else best["competition_total_score"]["value"]
    if checkpoint.get("best_score") != expected:
        raise ValueError("Checkpoint total best_score disagrees with its best_metrics")
    return copy.deepcopy(best)


def save_selected_checkpoints(output: Path, payload: dict[str, Any], improved: list[str]) -> None:
    from .checkpoint import atomic_save
    if any(key not in METRIC_FILES for key in improved):
        raise ValueError("Unknown selected checkpoint metric")
    # Commit selected files before last: a failed save never commits a resume
    # ledger claiming that a missing selected file has already been saved.
    for key in improved:
        atomic_save(output / METRIC_FILES[key], {**payload, "checkpoint_role": key})
    atomic_save(output / "last.pt", {**payload, "checkpoint_role": "last"})
