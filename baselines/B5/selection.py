"""B5-best: only best_r.pt and last.pt; legacy resume scores are rejected."""
from __future__ import annotations
import copy
import math
from pathlib import Path
from .scoring import SCORE_PROTOCOL, scores_from_summaries

SELECTION_PROTOCOL = "b5-mean-r-reliable-input-v2"
METRIC_FILES = {"main_correlation": "best_r.pt"}


def empty_best_metrics():
    return {"main_correlation": None}


def update_best_metrics(best, scores, epoch):
    if set(best) != set(METRIC_FILES) or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Invalid best-r state or epoch")
    if scores.get("score_protocol") != SCORE_PROTOCOL:
        raise ValueError("Cannot mix old total/bonus scores with mean-r selection")
    value = float(scores["main_correlation"])
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite mean r")
    result = copy.deepcopy(best)
    if result["main_correlation"] is None or value > result["main_correlation"]["value"]:
        result["main_correlation"] = {"value": value, "epoch": epoch}
        return result, ["main_correlation"]
    return result, []


def restore_best_metrics(checkpoint):
    if (checkpoint.get("selection_protocol") != SELECTION_PROTOCOL or
            checkpoint.get("score_protocol") != SCORE_PROTOCOL or
            checkpoint.get("selection_metric") != "main_correlation"):
        raise ValueError("Legacy score state cannot resume; use --init-checkpoint in a new output directory")
    if checkpoint.get("checkpoint_role") != "last":
        raise ValueError("Exact resume requires last.pt")
    best = checkpoint.get("best_metrics")
    if not isinstance(best, dict) or set(best) != set(METRIC_FILES):
        raise ValueError("Missing best-r ledger")
    entry = best["main_correlation"]
    if entry is not None and (not isinstance(entry, dict) or
            not math.isfinite(float(entry.get("value", float("nan")))) or
            isinstance(entry.get("epoch"), bool) or not isinstance(entry.get("epoch"), int) or entry["epoch"] < 0):
        raise ValueError("Invalid best-r entry")
    if checkpoint.get("best_score") != (None if entry is None else entry["value"]):
        raise ValueError("best_score must equal best mean r")
    return copy.deepcopy(best)


def save_selected_checkpoints(output: Path, payload, improved):
    from .checkpoint import atomic_save
    if any(key not in METRIC_FILES for key in improved):
        raise ValueError("Unknown selected metric")
    for key in improved:
        atomic_save(output / METRIC_FILES[key], {**payload, "checkpoint_role": key})
    atomic_save(output / "last.pt", {**payload, "checkpoint_role": "last"})
