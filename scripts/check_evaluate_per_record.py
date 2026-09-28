"""Check that ECG correlations are averaged over records, not pooled points."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.evaluate import evaluate_centered_diagnostic, evaluate_predictions, evaluate_record_predictions
from ecg12gen.contracts import ContractError


def main() -> None:
    time = np.linspace(0, 4 * np.pi, 5000, dtype=np.float32)
    waveform = np.sin(time)
    prediction = np.broadcast_to(waveform, (2, 12, 5000)).copy()
    target = prediction.copy()
    target[1] += 100.0

    metadata = [
        {"pair_id": "record_a", "target_record_id": "target_a", "start_sample_500hz": "0"},
        {"pair_id": "record_a", "target_record_id": "target_a", "start_sample_500hz": "5000"},
    ]
    overall, details = evaluate_record_predictions(prediction, target, "task1", metadata)
    centered, _ = evaluate_centered_diagnostic(prediction, target, "task1", metadata_rows=metadata)
    window_level, _ = evaluate_predictions(prediction, target, "task1")
    assert overall["task1_r1"] < 1.0
    assert np.isclose(window_level["task1_r1"], 1.0, atol=1e-6)
    assert np.isclose(centered["task1_r1"], overall["task1_r1"], atol=1e-6)
    assert centered["evaluation_aggregation"] == "record_macro_after_chronological_window_stitch"
    assert overall["twelve_lead_mean_rmse_uV"] > 0.0
    broken = [metadata[0], {**metadata[1], "start_sample_500hz": "10000"}]
    try:
        evaluate_record_predictions(prediction, target, "task1", broken)
        raise AssertionError("gapped record cache was accepted as official evaluation")
    except ContractError:
        pass

    print("PASS: per-record Pearson r ignores per-record constant offsets; raw RMSE retains them")


if __name__ == "__main__":
    main()
