"""Check that ECG correlations are averaged over records, not pooled points."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.evaluate import evaluate_centered_diagnostic, evaluate_predictions


def main() -> None:
    time = np.linspace(0, 4 * np.pi, 5000, dtype=np.float32)
    waveform = np.sin(time)
    prediction = np.broadcast_to(waveform, (2, 12, 5000)).copy()
    target = prediction.copy()
    target[1] += 100.0

    overall, details = evaluate_predictions(prediction, target, "task1")
    centered, _ = evaluate_centered_diagnostic(prediction, target, "task1")
    assert np.isclose(overall["task1_r1"], 1.0, atol=1e-6)
    assert all(np.isclose(row["pearson_r"], 1.0, atol=1e-6) for row in details)
    assert np.isclose(overall["task1_r1"], centered["task1_r1"], atol=1e-6)
    assert overall["twelve_lead_mean_rmse_uV"] > 0.0

    print("PASS: per-record Pearson r ignores per-record constant offsets; raw RMSE retains them")


if __name__ == "__main__":
    main()
