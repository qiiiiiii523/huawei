"""Check that ECG correlations are averaged over records, not pooled points."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.evaluate import _pearson, evaluate_record_predictions, evaluate_joint_anchor_predictions
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
    assert overall["r_missing11"] < 1.0
    assert np.isclose(_pearson(prediction[1, 1], target[1, 1]), 1.0)
    expected_r = _pearson(prediction[:, 1].reshape(-1), target[:, 1].reshape(-1))
    assert np.isclose(overall["r_missing11"], expected_r)
    assert overall["evaluation_aggregation"] == "record_macro_after_chronological_window_stitch"
    assert np.isclose(overall["missing11_mean_rmse_uV"], np.sqrt(5000.0))
    assert len(details) == 11 and all("pearson_r" not in row for row in details)
    summary, qc_details = evaluate_joint_anchor_predictions(
        prediction, target, target[:, :1], "task2", metadata, np.ones((2, 12), dtype=bool))
    assert np.isclose(summary["r_missing11"], overall["r_missing11"])
    assert all(key not in summary for key in ("r_raw_12", "r_submit_12", "task1_r1", "task2_r2"))
    assert np.array_equal(target[1], prediction[1] + 100)
    # Record offsets must not be removed during evaluation: RMSE remains raw.
    shifted_target = prediction + 100
    shifted, _ = evaluate_record_predictions(prediction, shifted_target, "task1", metadata)
    assert np.isclose(shifted["r_missing11"], 1)
    assert np.isclose(shifted["missing11_mean_rmse_uV"], 100)
    reordered, _ = evaluate_record_predictions(prediction[::-1], target[::-1], "task1", metadata[::-1])
    assert np.isclose(reordered["r_missing11"], overall["r_missing11"])
    # Identical targets paired with two contexts are separate scored samples.
    two_pairs = [metadata[0], {**metadata[1], "pair_id": "record_b", "start_sample_500hz": "0"}]
    separate, _ = evaluate_record_predictions(prediction, target, "task1", two_pairs)
    assert separate["n_records"] == 2 and np.isclose(separate["r_missing11"], 1)
    broken = [metadata[0], {**metadata[1], "start_sample_500hz": "10000"}]
    try:
        evaluate_record_predictions(prediction, target, "task1", broken)
        raise AssertionError("gapped record cache was accepted as official evaluation")
    except ContractError:
        pass
    missing_tail = [{**metadata[0], "expected_window_count": "2"}]
    try:
        evaluate_record_predictions(prediction[:1], target[:1], "task1", missing_tail)
        raise AssertionError("missing trailing window was accepted")
    except ContractError:
        pass

    print("PASS: only record r_missing11; raw RMSE unchanged; sorted sample grouping; missing windows rejected")


if __name__ == "__main__":
    main()
