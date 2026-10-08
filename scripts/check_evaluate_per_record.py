"""Check that ECG correlations are averaged over records, not pooled points."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.evaluate import _pearson, evaluate_record_predictions, evaluate_joint_anchor_predictions, competition_score
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
    # Limb errors must not change the chest-only Task2 bonus RMSE.
    limb_error = prediction.copy()
    limb_error[:, 1:6] += 1000
    chest_score, _ = evaluate_record_predictions(limb_error, prediction, "task2", metadata)
    assert np.isclose(chest_score["task2_missing_lead_mean_rmse_uV"], 0)
    assert chest_score["task2_rmse_scored_leads"] == "V1,V2,V3,V4,V5,V6"
    assert chest_score["missing11_mean_rmse_uV"] > 0
    # Averaging six per-lead RMSEs, not averaging their squared errors.
    offsets = np.asarray([20, 40, 60, 80, 100, 120])
    limb_error[:, 6:12] += offsets[None, :, None]
    chest_score, _ = evaluate_record_predictions(limb_error, prediction, "task2", metadata)
    assert np.isclose(chest_score["task2_missing_lead_mean_rmse_uV"], 70)
    assert np.isclose(competition_score(.8, .6, chest_score["task2_missing_lead_mean_rmse_uV"])["task2_rmse_bonus_score"], 10)
    assert competition_score(.8, .6, 70)["task2_rmse_bonus_score"] == 10
    assert competition_score(.8, .6, 140)["task2_rmse_bonus_score"] == 5
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
