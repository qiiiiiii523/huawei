"""Synthetic checks for full-record, equal-device correlation and nonlinear bonus weighting."""
from __future__ import annotations
import sys
import csv
import subprocess
import tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.evaluate import evaluate_record_predictions, competition_score, rmse_bonus
from ecg12gen.contracts import ContractError


def metadata(pair, source="watch_ecg"):
    return [{"pair_id": pair, "target_record_id": pair, "start_sample_500hz": str(i * 5000),
             "expected_window_count": "12", "input_type": source} for i in range(12)]


def main():
    wave = np.sin(np.linspace(0, 8 * np.pi, 60000)).reshape(12, 5000)
    target = np.broadcast_to(wave[:, None], (12, 12, 5000)).copy()
    prediction = target.copy()
    prediction[6:] += 100
    rows = metadata("task1")
    summary, _ = evaluate_record_predictions(prediction, target, "task1", rows)
    assert summary["r_missing11"] < .1
    assert np.isclose(summary["missing11_mean_rmse_uV"], np.sqrt(5000))
    copied = prediction.copy()
    evaluate_record_predictions(prediction, target, "task1", rows)
    assert np.array_equal(prediction, copied)
    reordered, _ = evaluate_record_predictions(prediction[::-1], target[::-1], "task1", rows[::-1])
    assert np.isclose(reordered["r_missing11"], summary["r_missing11"])
    # 4 machine records vs 1 body record: equal-device r, not 4:1 averaging.
    y = np.concatenate([target] * 5)
    p = y.copy()
    all_rows = sum([metadata(f"machine_{i}", "ecg_machine_d6") for i in range(4)], []) + metadata("body", "body_scale_d6")
    p[-12:, 1:] *= -1
    result, details = evaluate_record_predictions(p, y, "task2", all_rows)
    assert np.isclose(result["r_missing11"], 0, atol=1e-6)
    assert result["ecg_machine_d6_n_records"] == 4 and result["body_scale_d6_n_records"] == 1
    assert len(details) == 22
    p = y.copy()
    p[:, 1:6] += 1000  # limb errors never contribute to chest RMSE bonus
    p[-12:, 6:] += 140
    result, details = evaluate_record_predictions(p, y, "task2", all_rows)
    assert np.isclose(result["r_missing11"], 1)
    assert np.isclose(result["ecg_machine_d6_task2_missing_lead_mean_rmse_uV"], 0)
    assert np.isclose(result["body_scale_d6_task2_missing_lead_mean_rmse_uV"], 140)
    assert np.isclose(result["task2_missing_lead_mean_rmse_uV"], 70)  # diagnostic only
    assert np.isclose(result["task2_rmse_bonus_score"], 7.5)  # NOT b(70)=10
    score = competition_score(.8, result["r_missing11"], 0, 140)
    assert np.isclose(score["task2_rmse_bonus_score"], 7.5)
    assert rmse_bonus(70) == 10 and rmse_bonus(140) == 5
    for bad_prediction, bad_target, bad_rows in (
        (target[:11], target[:11], rows[:11]),
        (target, target, [{**r, "start_sample_500hz": "10000"} if i == 1 else r for i, r in enumerate(rows)]),
        (target, target, [{**r, "input_type": "ecg_machine_d6"} for r in rows]),
    ):
        task = "task2" if bad_rows[0]["input_type"] == "ecg_machine_d6" else "task1"
        try:
            evaluate_record_predictions(bad_prediction, bad_target, task, bad_rows)
            raise AssertionError("Incomplete record/device group was accepted")
        except ContractError:
            pass
    # Exercise CSV serialization and the actual cross-task summary consumer.
    from ecg12gen.evaluate import write_report
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        task1, detail1 = evaluate_record_predictions(target, target, "task1", rows)
        write_report(root / "task1", task1, detail1)
        write_report(root / "task2", result, details)
        run = subprocess.run([sys.executable, str(ROOT / "scripts/summarize_competition_score.py"),
            "--task1-overall", str(root / "task1/overall_metrics.csv"), "--task2-overall", str(root / "task2/overall_metrics.csv"),
            "--output-dir", str(root / "total")], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        with (root / "total/competition_score.csv").open() as handle:
            total = next(csv.DictReader(handle))
        assert np.isclose(float(total["task2_rmse_bonus_score"]), 7.5)
        assert np.isclose(float(total["competition_total_score"]), 8.5)
    print("PASS: 120-second record r; 4:1 data still scored 1:1; per-device RMSE bonus=7.5, not 10; immutable raw inputs")


if __name__ == "__main__":
    main()
