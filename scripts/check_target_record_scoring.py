"""CPU-only scoring regression tests; no training or production data access."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.contracts import ContractError
from ecg12gen.evaluate import (evaluate_record_predictions, competition_score,
                              TARGET_RECORD_AGGREGATION, TASK2_DEVICE_AGGREGATION)


def dataset(specs):
    """Specs: (pair, physical target, multiplier or 'constant', device, patient)."""
    signal = (100 * np.sin(np.linspace(0, 200, 60000))).astype(np.float32)
    truth = np.broadcast_to(signal, (12, 60000)).copy()
    predictions, targets, rows = [], [], []
    for pair, target_id, multiplier, device, patient in specs:
        prediction = np.zeros_like(truth) if multiplier == "constant" else truth * multiplier
        for start in range(0, 60000, 5000):
            predictions.append(prediction[:, start:start+5000])
            targets.append(truth[:, start:start+5000])
            rows.append({"pair_id": pair, "target_record_id": target_id, "subject_id": patient,
                         "input_type": device, "start_sample_500hz": str(start), "expected_window_count": "12"})
    return np.stack(predictions), np.stack(targets), rows


def watch(pair, target, multiplier=1, subject="same_patient"):
    return pair, target, multiplier, "watch_ecg", subject


class TargetScoringTests(unittest.TestCase):
    def test_identical_duplicate_pairs_do_not_reweight_targets(self):
        a = dataset([watch("a", "A"), watch("b", "B", -1)])
        repeated = dataset([watch("a1", "A"), watch("a2", "A"), watch("a3", "A"), watch("b", "B", -1)])
        first, _ = evaluate_record_predictions(*a[:2], "task1", a[2])
        second, details = evaluate_record_predictions(*repeated[:2], "task1", repeated[2])
        self.assertAlmostEqual(first["r_missing11"], 0, places=10)
        self.assertAlmostEqual(first["r_missing11"], second["r_missing11"], places=10)
        self.assertAlmostEqual(first["missing11_mean_rmse_uV"], second["missing11_mean_rmse_uV"], places=9)
        self.assertEqual(second["n_pairs"], 4)
        self.assertEqual(second["n_unique_target_records"], 2)
        self.assertEqual(details[0]["n_validation_points"], 120000)
        self.assertEqual(second["target_record_aggregation"], TARGET_RECORD_AGGREGATION)

    def test_different_context_predictions_are_scored_before_target_average(self):
        p, t, rows = dataset([watch("a_good", "A"), watch("a_bad", "A", -1), watch("b", "B")])
        original = p.copy()
        summary, _ = evaluate_record_predictions(p, t, "task1", rows)
        # A: average(1,-1)=0; B:1. Target-equal r=.5, not pooled pair r=1/3.
        self.assertAlmostEqual(summary["r_missing11"], .5, places=10)
        # A pair MSE=(0+4*E[t^2])/2; B MSE=0. Overall RMSE=sqrt(E[t^2]).
        expected = float(np.sqrt(np.mean(t[0, 1].astype(np.float64) ** 2)))
        # The waveform is stitched, so derive the independent full-record mean.
        full_truth = np.concatenate([t[i, 1] for i in range(12)]).astype(np.float64)
        expected = float(np.sqrt(np.mean(full_truth ** 2)))
        self.assertAlmostEqual(summary["missing11_mean_rmse_uV"], expected, places=8)
        np.testing.assert_array_equal(p, original)

    def test_different_targets_of_same_patient_remain_two_targets(self):
        p, t, rows = dataset([watch("a", "A"), watch("b", "B", -1)])
        summary, _ = evaluate_record_predictions(p, t, "task1", rows)
        self.assertEqual(summary["n_unique_target_records"], 2)
        self.assertAlmostEqual(summary["r_missing11"], 0, places=10)

    def test_task2_deduplicates_within_device_not_across_devices(self):
        specs = [("m1", "A", 1, "ecg_machine_d6", "patient"),
                 ("m2", "A", 1, "ecg_machine_d6", "patient"),
                 ("m3", "A", 1, "ecg_machine_d6", "patient"),
                 ("mb", "B", -1, "ecg_machine_d6", "patient"),
                 ("body", "A", 1, "body_scale_d6", "patient")]
        p, t, rows = dataset(specs)
        summary, details = evaluate_record_predictions(p, t, "task2", rows)
        self.assertAlmostEqual(summary["ecg_machine_d6_r_missing11"], 0, places=10)
        self.assertAlmostEqual(summary["body_scale_d6_r_missing11"], 1, places=10)
        self.assertAlmostEqual(summary["r_missing11"], .5, places=10)
        self.assertEqual(summary["n_unique_physical_target_records"], 2)
        self.assertEqual(summary["n_unique_target_records"], 3)  # two machine + one body target
        self.assertEqual(summary["task2_device_aggregation"], TASK2_DEVICE_AGGREGATION)
        self.assertEqual(len(details), 22)
        duplicate_free = [specs[0], specs[3], specs[4]]
        qp, qt, qrows = dataset(duplicate_free)
        other, _ = evaluate_record_predictions(qp, qt, "task2", qrows)
        self.assertAlmostEqual(summary["task2_rmse_bonus_score"], other["task2_rmse_bonus_score"], places=10)
        self.assertEqual(competition_score(1, 1, 70, 0)["competition_total_score"], 60)

    def test_same_target_different_true_waveforms_or_subjects_are_rejected(self):
        p, t, rows = dataset([watch("a1", "A"), watch("a2", "A")])
        t[12:, 3] += 1
        with self.assertRaisesRegex(ContractError, "different raw target"):
            evaluate_record_predictions(p, t, "task1", rows)
        p, t, rows = dataset([watch("a1", "A", subject="one"), watch("a2", "A", subject="two")])
        with self.assertRaisesRegex(ContractError, "inconsistent subject_id"):
            evaluate_record_predictions(p, t, "task1", rows)

    def test_target_id_and_full_120_seconds_are_required(self):
        p, t, rows = dataset([watch("a", "A")])
        rows[0]["start_sample_500hz"] = "5000"
        with self.assertRaises(ContractError):
            evaluate_record_predictions(p, t, "task1", rows)
        p, t, rows = dataset([watch("a", "A")])
        for row in rows:
            del row["target_record_id"]
        with self.assertRaisesRegex(ContractError, "requires target_record_id"):
            evaluate_record_predictions(p, t, "task1", rows)

    def test_undefined_pair_r_does_not_discard_defined_target_r(self):
        p, t, rows = dataset([watch("a_constant", "A", "constant"), watch("a_good", "A"), watch("b", "B", -1)])
        summary, _ = evaluate_record_predictions(p, t, "task1", rows)
        self.assertAlmostEqual(summary["r_missing11"], 0, places=10)
        self.assertEqual(summary["n_undefined_pair_lead_correlations"], 11)
        self.assertEqual(summary["n_undefined_record_lead_correlations"], 0)


if __name__ == "__main__":
    unittest.main()
