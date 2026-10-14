"""Synthetic 120-second checks for explicit lead-QC scoring; no training."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg12gen.contracts import ContractError
from ecg12gen.evaluate import (evaluate_record_predictions, evaluate_joint_anchor_predictions,
    competition_score, QC_COMPETITION_SCORE_PROTOCOL, QC_TASK2_DEVICE_AGGREGATION)
from check_target_record_scoring import dataset, watch


class LeadQCTests(unittest.TestCase):
    def build(self):
        p, t, rows = dataset([watch('a1', 'A'), watch('a2', 'A'), watch('b', 'B')])
        q = np.ones((len(p), 12), dtype=bool)
        q[:24, 9] = False  # Only target A's V4, repeated in two pairs.
        p[:24, 9] = -t[:24, 9] + 10000
        return p, t, rows, q

    def test_excludes_bad_lead_not_record_or_denominator(self):
        p, t, rows, q = self.build()
        original = p.copy()
        result, detail = evaluate_record_predictions(p, t, 'task1', rows, target_quality_mask=q)
        by_lead = {row['lead']: row for row in detail}
        self.assertAlmostEqual(result['r_missing11'], 1)
        self.assertEqual(result['missing11_mean_rmse_uV'], 0)
        self.assertEqual(result['n_unique_target_records'], 2)
        self.assertEqual(by_lead['V4']['n_records'], 1)
        self.assertEqual(by_lead['V4']['n_unique_target_records'], 1)
        self.assertEqual(by_lead['V4']['n_validation_points'], 60000)
        self.assertEqual(by_lead['V3']['n_validation_points'], 120000)
        self.assertEqual(by_lead['V4']['n_qc_excluded_pairs'], 2)
        self.assertEqual(result['n_qc_excluded_target_leads'], 1)
        self.assertEqual(result['competition_score_protocol'], QC_COMPETITION_SCORE_PROTOCOL)
        np.testing.assert_array_equal(p, original)
        legacy, _ = evaluate_record_predictions(p, t, 'task1', rows)
        self.assertLess(legacy['r_missing11'], 1)
        self.assertGreater(legacy['missing11_mean_rmse_uV'], 100)

    def test_all_good_mask_preserves_scores(self):
        p, t, rows, q = self.build()
        q[:] = True
        legacy, old_detail = evaluate_record_predictions(p, t, 'task1', rows)
        qc, detail = evaluate_record_predictions(p, t, 'task1', rows, target_quality_mask=q)
        for key in ('r_missing11', 'missing11_mean_rmse_uV'):
            self.assertEqual(legacy[key], qc[key])
        for old, new in zip(old_detail, detail):
            for key in old:
                self.assertEqual(old[key], new[key])

    def test_mask_cannot_vary_by_window_pair_or_device(self):
        p, t, rows, q = self.build()
        for index in (1, 12):
            bad = q.copy(); bad[index, 9] = True
            with self.assertRaisesRegex(ContractError, 'Inconsistent target quality'):
                evaluate_record_predictions(p, t, 'task1', rows, target_quality_mask=bad)
        p, t, rows = dataset([('m', 'A', 1, 'ecg_machine_d6', 'p'),
                              ('b', 'A', 1, 'body_scale_d6', 'p')])
        q = np.ones((len(p), 12), dtype=bool); q[:12, 9] = False
        with self.assertRaisesRegex(ContractError, 'Inconsistent target quality'):
            evaluate_record_predictions(p, t, 'task2', rows, target_quality_mask=q)

    def test_shape_values_and_empty_lead_rejected(self):
        p, t, rows, q = self.build()
        for invalid in (q[:, :11], np.full(q.shape, 2), np.full(q.shape, np.nan)):
            with self.assertRaises(ContractError):
                evaluate_record_predictions(p, t, 'task1', rows, target_quality_mask=invalid)
        q[:, 9] = False
        with self.assertRaisesRegex(ContractError, 'No QC-eligible.*V4'):
            evaluate_record_predictions(p, t, 'task1', rows, target_quality_mask=q)

    def test_excluded_undefined_r_is_not_a_scored_failure(self):
        p, t, rows, q = self.build(); p[:24, 9] = 0
        result, _ = evaluate_record_predictions(p, t, 'task1', rows, target_quality_mask=q)
        self.assertEqual(result['n_undefined_pair_lead_correlations'], 0)
        self.assertEqual(result['n_undefined_record_lead_correlations'], 0)

    def test_device_equal_bonus_uses_filtered_rmse(self):
        specs = [('m1', 'A', 1, 'ecg_machine_d6', 'p'),
                 ('m2', 'A', 1, 'ecg_machine_d6', 'p'),
                 ('mb', 'B', 1, 'ecg_machine_d6', 'p'),
                 ('b', 'B', 1, 'body_scale_d6', 'p')]
        p, t, rows = dataset(specs); q = np.ones((len(p), 12), dtype=bool)
        p[:36, 6:] += 70; p[36:, 6:] += 140
        q[:24, 9] = False; p[:24, 9] += 10000
        result, detail = evaluate_record_predictions(p, t, 'task2', rows, target_quality_mask=q)
        self.assertAlmostEqual(result['ecg_machine_d6_task2_missing_lead_mean_rmse_uV'], 70, places=4)
        self.assertAlmostEqual(result['body_scale_d6_task2_missing_lead_mean_rmse_uV'], 140, places=4)
        self.assertAlmostEqual(result['task2_rmse_bonus_score'], 7.5, places=4)
        self.assertEqual(result['task2_device_aggregation'], QC_TASK2_DEVICE_AGGREGATION)
        self.assertEqual(result['n_qc_excluded_target_leads'], 1)
        self.assertEqual(len(detail), 22)

    def test_joint_adapter_requires_explicit_opt_in(self):
        p, t, rows, q = self.build()
        legacy, _ = evaluate_joint_anchor_predictions(p, t, t[:, :1], 'task1', rows, q)
        qc, _ = evaluate_joint_anchor_predictions(p, t, t[:, :1], 'task1', rows, q, apply_target_qc=True)
        self.assertLess(legacy['r_missing11'], qc['r_missing11'])
        with self.assertRaises(ContractError):
            evaluate_joint_anchor_predictions(p, t, t[:, :1], 'task1', rows, apply_target_qc=True)

    def test_formula_is_unchanged_and_protocol_is_explicit(self):
        old = competition_score(.8, .6, 70, 140)
        qc = competition_score(.8, .6, 70, 140, score_protocol=QC_COMPETITION_SCORE_PROTOCOL)
        self.assertAlmostEqual(qc['competition_total_score'], 42.5)
        self.assertEqual(old['competition_total_score'], qc['competition_total_score'])
        self.assertNotEqual(old['competition_score_protocol'], qc['competition_score_protocol'])


if __name__ == '__main__':
    unittest.main()
