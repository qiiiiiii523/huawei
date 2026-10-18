"""Meaningful CPU-only checks for fixed-alpha scoring and target-level oracle."""
from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.B5_best.common import read, sha
from baselines.B5_best_context.body_alpha_core import (ALPHAS, group_name, mix_ii,
    SharedCorrection, FixedAlphaAdapter, cases, oracle_diagnostic, target_mean)
from baselines.B5_best_context.body_alpha_study import reference_check, summarize, table
from baselines.B5_best_context.body_study import run_study
from . import test_context


class AlphaTests(unittest.TestCase):
    def test_physical_voltage_mixing_and_exact_endpoints(self):
        base = np.array([1000., 1200., -900.], np.float32)
        corrected = np.array([900., 1000., -600.], np.float32)
        np.testing.assert_array_equal(mix_ii(base, corrected, .5), [950., 1100., -750.])
        np.testing.assert_array_equal(mix_ii(base, corrected, 0), base)
        np.testing.assert_array_equal(mix_ii(base, corrected, 1), corrected)
        self.assertEqual(mix_ii(base, corrected, .75).dtype, np.float32)
        with self.assertRaises(ValueError):
            mix_ii(base, corrected, 1.5)

    def test_oracle_averages_pairs_before_equal_target_weights(self):
        records = []
        for target, pairs in (("A", 3), ("B", 1)):
            for pair in range(pairs):
                for a in ALPHAS:
                    records.append({"target_record_id": target, "alpha": a,
                        "r_II": .9 - .8 * a if target == "A" else .1 + .8 * a})
        result, choices = oracle_diagnostic(records)
        self.assertAlmostEqual(result["best_fixed_II_r"], .5)
        self.assertAlmostEqual(result["oracle_II_r"], .9)
        self.assertEqual(choices, {"A": 0., "B": 1.})
        self.assertFalse(result["deployable"])
        # Undefined targets must not improve the oracle by changing its cohort.
        records += [{"target_record_id": "undefined", "alpha": a,
                     "r_II": None if a == .25 else 1.} for a in ALPHAS]
        result, _ = oracle_diagnostic(records)
        self.assertEqual(result["excluded_targets"], ["undefined"])
        self.assertAlmostEqual(result["best_fixed_II_r"], .5)

    def test_inference_signature_has_no_target_and_shares_forward(self):
        self.assertEqual(list(inspect.signature(SharedCorrection.get).parameters),
                         ["self", "prediction", "observed_i", "history"])
        class Engine:
            calls = 0
            def __call__(self, p, i, history):
                self.calls += 1
                p[1] += .1 * i[0]
                return p, {}
        engine, base = Engine(), np.ones((12, 60000), np.float32)
        i = np.full((1, 60000), 20., np.float32)
        shared = SharedCorrection(engine)
        for a in ALPHAS:
            value, status = FixedAlphaAdapter(shared, a)(base, i, None)
            np.testing.assert_array_equal(value[1], 1 + 2 * a)
            np.testing.assert_array_equal(value[[0, *range(2, 12)]], base[[0, *range(2, 12)]])
            self.assertFalse(status["target_used_for_alpha"])
        self.assertEqual(engine.calls, 1)
        np.testing.assert_array_equal(base, 1)
        FixedAlphaAdapter(shared, 1)(base, i * 2, None)
        self.assertEqual(engine.calls, 2)

    def test_reference_requires_same_epoch_scores_and_population(self):
        rows = [{"pair_id": "a", "target_record_id": "A", "alpha": 1., "r_II": .7}]
        reference = {"pairs": [{"pair_id": "a", "target_record_id": "A", "r_II": .7}],
                     "validation_full120s_II_r": .7}
        self.assertTrue(reference_check(rows, reference)["reproduced"])
        with self.assertRaises(ValueError):
            reference_check(rows, {**reference, "validation_full120s_II_r": .8})
        with self.assertRaises(ValueError):
            reference_check(rows, {**reference, "pairs": []})

    def test_full_scoring_preserves_QC_routes_and_base_arrays(self):
        class Engine:
            def __call__(self, p, i, history):
                p[1] += .04 * i[0]
                return p, {}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            test_context.StudyTests().baseline(root / "base")
            before = sha(root / "base/task2/prediction_uV.npy")
            shared = SharedCorrection(Engine())
            summary = run_study(root / "base", root / "study", lambda row: None, None, None,
                case_specs=cases(), adapter_map={group_name(a): FixedAlphaAdapter(shared, a) for a in ALPHAS},
                additive_routes=[], notes=["Synthetic fixed-alpha experiment"])
            self.assertEqual(shared.forward_calls, 1)  # Synthetic pairs share exact visible inputs.
            groups = {r["group"]: r for r in summary}
            for a in ALPHAS:
                self.assertEqual(groups[group_name(a)]["machine_r"], groups["d6_only"]["machine_r"])
                self.assertEqual(groups[group_name(a)]["task1_r"], groups["d6_only"]["task1_r"])
            self.assertEqual(groups[group_name(0)]["mean_r"], groups["d6_only"]["mean_r"])
            metrics = [r for r in table(root / "study/alpha_100/pair_metrics.csv")
                       if r["input_type"] == "body_scale_d6" and r["input_I_reliable"] == "True"]
            pairs = [{"pair_id": r["pair_id"], "target_record_id": r["target_record_id"], "r_II": float(r["r_II"])} for r in metrics]
            reference = {"pairs": pairs, "validation_full120s_II_r": target_mean(pairs)}
            cache = {"records": [{"split": "validation", "I_II_eligible": True,
                "subject_id": r["subject_id"], "pair_id": r["pair_id"], "target_record_id": r["target_record_id"]} for r in metrics]}
            rows, check = summarize(root / "study", cache, reference, summary)
            self.assertTrue(check["reproduced"])
            self.assertEqual(len(rows), 5)
            self.assertEqual(len(table(root / "study/record_alpha.csv")), 10)  # Bad current I excluded.
            self.assertEqual(read(root / "study/oracle_summary.json")["comparable_targets"], 2)
            self.assertEqual(sha(root / "base/task2/prediction_uV.npy"), before)


if __name__ == "__main__":
    unittest.main()
