"""CPU synthetic regression/integration checks: actual-input QC + best_r mean."""
from pathlib import Path
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.contracts import ContractError
from ecg12gen.evaluate import (evaluate_record_predictions, evaluate_joint_anchor_predictions,
    correlation_score, correlation_score_from_summaries, RELIABLE_INPUT_MEAN_R_SCORE_PROTOCOL)
from check_target_record_scoring import dataset, watch


class ReliableInputTests(unittest.TestCase):
    def build(self):
        p, t, rows = dataset([watch("good", "A"), watch("bad1", "B", -1), watch("bad2", "B", -1)])
        q = np.ones((len(p), 12), bool)
        q[12:, 0] = False
        return p, t, rows, q

    def evaluate(self, p, t, rows, q, task="task1", observed=None, source="same_record_target_I"):
        return evaluate_record_predictions(p, t, task, rows, target_quality_mask=q,
                    input_quality_mask=q[:, 0] if observed is None else observed,
                    require_reliable_input=True, input_quality_source=source)

    def test_whole_record_exclusion_and_source_unchanged(self):
        p, t, rows, q = self.build()
        original = p.copy()
        legacy, _ = evaluate_record_predictions(p, t, "task1", rows, target_quality_mask=q)
        result, detail = self.evaluate(p, t, rows, q)
        self.assertAlmostEqual(legacy["r_missing11"], 0)
        self.assertAlmostEqual(result["r_missing11"], 1)
        self.assertEqual(result["n_windows"], 12)
        self.assertEqual(result["input_qc_excluded_windows"], 24)
        self.assertEqual(result["input_qc_excluded_pairs"], 2)
        self.assertEqual(result["input_qc_fully_excluded_physical_targets"], 1)
        self.assertEqual({r["target_record_id"] for r in result["input_qc_exclusions"]}, {"B"})
        self.assertEqual(detail[0]["n_validation_points"], 60000)
        self.assertFalse(any("bonus" in k for k in result))
        np.testing.assert_array_equal(p, original)

    def test_all_good_preserves_metrics_legacy_default_unchanged(self):
        p, t, rows, q = self.build()
        q[:] = True
        old, old_detail = evaluate_record_predictions(p, t, "task1", rows, target_quality_mask=q)
        new, detail = self.evaluate(p, t, rows, q)
        for key in ("r_missing11", "missing11_mean_rmse_uV", "n_windows", "n_records", "n_unique_target_records"):
            self.assertEqual(old[key], new[key])
        self.assertEqual(old_detail, detail)
        self.assertNotIn("input_quality_protocol", old)

    def test_target_bad_lead_excludes_only_that_lead(self):
        p, t, rows = dataset([watch("a", "A"), watch("b", "B")])
        q = np.ones((len(p), 12), bool)
        q[:12, 9] = False
        p[:12, 9] = -t[:12, 9]
        result, detail = self.evaluate(p, t, rows, q)
        self.assertEqual(result["input_qc_excluded_pairs"], 0)
        self.assertAlmostEqual(result["r_missing11"], 1)
        by_lead = {r["lead"]: r for r in detail}
        self.assertEqual(by_lead["V4"]["n_unique_target_records"], 1)
        self.assertEqual(by_lead["V3"]["n_unique_target_records"], 2)

    def test_requires_explicit_masks_source_and_opt_in(self):
        p, t, rows, q = self.build()
        with self.assertRaisesRegex(ContractError, "explicit require"):
            evaluate_record_predictions(p, t, "task1", rows, input_quality_mask=q[:, 0])
        with self.assertRaisesRegex(ContractError, "requires both"):
            evaluate_record_predictions(p, t, "task1", rows, require_reliable_input=True)
        with self.assertRaisesRegex(ContractError, "source"):
            self.evaluate(p, t, rows, q, source="")
        with self.assertRaisesRegex(ContractError, "disagrees"):
            self.evaluate(p, t, rows, q, observed=np.ones(len(p), bool))

    def test_input_cannot_change_mid_record_or_have_invalid_values(self):
        p, t, rows, q = self.build()
        observed = q[:, 0].copy()
        observed[1] = False
        with self.assertRaisesRegex(ContractError, "constant across"):
            self.evaluate(p, t, rows, q, observed=observed, source="external_anchor_qc")
        for invalid in (np.ones((len(p), 2)), np.full(len(p), 2), np.full(len(p), np.nan)):
            with self.assertRaisesRegex(ContractError, "input_quality_mask"):
                self.evaluate(p, t, rows, q, observed=invalid, source="external_anchor_qc")

    def test_empty_population_and_empty_required_lead_fail(self):
        p, t, rows, q = self.build()
        q[:, 0] = False
        with self.assertRaisesRegex(ContractError, "No records remain"):
            self.evaluate(p, t, rows, q)
        q[:, 0] = True
        q[:, 6] = False
        with self.assertRaisesRegex(ContractError, "No QC-eligible.*V1"):
            self.evaluate(p, t, rows, q)

    def test_shuffle_preserves_complete_record_scores(self):
        p, t, rows, q = self.build()
        expected, _ = self.evaluate(p, t, rows, q)
        order = np.random.default_rng(5).permutation(len(p))
        actual, _ = self.evaluate(p[order], t[order], [rows[i] for i in order], q[order])
        self.assertEqual(expected["r_missing11"], actual["r_missing11"])
        self.assertEqual(expected["input_qc_excluded_windows"], actual["input_qc_excluded_windows"])

    def test_different_source_quality_is_not_inferred_from_target_I(self):
        p, t, rows, q = self.build()
        q[:, 0] = True
        observed = np.zeros((len(p), 1), bool)
        observed[:12] = True
        result, _ = self.evaluate(p, t, rows, q, observed=observed, source="separate_observed_I_qc")
        self.assertAlmostEqual(result["r_missing11"], 1)
        self.assertEqual(result["input_qc_excluded_pairs"], 2)

    def test_pair_specific_source_can_retain_other_pair_of_same_target(self):
        p, t, rows = dataset([watch("a1", "A"), watch("a2", "A"), watch("b", "B")])
        q = np.ones((len(p), 12), bool)
        observed = np.ones(len(p), bool)
        observed[12:24] = False
        result, _ = self.evaluate(p, t, rows, q, observed=observed, source="separate_pair_inputs")
        self.assertEqual(result["n_unique_target_records"], 2)
        self.assertEqual(result["input_qc_fully_excluded_physical_targets"], 0)
        self.assertEqual(result["input_qc_excluded_pairs"], 1)

    def test_task2_device_equal_after_filter(self):
        specs = [("m1", "A", 1, "ecg_machine_d6", "a"),
                 ("m2", "A", 1, "ecg_machine_d6", "a"),
                 ("m_bad", "B", -1, "ecg_machine_d6", "b"),
                 ("body", "C", -1, "body_scale_d6", "c")]
        p, t, rows = dataset(specs)
        q = np.ones((len(p), 12), bool)
        q[24:36, 0] = False
        result, _ = self.evaluate(p, t, rows, q, "task2")
        self.assertAlmostEqual(result["ecg_machine_d6_r_missing11"], 1)
        self.assertAlmostEqual(result["body_scale_d6_r_missing11"], -1)
        self.assertAlmostEqual(result["r_missing11"], 0)
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["n_unique_physical_target_records"], 2)
        q[36:, 0] = False
        with self.assertRaisesRegex(ContractError, "missing body_scale_d6"):
            self.evaluate(p, t, rows, q, "task2")

    def test_joint_adapter_is_also_explicit(self):
        p, t, rows, q = self.build()
        result, _ = evaluate_joint_anchor_predictions(p, t, t[:, :1], "task1", rows, q,
                        apply_target_qc=True, require_reliable_input=True,
                        input_quality_mask=q[:, 0], input_quality_source="same_record_target_I")
        self.assertAlmostEqual(result["r_missing11"], 1)
        with self.assertRaisesRegex(ContractError, "apply_target_qc"):
            evaluate_joint_anchor_predictions(p, t, t[:, :1], "task1", rows, q, require_reliable_input=True)

    def test_mean_r_is_independent_of_RMSE(self):
        score = correlation_score(.8, .6)
        self.assertAlmostEqual(score["main_correlation"], .7)
        self.assertEqual(set(score), {"task1_r_missing11", "task2_r_missing11", "main_correlation",
                                      "selection_metric", "score_protocol"})
        for value in (np.nan, np.inf, 1.1):
            with self.assertRaises(ContractError):
                correlation_score(value, .6)

    def test_summary_protocol_mismatch_is_rejected(self):
        p, t, rows, q = self.build()
        a, _ = self.evaluate(p, t, rows, q)
        p2, t2, rows2 = dataset([("m", "A", 1, "ecg_machine_d6", "a"),
                                  ("b", "A", 1, "body_scale_d6", "a")])
        q2 = np.ones((len(p2), 12), bool)
        b, _ = self.evaluate(p2, t2, rows2, q2, "task2")
        score = correlation_score_from_summaries({"task1": a, "task2": b})
        self.assertEqual(score["score_protocol"], RELIABLE_INPUT_MEAN_R_SCORE_PROTOCOL)
        b["missing11_mean_rmse_uV"] = 1e12
        self.assertEqual(score, correlation_score_from_summaries({"task1": a, "task2": b}))
        legacy, _ = evaluate_record_predictions(p2, t2, "task2", rows2, target_quality_mask=q2)
        with self.assertRaisesRegex(ContractError, "mix reliable"):
            correlation_score_from_summaries({"task1": a, "task2": legacy})

    def test_saved_array_cli_and_mean_summary_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation = root / "old_evaluation"
            for task in ("task1", "task2"):
                if task == "task1":
                    p, t, rows, q = self.build()
                else:
                    p, t, rows = dataset([("m", "A", 1, "ecg_machine_d6", "a"),
                                          ("b", "A", -1, "body_scale_d6", "a")])
                    q = np.ones((len(p), 12), bool)
                folder = evaluation / task
                folder.mkdir(parents=True)
                for name, value in (("prediction_uV", p), ("target_uV", t), ("target_quality_mask", q)):
                    np.save(folder / (name + ".npy"), value)
                with (folder / "window_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            output = root / "new_report"
            subprocess.run([sys.executable, str(ROOT / "scripts/rescore_reliable_i.py"),
                "--evaluation-dir", str(evaluation), "--output-dir", str(output),
                "--input-quality-source", "same_record_target_I"], check=True, capture_output=True)
            audit = json.loads((output / "rescore.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["completed"])
            self.assertFalse(audit["new_inference"])
            self.assertAlmostEqual(audit["score"]["main_correlation"], .5)
            with (output / "task1/input_qc_exclusions.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            subprocess.run([sys.executable, str(ROOT / "scripts/summarize_mean_r.py"),
                "--task1-overall", str(output / "task1/overall_metrics.csv"),
                "--task2-overall", str(output / "task2/overall_metrics.csv"),
                "--output-dir", str(root / "summary")], check=True, capture_output=True)
            self.assertEqual(json.loads((root / "summary/correlation_score.json").read_text()), audit["score"])
            # Never overwrite historical arrays/reports.
            spec = importlib.util.spec_from_file_location("cpu_rescore", ROOT / "scripts/rescore_reliable_i.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.assertRaises(ValueError):
                module.rescore(evaluation, evaluation / "new", "same_record_target_I")
            with self.assertRaises(FileExistsError):
                module.rescore(evaluation, output, "same_record_target_I")


if __name__ == "__main__":
    unittest.main()
