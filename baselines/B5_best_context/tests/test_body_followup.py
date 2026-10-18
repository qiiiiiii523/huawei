"""Pure-CPU checks for fixed-case diagnosis and fair across-seed comparisons."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.B5_best.common import write, read, csv_write
from baselines.B5_best_context.body_diagnose import waveform_metrics, select_examples, DEFAULT_SUBJECTS
from baselines.B5_best_context.body_compare import compare
from baselines.B5_best_context.body_study import groups


class DiagnosisTests(unittest.TestCase):
    def test_metrics_keep_voltage_offset_and_whole_record(self):
        t = np.arange(60000)
        target = np.sin(t / 55) * 200 + np.cos(t / 5000) * 100
        base = target + 300
        result = waveform_metrics(target, target, {"base": base, "no_context": base, "context": target.copy()})
        self.assertAlmostEqual(result["base"]["r_II"], 1.)
        self.assertAlmostEqual(result["base"]["RMSE_uV"], 300.)
        self.assertAlmostEqual(result["base"]["mean_error_uV"], 300.)
        self.assertAlmostEqual(result["context"]["RMSE_uV"], 0.)
        self.assertAlmostEqual(result["context"]["correction_RMS_uV"], 300.)
        with self.assertRaises(ValueError):
            waveform_metrics(target[:5000], target[:5000], {"base": base[:5000]})

    def test_fixed_cases_stay_selected_regardless_of_new_seed_ranking(self):
        rows = [{"subject_id": s, "pair_id": str(i), "delta_II_r": (-1)**i * .1}
                for i, s in enumerate(DEFAULT_SUBJECTS)]
        rows += [{"subject_id": "near_zero", "pair_id": "other", "delta_II_r": .00001},
                 {"subject_id": "new_best", "pair_id": "best", "delta_II_r": .5}]
        selected = select_examples(rows)
        self.assertEqual([r["subject_id"] for r in selected], [*DEFAULT_SUBJECTS, "near_zero"])
        with self.assertRaises(ValueError):
            select_examples(rows, ["missing"])


class ComparisonTests(unittest.TestCase):
    def study(self, root, seed):
        study = root / "study"
        configs = {"seed": seed, "epochs": 10, "batch_size": 16, "learning_rate": .0002,
            "cache_sha256": "same-cache", "base_checkpoint_sha256": "base",
            "initial_weights_sha256": f"seed{seed}"}
        for group in ("context", "no_context"):
            write(root / group / "config.json", {"group": group, **configs})
        write(study / "body_checkpoints.json", {"context": str(root / "context/best.pt"),
            "no_context": str(root / "no_context/best.pt"), "initial_weights_sha256": f"seed{seed}"})
        write(study / "study.json", {"completed": True, "base_checkpoint_sha256": "base", "population": {"fixed": True}})
        rows = []
        for name, *_ in groups():
            body = .7 if name in ("baseline", "d6_only") else .7 + (seed - 40) * .0001
            machine = .705 if name in ("d6_only", "both") else .7
            task2 = (body + machine) / 2
            rows.append({"group": name, "task1_r": .71, "machine_r": machine, "body_r": body,
                "task2_r": task2, "mean_r": (.71 + task2) / 2, "delta_mean_r": (.71 + task2) / 2 - .705})
        csv_write(study / "summary.csv", rows)
        return study

    def test_comparison_accepts_only_training_seed_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            before, after = self.study(root / "seed42", 42), self.study(root / "seed43", 43)
            compare(before, after, root / "comparison")
            result = read(root / "comparison/comparison.json")
            self.assertTrue(result["only_body_training_seed_changed"])
            self.assertEqual([r["body_training_seed"] for r in result["rows"]], [42, 43])

    def test_reject_changed_population_budget_and_mixed_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            before, after = self.study(root / "seed42", 42), self.study(root / "seed43", 43)
            report = read(after / "study.json")
            report["population"] = {"changed": True}
            write(after / "study.json", report)
            with self.assertRaises(ValueError):
                compare(before, after, root / "bad1")
            after = self.study(root / "seed43", 43)
            config = read(root / "seed43/context/config.json")
            config["epochs"] = 20
            write(root / "seed43/context/config.json", config)
            with self.assertRaises(ValueError):
                compare(before, after, root / "bad2")
            write(root / "seed43/no_context/config.json", {**config, "group": "no_context"})
            with self.assertRaises(ValueError):
                compare(before, after, root / "bad3")


if __name__ == "__main__":
    unittest.main()
