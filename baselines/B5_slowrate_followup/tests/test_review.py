"""Regression checks for review findings; synthetic data, no selected assets."""
import argparse
from contextlib import ExitStack, nullcontext
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from baselines.B5_best.common import write, read
from baselines.B5_slowrate.tests import test_slowrate as rate_tests
from baselines.B5_slowrate.model import RateCorrection
from baselines.B5_slowrate import evaluate as rate_eval
from baselines.B5_slowrate_followup import evaluate as follow_eval, ensemble
from baselines.B5_slowrate_followup.model import Followup
from baselines.B5_slowrate_followup.suite import Runner


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_resume_refuses_modified_completed_report_and_bound_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out, source = root / "out", root / "input.json"
            write(source, {"version": 1})
            runner = Runner(root, {"steps": {}})
            def process(*_, **__):
                write(out / "completed.json", {"completed": True})
                return SimpleNamespace(pid=123, stdout=io.StringIO("ok\n"), wait=lambda: 0)
            with patch("baselines.B5_slowrate_followup.suite.subprocess.Popen", side_effect=process):
                runner.step("test", "fake", [], out / "completed.json", output=out, bound_files=[source])
            write(out / "completed.json", {"completed": True, "changed": True})
            with self.assertRaisesRegex(ValueError, "report changed"):
                runner.step("test", "fake", [], out / "completed.json", output=out, bound_files=[source])
            write(source, {"version": 2})
            with self.assertRaisesRegex(ValueError, "input changed"):
                runner.step("test", "fake", [], out / "completed.json", output=out, bound_files=[source])

    def test_cross_seed_ensemble_uses_strongest_single_across_all_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            roots = [root / "seed43", root / "seed42"]
            parent = {"source_study_sha256": "same", "population": {}, "scale_uV": [100.] * 12,
                      "sampling": {}, "base_sha256": "base", "body_sha256": "body", "reference": {"mean_r": .7},
                      "training": {"seed": 43}}
            def selected(run, study):
                return {g: {"summary": {"mean_r": .73 if run == roots[1] and g == "R5" else .72}}
                        for g in ("R2p5", "R5", "R10")}
            for run in roots:
                write(run / "shared/study.json", parent)
                write(run / "R5/best.pt", {"synthetic": True})
            class Evaluator:
                def __init__(self, *args): pass
                def close(self): pass
                def evaluate(self, model, output, *args, **kwargs): return {"mean_r": .725}
            with patch.object(ensemble, "load_rates", return_value=parent), \
                    patch.object(ensemble, "source_checkpoints", side_effect=selected), \
                    patch.object(ensemble, "load_correction", side_effect=lambda *a: (RateCorrection("R5"), {"group": "R5"})), \
                    patch.object(ensemble, "Evaluator", Evaluator):
                ensemble.run(argparse.Namespace(rate_runs=list(map(str, roots)), groups=["R5", "R5"],
                    device="cpu", output=str(root / "merged"), keep_predictions=False))
            result = read(root / "merged/ensemble.json")
            self.assertEqual(result["reference_best_single"]["mean_r"], .73)
            self.assertAlmostEqual(result["delta_best_single"], -.005)

    def test_actual_record_evaluation_rejects_bad_epoch0_without_completed_marker(self):
        data = rate_tests.fixture()
        data["anchor50"] = data["anchor_uV"].reshape(1, 6000, 10).mean(-1) / 100
        for group in ("R2p5", "R5", "R10"):
            data["signal_" + group] = data["global_i"]
        # External patient/context readers and summary scoring are replaced;
        # the actual evaluator runs input construction, model, native->uV
        # correction, record assembly, unchanged-I checks and report writing.
        class Router:
            def apply(self, base, *args, **kwargs): return base, {}
        with tempfile.TemporaryDirectory() as folder:
            for module in (rate_eval, follow_eval):
                for failed in (False, True):
                    with self.subTest(module=module.__name__, failed=failed):
                        evaluator = object.__new__(module.Evaluator)
                        evaluator.device, evaluator.run = "cpu", Path(folder)
                        model = RateCorrection("R5", channels=8, layers=1, heads=2)
                        if module is follow_eval:
                            src = {g: RateCorrection(g, channels=8, layers=1, heads=2) for g in ("R2p5", "R5", "R10")}
                            model = Followup("CONT", src)
                        group = "R5" if module is rate_eval else "CONT"
                        entries, loaded = [], {}
                        for task in ("task1", "task2"):
                            row = {"pair_id": task, "subject_id": "p1", "target_record_id": "r1", "input_type": "watch_ecg"}
                            loaded[task] = {"rows": [row] * 12, "prediction": np.zeros((12, 12, 5000), np.float32)}
                            entries.append({**row, "task": task, "window_indices": list(range(12))})
                        evaluator.loaded = (None, loaded)
                        evaluator.provider, evaluator.router = lambda _: None, Router()
                        evaluator.study = {"scale_uV": [100.] * 12, "validation": entries,
                            "reference": {k: 0. if failed else .7 for k in
                                ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")},
                            "sampling": {}, "reference_group": "R5", "population": {"task1": {}, "task2": {}},
                            "baseline_evaluation": folder}
                        scores = {"task1_r_missing11": .7, "task2_r_missing11": .7, "main_correlation": .7}
                        overall = {"ecg_machine_d6_r_missing11": .7, "body_scale_d6_r_missing11": .7}
                        output = Path(folder) / (module.__name__ + str(failed))
                        with ExitStack() as stack:
                            stack.enter_context(patch.object(module, "record_arrays", side_effect=lambda *a: nullcontext(data)))
                            stack.enter_context(patch.object(module, "_score", return_value=(overall, {})))
                            stack.enter_context(patch.object(module, "scores_from_summaries", return_value=scores))
                            stack.enter_context(patch.object(module, "write_score_report"))
                            stack.enter_context(patch("ecg12gen.evaluate.write_report"))
                            if failed:
                                with self.assertRaisesRegex(ValueError, "Epoch0 failed"):
                                    evaluator.evaluate(model, output, group, 0, verify_zero=True)
                            else:
                                evaluator.evaluate(model, output, group, 0, verify_zero=True)
                        self.assertEqual(read(output / "evaluation.json")["completed"], not failed)
                        self.assertTrue((output / "pair_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
