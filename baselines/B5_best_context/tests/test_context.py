"""CPU synthetic tests: physical units, routing, immutable cache, tails, fallback, and QC scoring."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from baselines.B5_best_context.adapters import ContextRouter, HistoryRecord, history_from_sample, load_settings
from baselines.B5_best_context.fir import B, C, _predict, apply_postprocess, historical_fir


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.anchor = self.rng.normal(size=(1, 60000)).astype(np.float32) * 200
        self.prediction = self.rng.normal(size=(12, 60000)).astype(np.float32) * 300

    def history(self):
        i = self.rng.normal(size=20000) * 200
        coef = np.zeros(61)
        coef[30], coef[29] = .7, .2
        ii = _predict(i, (coef, .02))
        raw = np.vstack([i, ii, np.zeros((4, len(i)))])
        return HistoryRecord("ecg_machine_d6", "legal_history", raw, np.ones_like(raw, bool))

    def test_fir25_matches_donor_and_does_not_mutate_baseline(self):
        h = self.history()
        original = self.prediction.copy()
        model = historical_fir(h.raw_uV, h.valid_mask)
        self.assertTrue(model.available)
        self.assertGreater(model.holdout_r, .99)
        expected, _ = apply_postprocess(original, self.anchor, model, strength=.25)
        router = ContextRouter()
        result, status = router.apply(original, self.anchor, task_id="task2", input_type="ecg_machine_d6", history=h)
        np.testing.assert_array_equal(result, expected.astype(original.dtype))
        np.testing.assert_array_equal(original, self.prediction)
        np.testing.assert_array_equal(result[6:], original[6:])
        np.testing.assert_array_equal(result[0], original[0])
        self.assertTrue(status["personal_fir_used"])
        self.assertEqual(len(router._fir_cache), 1)
        router.apply(original, self.anchor, task_id="task2", input_type="ecg_machine_d6", history=h)
        self.assertEqual(len(router._fir_cache), 1)

    def test_missing_invalid_history_is_projection_not_identity(self):
        expected_q = (B * (self.prediction[1:6] - C * self.anchor)).sum(0) / float((B*B).sum())
        expected = (B*expected_q + C*self.anchor).astype(np.float32)
        for raw, mask in ((None, None), (np.zeros((6, 200)), np.ones((6, 200), bool)),
                          (np.ones((6, 10000)), np.zeros((6, 10000), bool)),
                          (np.full((6, 10000), np.nan), np.ones((6, 10000), bool)),
                          (np.ones((6, 10000)), np.ones((6, 10000), bool))):
            h = HistoryRecord("ecg_machine_d6", "invalid", raw, mask)
            out, status = ContextRouter().apply(self.prediction, self.anchor,
                task_id="task2", input_type="ecg_machine_d6", history=h)
            self.assertFalse(status["personal_fir_used"])
            self.assertEqual(status["fallback"], "projected_base")
            np.testing.assert_array_equal(out[1:6], expected)

    def test_other_devices_unchanged_even_after_machine_for_same_target(self):
        router = ContextRouter()
        base = self.prediction.copy()
        enhanced, _ = router.apply(base, self.anchor, task_id="task2", input_type="ecg_machine_d6", history=self.history())
        self.assertFalse(np.array_equal(enhanced[1:6], base[1:6]))
        for task, source in (("task1", "watch_ecg"), ("task2", "body_scale_d6")):
            out, status = router.apply(base, self.anchor, task_id=task, input_type=source)
            np.testing.assert_array_equal(out, self.prediction)
            self.assertEqual(status["adapter"], "identity")
        none, _ = ContextRouter(machine_mode="none").apply(base, self.anchor,
            task_id="task2", input_type="ecg_machine_d6", history=self.history())
        np.testing.assert_array_equal(none, base)

    def test_masked_history_and_physical_tail_reassembly(self):
        raw = np.full((2, 6, 5000), 1234., np.float32)
        raw[0] = np.arange(5000)
        raw[1, :, :4] = np.arange(5000, 5004)
        mask = np.ones_like(raw, bool)
        mask[1, :, 4:] = False
        mask[0, :, 120:130] = False
        sample = SimpleNamespace(context_source_type="ecg_machine_d6", context_record_id="history",
            input_quality_mask=np.array([1, 1, 0, 1, 1, 1], bool),
            context=SimpleNamespace(raw_uV=raw, time_mask=mask, starts=np.array([0, 5000]),
                valid_lengths=np.array([5000, 4]), window_mask=np.ones(2, bool)))
        h = history_from_sample(sample)
        self.assertEqual(h.raw_uV.shape, (6, 5004))
        np.testing.assert_array_equal(h.raw_uV[0], np.arange(5004))
        self.assertFalse(h.valid_mask[:, 120:130].any())
        self.assertFalse(h.valid_mask[2].any())
        self.assertTrue(h.valid_mask[0, -4:].all())
        sample.context.starts[1] = 6000
        with self.assertRaises(ValueError):
            history_from_sample(sample)

    def test_known_fir_respects_gaps_and_body_extension_is_explicit(self):
        h = self.history()
        h.raw_uV[:, 1000:2000] = np.nan
        h.valid_mask[:, 1000:2000] = False
        self.assertTrue(historical_fir(h.raw_uV, h.valid_mask).available)
        def body(prediction, anchor, history):
            prediction[1] *= .5
            return prediction, {"adapter": "future_body", "personal_fir_used": False}
        out, status = ContextRouter(body_adapter=body).apply(self.prediction, self.anchor,
            task_id="task2", input_type="body_scale_d6")
        np.testing.assert_array_equal(out[1], self.prediction[1]*.5)
        self.assertEqual(status["adapter"], "future_body")

    def test_contract_rejects_wrong_device_short_record_and_nonfinite(self):
        router = ContextRouter()
        for kwargs in ({"task_id": "task1", "input_type": "ecg_machine_d6"},
                       {"task_id": "task2", "input_type": "watch_ecg"}):
            with self.assertRaises(ValueError):
                router.apply(self.prediction, self.anchor, **kwargs)
        with self.assertRaises(ValueError):
            router.apply(self.prediction[:, :5000], self.anchor[:, :5000], task_id="task2", input_type="ecg_machine_d6")
        with self.assertRaises(ValueError):
            ContextRouter(machine_mode="fit_validation_strength")
        with self.assertRaises(ValueError):
            router.apply(self.prediction, self.anchor, task_id="task2", input_type="body_scale_d6", history=self.history())
        self.anchor[0, 3] = np.nan
        with self.assertRaises(ValueError):
            router.apply(self.prediction, self.anchor, task_id="task2", input_type="ecg_machine_d6")


class StudyTests(unittest.TestCase):
    def baseline(self, root):
        from baselines.B5_best.common import csv_write, sha, write
        from baselines.B5_best_context.evaluate import _score, pairs
        from baselines.B5.scoring import SCORE_PROTOCOL, scores_from_summaries
        rng = np.random.default_rng(4)
        i = (200*np.sin(np.arange(60000)/55)).astype(np.float32)
        q = (.7*i + 50*np.cos(np.arange(60000)/97)).astype(np.float32)
        wave = np.vstack([i, q, q-i, -(i+q)/2, i-q/2, q-i/2, *[i*.3 + q*(j+.1) for j in range(6)]])
        predicted = (wave + rng.normal(size=wave.shape)*12).astype(np.float32)
        tasks, population = {}, {}
        for task in ("task1", "task2"):
            sources = ("watch_ecg",) if task == "task1" else ("ecg_machine_d6", "body_scale_d6")
            rows, pred, target, quality = [], [], [], []
            for source in sources:
                for key in ("good", "flagged_V4", "bad_I"):
                    for start in range(0, 60000, 5000):
                        rows.append({"pair_id": source+key, "subject_id": key, "target_record_id": key,
                            "input_type": source, "start_sample_500hz": str(start), "expected_window_count": "12"})
                        pred.append(predicted[:, start:start+5000])
                        target.append(wave[:, start:start+5000])
                        mask = np.ones(12, bool)
                        mask[0] = key != "bad_I"
                        mask[9] = key != "flagged_V4"  # One bad V4; another record remains eligible.
                        quality.append(mask)
            folder = root / task
            folder.mkdir(parents=True)
            arrays = {"prediction": np.stack(pred), "target": np.stack(target), "quality": np.stack(quality)}
            for key, filename in (("prediction", "prediction_uV.npy"), ("target", "target_uV.npy"), ("quality", "target_quality_mask.npy")):
                np.save(folder / filename, arrays[key])
            csv_write(folder / "window_metadata.csv", rows)
            population[task] = {"metadata_sha256": sha(folder/"window_metadata.csv"),
                "target_sha256": sha(folder/"target_uV.npy"), "QC_sha256": sha(folder/"target_quality_mask.npy")}
            tasks[task] = _score(arrays["prediction"], {**arrays, "rows": rows}, task)[0]
        write(root / "evaluation.json", {"completed": True, "require_reliable_input": True,
            "score_protocol": SCORE_PROTOCOL, "sampling": {"solver": "heun", "steps": 16, "samples": 16, "seed": 42},
            "slow_beta": .75, "checkpoint_sha256": "synthetic_only", "population": population,
            "score": scores_from_summaries(tasks)})

    def test_three_groups_shared_targets_preserve_QC_and_report_scores(self):
        from baselines.B5_best.common import read, sha
        from baselines.B5_best_context.evaluate import load_baseline, run_study
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "original"
            self.baseline(source)
            loaded = load_baseline(source)
            before = sha(source/"task2/prediction_uV.npy")
            calls = []
            def provider(row):
                calls.append(row["pair_id"])
                return HistoryRecord("ecg_machine_d6", "unavailable", None, None)
            try:
                result = run_study(source, root/"study", provider, loaded=loaded)
                self.assertEqual(len(calls), 3)  # Once per machine pair, never 12 windows/body/watch.
                self.assertEqual([r["group"] for r in result], ["baseline", "projection", "fir25"])
                self.assertEqual(result[1]["mean_r"], result[2]["mean_r"])
                for row in result:
                    self.assertEqual(row["task1_r"], result[0]["task1_r"])
                    self.assertEqual(row["body_r"], result[0]["body_r"])
                    self.assertAlmostEqual(row["delta_mean_r"], (row["machine_r"]-result[0]["machine_r"])/4)
                report = read(root/"study/fir25/evaluation.json")
                self.assertEqual(report["fallback_pairs_including_input_qc_exclusions"], 3)
                self.assertTrue((root/"study/fir25/task2/input_qc_exclusions.csv").exists())
                self.assertTrue((root/"study/pair_comparison.csv").exists())
                self.assertEqual(sha(source/"task2/prediction_uV.npy"), before)
                with self.assertRaises(FileExistsError):
                    run_study(source, root/"study", provider, loaded=loaded)
            finally:
                for task in loaded[1].values():
                    for key in ("prediction", "target", "quality"):
                        task[key]._mmap.close()

    def test_reject_changed_population_old_QC_and_changed_sampling(self):
        from baselines.B5_best.common import read, write
        from baselines.B5_best_context.evaluate import load_baseline
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.baseline(root)
            report = read(root/"evaluation.json")
            write(root/"evaluation.json", {**report, "require_reliable_input": False})
            with self.assertRaisesRegex(ValueError, "reliable-I"):
                load_baseline(root)
            write(root/"evaluation.json", {**report, "sampling": {"solver": "heun", "steps": 4, "samples": 1}})
            with self.assertRaisesRegex(ValueError, "Heun16"):
                load_baseline(root)


if __name__ == "__main__":
    unittest.main()
