"""CPU synthetic checks for visible features, read-only clipping audit and gate study routing."""
from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from baselines.B5_best.common import read
from baselines.B5_best_context.adapters import HistoryRecord
from baselines.B5_best_context.body_features import clipping_summary, relation_features
from baselines.B5_best_context.body_gate_features import (visible_record_features,
    fit_normalization, normalize_features, GATE_PROTOCOL)
from baselines.B5_best_context.body_gate_study import cases, check_pair
from baselines.B5_best_context.body_study import run_study
from . import test_context


class GateFeatureTests(unittest.TestCase):
    def example(self):
        t = np.arange(60000)
        i = (150 * np.sin(t / 50) + 20 * np.cos(t / 4000)).astype(np.float32)
        return i, i * .8, np.ones((2, 300), np.float32), np.arange(27, dtype=np.float32), True

    def test_only_visible_inputs_and_one_feature_vector_per_record(self):
        self.assertEqual(list(inspect.signature(visible_record_features).parameters),
            ["anchor", "predicted_ii", "beat", "stats", "available"])
        args = self.example()
        before = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
        value = visible_record_features(*args)
        self.assertEqual(value.shape, (48,))
        self.assertTrue(np.isfinite(value).all())
        np.testing.assert_array_equal(value, visible_record_features(*args))
        for a, b in zip(args, before):
            np.testing.assert_array_equal(a, b)
        disabled = visible_record_features(*args[:4], False)
        np.testing.assert_array_equal(disabled[:39], np.zeros(39))

    def test_normalization_is_fitted_once_on_training_features(self):
        first = visible_record_features(*self.example())
        second = first + .1
        normal = fit_normalization([first, second])
        normalized = normalize_features(np.stack([first, second]), normal)
        np.testing.assert_allclose(normalized.mean(0), 0, atol=1e-5)
        saved = str(normal)
        normalize_features(first + 10000, normal)
        self.assertEqual(str(normal), saved)
        self.assertEqual(normal["training_records"], 2)

    def test_clip_audit_measures_preprocessing_not_equal_boundary(self):
        x = np.zeros((2, 2, 300), np.float32)
        x[:, 0, :10] = 8
        x[:, 1, :10] = 12
        result = clipping_summary(x)
        self.assertEqual(result["I_beat_clipped_fraction"], 0)
        self.assertAlmostEqual(result["II_beat_clipped_fraction"], 10 / 300)
        self.assertEqual(result["II_peak_before_clip"], 12)

    def test_audit_path_exactly_preserves_original_features(self):
        # Mock event detection only: exercise real normalization/clipping/audit code.
        signal = ModuleType("scipy.signal")
        signal.butter = lambda *a, **k: np.array([1.])
        signal.sosfiltfilt = lambda f, x: x.copy()
        signal.find_peaks = lambda x, **k: (np.arange(400, 4800, 500), {})
        t = np.arange(10000)
        i = np.sin(t / 50) * 100
        raw = np.vstack([i, i * 10, i * .2, i * .3, i * .4, i * .5])
        history = HistoryRecord("body_scale_d6", "input_history", raw, np.ones_like(raw, bool))
        with patch.dict("sys.modules", {"scipy": ModuleType("scipy"), "scipy.signal": signal}):
            original = relation_features(history)
            report = {}
            audited = relation_features(history, audit=report)
        for a, b in zip(original, audited):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(audited[2])
        self.assertGreater(report["II_beat_clipped_fraction"], 0)
        self.assertEqual(report["I_beat_clipped_fraction"], 0)

    def test_paired_gate_configs_allow_only_C_G_difference(self):
        common = {"protocol": GATE_PROTOCOL, "init_body_checkpoint_sha256": "source", "cache_sha256": "cache",
            "initial_weights_sha256": "same", "seed": 42, "epochs": 10, "learning_rate": .0002}
        c, g = {**common, "record_gate": "C"}, {**common, "record_gate": "G"}
        check_pair(c, g, "source", "cache")
        with self.assertRaises(ValueError):
            check_pair(c, {**g, "epochs": 20}, "source", "cache")
        with self.assertRaises(ValueError):
            check_pair(c, g, "wrong_source", "cache")


class GateStudyTests(unittest.TestCase):
    def test_all_twelve_groups_preserve_d6_and_unrelated_leads(self):
        class Fake:
            def __init__(self, alpha):
                self.alpha, self.mode = alpha, "correct"
            def __call__(self, p, anchor, history):
                p[1] += self.alpha * .02 * anchor[0]
                return p, {"record_alpha": self.alpha}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            test_context.StudyTests().baseline(root / "base")
            def provider(row):
                i = np.sin(np.arange(12000) / 55) * 200
                raw = np.vstack([i, .6 * i, np.zeros((4, len(i)))])
                return HistoryRecord(row["input_type"], row["subject_id"], raw, np.ones_like(raw, bool))
            source, control, gated = Fake(1), Fake(1), Fake(.5)
            summary = run_study(root / "base", root / "study", provider, None, None,
                case_specs=cases(), adapter_map={"source": source, "C": control, "G": gated, "G_disabled": gated},
                additive_routes=[(x + "_both", "d6_only", x + "_body") for x in ("source", "C", "G")],
                notes=["Synthetic gate routing test"])
            self.assertEqual(len(summary), 12)
            rows = {r["group"]: r for r in summary}
            self.assertEqual(rows["source_both"]["machine_r"], rows["G_both"]["machine_r"])
            self.assertEqual(rows["C_both"]["machine_r"], rows["G_both"]["machine_r"])
            self.assertTrue(all(r["task1_r"] == rows["baseline"]["task1_r"] for r in summary))
            self.assertTrue(read(root / "study/study.json")["device_additivity_checked"])


if __name__ == "__main__":
    unittest.main()
