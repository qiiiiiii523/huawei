"""CPU numpy integration tests; real torch forward/backward is exposed by --check-only."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.B5_best.common import write, sha
from baselines.B5_best_context.adapters import ContextRouter, HistoryRecord
from baselines.B5_best_context.body_features import (BODY_PROTOCOL, normalize_i,
    relation_features, encode_windows, restore_ii)
from baselines.B5_best_context.body_prepare import load_cache
from baselines.B5_best_context.body_study import shuffled_donors, run_study
from baselines.B5_best_context.body_train import target_macro
from . import test_context


class BodyFeatureTests(unittest.TestCase):
    def test_uV_residual_restoration_keeps_absolute_voltage_and_bound(self):
        rng = np.random.default_rng(5)
        anchor = rng.normal(1000, 230, 60000).astype(np.float32)
        base = rng.normal(-1500, 340, 60000).astype(np.float32)
        before = base.copy()
        _, encoded, scales = encode_windows(anchor, base)
        np.testing.assert_array_equal(restore_ii(base, encoded, encoded.copy(), scales), base)
        result = restore_ii(base, encoded, encoded + .2, scales)
        expected = np.repeat(.2 * scales, 5000)
        np.testing.assert_allclose(result - base, expected, atol=.0003)
        self.assertTrue(np.all(np.abs(result - base) <= np.repeat(.35 * scales, 5000)))
        np.testing.assert_array_equal(base, before)

    def test_flat_I_and_unavailable_history(self):
        signal, scale = normalize_i(np.ones(5000, np.float32) * 300)
        self.assertEqual(scale, 0)
        self.assertFalse(signal.any())
        for history in (None, HistoryRecord("body_scale_d6", "empty", None, None),
                        HistoryRecord("body_scale_d6", "masked", np.ones((6, 5000)), np.zeros((6, 5000), bool))):
            beat, stats, available = relation_features(history)
            self.assertFalse(available)
            self.assertEqual(beat.shape, (2, 300)); self.assertEqual(stats.shape, (27,))
            self.assertFalse(beat.any() or stats.any())

    def test_no_target_field_and_illegal_body_output_rejected(self):
        self.assertEqual(set(HistoryRecord.__dataclass_fields__), {"source", "record_id", "raw_uV", "valid_mask"})
        p = np.ones((12, 60000), np.float32)
        def bad(prediction, anchor, history):
            prediction[6] += 1
            return prediction, {}
        with self.assertRaises(ValueError):
            ContextRouter(body_adapter=bad).apply(p, p[:1], task_id="task2", input_type="body_scale_d6")

    def test_derangement_is_per_subject_and_repeatable(self):
        rows = [{"subject_id": str(i), "pair_id": str(i) + str(j)} for i in range(7) for j in range(2)]
        a, b = shuffled_donors(rows, 101), shuffled_donors(rows, 101)
        self.assertEqual(a, b)
        self.assertTrue(all(key != value["subject_id"] for key, value in a.items()))
        self.assertEqual(len({r["subject_id"] for r in a.values()}), 7)

    def test_target_macro_averages_pairs_before_targets(self):
        rows = [{"target_record_id": "repeated", "r_II": r} for r in (.2, .4, .6)]
        rows.append({"target_record_id": "single", "r_II": .8})
        rows.append({"target_record_id": "undefined", "r_II": None})
        self.assertAlmostEqual(target_macro(rows), .6)

    def test_cache_checks_hashes_and_patient_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "one.npz").write_bytes(b"frozen one")
            (root / "two.npz").write_bytes(b"frozen two")
            records = [{"split": s, "subject_id": s, "target_record_id": s, "file": f,
                "sha256": sha(root / f)} for s, f in (("train", "one.npz"), ("validation", "two.npz"))]
            manifest = {"completed": True, "protocol": BODY_PROTOCOL, "records": records}
            write(root / "cache.json", manifest)
            load_cache(root)
            records[1]["subject_id"] = "train"
            write(root / "cache.json", manifest)
            with self.assertRaises(ValueError):
                load_cache(root)
            records[1]["subject_id"] = "validation"
            write(root / "cache.json", manifest)
            (root / "two.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_cache(root)


class BodyStudyTests(unittest.TestCase):
    def test_all_device_groups_and_QC_reuse_one_base(self):
        # A fake target-free body adapter exercises the complete real scorer/router.
        # Scores here are synthetic and are not claims of ECG performance.
        class FakeAdapter:
            def __init__(self, context):
                self.mode, self.context = "correct", context
            def __call__(self, p, anchor, history):
                scale = .025 if not self.context or self.mode == "disabled" else .04
                p[1] += scale * anchor[0]
                return p, {"body_context_used": bool(self.context and self.mode == "correct")}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            helper = test_context.StudyTests()
            helper.baseline(root / "base")
            def history(row):
                t = np.arange(12000)
                i = np.sin(t / 55) * 200
                raw = np.vstack([i, .6 * i, np.zeros((4, len(i)))])
                return HistoryRecord(row["input_type"], row["subject_id"], raw, np.ones_like(raw, bool))
            summary = run_study(root / "base", root / "study", history, FakeAdapter(True), FakeAdapter(False))
            values = {row["group"]: row for row in summary}
            self.assertEqual(len(values), 9)
            self.assertTrue(all(row["task1_r"] == values["baseline"]["task1_r"] for row in summary))
            self.assertAlmostEqual(values["both"]["machine_r"], values["d6_only"]["machine_r"])
            self.assertAlmostEqual(values["both"]["body_r"], values["body_only"]["body_r"])
            self.assertAlmostEqual(values["both"]["mean_r"] - values["d6_only"]["mean_r"],
                values["body_only"]["mean_r"] - values["baseline"]["mean_r"])
            self.assertTrue((root / "study/body_shuffled_303/shuffled_subjects.csv").is_file())


if __name__ == "__main__":
    unittest.main()
