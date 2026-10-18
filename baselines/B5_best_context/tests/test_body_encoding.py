"""Encoding/data/route checks; real torch forwards run on server CPU preflight."""
from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.adapters import HistoryRecord, _visible_digest
from baselines.B5_best_context.body_features import (BODY_PROTOCOL, LEGACY_ENCODING, SHAPE_ENCODING, UNCLIPPED_ENCODING,
    relation_features, feature_encoding, shape_amplitude_stats)
from baselines.B5_best_context.body_encoding_prepare import (ENCODING_STUDY, FIXED_ARRAYS,
    check_cached_pair, check_unclipped_pair, run as prepare)
from baselines.B5_best_context.body_encoding_study import cases, check_training_pair
from baselines.B5_best_context.body_study import run_study
from . import test_context


def mock_signal():
    signal = ModuleType("scipy.signal")
    signal.butter = lambda *a, **k: np.array([1.])
    signal.sosfiltfilt = lambda f, x: x.copy()
    signal.find_peaks = lambda x, **k: (np.arange(400, 4800, 500), {})
    return {"scipy": ModuleType("scipy"), "scipy.signal": signal}


def history(gain=20.):
    i = np.sin(np.arange(10000) / 50) * 100
    raw = np.vstack([i, gain * i, -i * .3, i * .4, i * .5, i * .6])
    return HistoryRecord("body_scale_d6", "visible_history", raw, np.ones_like(raw, bool))


def make_source(root):
    root.mkdir(parents=True)
    records = []
    for split in ("train", "validation"):
        t = np.arange(60000)
        anchor = (150 * np.sin(t / 50)).astype(np.float32)
        beat, stats, available = relation_features(history())
        file = split + "/record.npz"
        (root / split).mkdir()
        np.savez_compressed(root / file, anchor=anchor, base_ii=anchor * .7, target_ii=anchor * .8,
            quality=np.ones(12, bool), beat=beat, stats=stats, available=np.array(available))
        records.append({"split": split, "subject_id": split, "pair_id": split, "target_record_id": split,
            "history_record_id": "visible_history", "history_visible_sha256": _visible_digest(history()),
            "history_available": available, "I_II_eligible": True, "file": file, "sha256": sha(root / file)})
    value = {"protocol": BODY_PROTOCOL, "completed": True, "records": records, "frozen": {"unchanged": True}}
    write(root / "cache.json", value)
    return value


class EncodingTests(unittest.TestCase):
    def test_legacy_default_unknown_encoding_and_no_hidden_input(self):
        self.assertEqual(feature_encoding({}), LEGACY_ENCODING)
        self.assertEqual(feature_encoding({"feature_encoding": SHAPE_ENCODING}), SHAPE_ENCODING)
        with self.assertRaises(ValueError):
            feature_encoding({"feature_encoding": "unknown"})
        self.assertEqual(set(inspect.signature(relation_features).parameters), {"history", "audit", "encoding"})
        with patch.dict("sys.modules", mock_signal()):
            for x, y in zip(relation_features(history()), relation_features(history(), encoding=LEGACY_ENCODING)):
                np.testing.assert_array_equal(x, y)
        with self.assertRaises(ValueError):
            relation_features(None, encoding="unknown")

    def test_separate_shape_and_amplitude_preserve_gain_without_cross_lead_clip(self):
        h = history()
        before = h.raw_uV.copy()
        audit_o, audit_e = {}, {}
        with patch.dict("sys.modules", mock_signal()):
            old = relation_features(h, audit=audit_o)
            new = relation_features(h, encoding=SHAPE_ENCODING, audit=audit_e)
            larger = relation_features(history(40), encoding=SHAPE_ENCODING)
        self.assertTrue(new[2] and larger[2])
        self.assertGreater(audit_o["II_beat_clipped_fraction"], .1)
        self.assertEqual(audit_e["II_beat_clipped_fraction"], 0)
        self.assertLess(np.max(np.abs(new[0])), 8)
        np.testing.assert_allclose(new[0], larger[0], atol=1e-6)
        np.testing.assert_allclose(new[1][:21], larger[1][:21], atol=1e-6)
        self.assertAlmostEqual(larger[1][22] - new[1][22], np.log(2), places=5)
        self.assertAlmostEqual(new[1][22], np.log(20), places=5)
        self.assertFalse(np.array_equal(old[0][1], new[0][1]))
        np.testing.assert_array_equal(h.raw_uV, before)

    def test_new_correlations_are_computed_before_amplitude_clipping(self):
        t = np.sin(np.arange(1000) / 17)
        x = np.vstack([t, 1e6 * t, -3 * t, np.zeros_like(t), .3 * t, 5 * t])
        stats = shape_amplitude_stats(x)
        self.assertTrue(np.isfinite(stats).all())
        self.assertAlmostEqual(stats[1], 1, places=6)  # I/II
        self.assertAlmostEqual(stats[2], -1, places=6)
        self.assertAlmostEqual(stats[22], np.log(1e4), places=6)
        self.assertTrue(np.all(np.abs(stats[:21]) <= 1))

    def test_E0_retains_narrow_peaks_and_changes_only_template_clip(self):
        raw = np.zeros((6, 10000))
        for start in (0, 5000):
            raw[:, start + np.arange(400, 4800, 500)] = np.arange(1, 7)[:, None] * 100
        h = HistoryRecord("body_scale_d6", "visible_narrow_peaks", raw, np.ones_like(raw, bool))
        a, b = {}, {}
        with patch.dict("sys.modules", mock_signal()):
            clipped = relation_features(h, encoding=SHAPE_ENCODING, audit=a)
            uncut = relation_features(h, encoding=UNCLIPPED_ENCODING, audit=b)
        self.assertEqual(clipped[2], uncut[2])
        np.testing.assert_array_equal(clipped[1], uncut[1])
        self.assertEqual(np.max(clipped[0]), 8)
        self.assertGreater(np.max(uncut[0]), 8)
        self.assertTrue(np.isfinite(uncut[0]).all())
        np.testing.assert_allclose(uncut[0], b["_unclipped_template"], rtol=1e-7, atol=1e-6)
        self.assertEqual(b["II_beat_clipped_fraction"], 0)
        self.assertGreater(b["II_beat_exceeds_8_fraction"], 0)
        self.assertFalse(b["template_clip_applied"])
        self.assertTrue(a["template_clip_applied"])
        self.assertEqual(a["selected_beats"], b["selected_beats"])

    def test_absent_or_masked_history_keeps_zero_features_and_same_availability(self):
        for h in (None, HistoryRecord("body_scale_d6", "invalid", np.zeros((6, 5000)), np.zeros((6, 5000), bool))):
            for encoding in (LEGACY_ENCODING, SHAPE_ENCODING, UNCLIPPED_ENCODING):
                beat, stats, available = relation_features(h, encoding=encoding)
                self.assertFalse(available or beat.any() or stats.any())
                self.assertEqual(beat.shape, (2, 300)); self.assertEqual(stats.shape, (27,))


class EncodingCacheTests(unittest.TestCase):
    def test_prepare_and_resume_reuse_original_predictions_and_live_features(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", mock_signal()):
            root = Path(folder)
            source = root / "source"
            manifest = make_source(source)
            original_sha = sha(source / "cache.json")
            config = root / "config.yaml"
            config.write_text("test", encoding="utf-8")
            modules = {name: ModuleType(name) for name in (
                "baselines.B5.config", "baselines.B5.data", "baselines.B5.metadata", "ecg12gen.dataset")}
            modules["baselines.B5.config"].load_config = lambda *a, **k: None
            modules["baselines.B5.data"].common_config = lambda c: c
            modules["baselines.B5.metadata"].subject_key = lambda s: s
            class Dataset:
                def __init__(self, cfg, task, split):
                    self.split = split
                    self._indices = list(range(12))
                    self._rows = [{"pair_id": split, "target_record_id": split, "subject_id": split,
                        "input_type": "body_scale_d6", "start_sample_500hz": str(i * 5000)} for i in range(12)]
                def __getitem__(self, i):
                    with np.load(source / self.split / "record.npz") as a:
                        part = slice(i * 5000, (i + 1) * 5000)
                        return SimpleNamespace(anchor_i_ecg=a["anchor"][None, part],
                            Y_12lead=np.vstack([a["anchor"][part], a["target_ii"][part], np.zeros((10, 5000))]),
                            target_quality_mask=a["quality"])
                def close(self):
                    pass
            modules["ecg12gen.dataset"].JointAnchorDataset = Dataset
            args = SimpleNamespace(cache=source, output=root / "new", config=config, resume=False, subjects=[])
            with patch.dict("sys.modules", modules), patch("baselines.B5_best_context.body_encoding_prepare.history_from_sample", lambda s: history()):
                prepare(args)
                (args.output / "preflight.json").unlink()  # Stop after cache commit, before report export.
                args.resume = True
                prepare(args)
                e0_args = SimpleNamespace(cache=source, output=root / "uncut", config=config,
                    resume=False, subjects=[], encoding=UNCLIPPED_ENCODING)
                prepare(e0_args)
            self.assertEqual(sha(source / "cache.json"), original_sha)
            old, new = check_cached_pair(source, root / "new")
            self.assertEqual(len(new["records"]), len(manifest["records"]))
            self.assertEqual(feature_encoding(new), SHAPE_ENCODING)
            self.assertTrue(read(root / "new/preflight.json")["same_I_predictions_targets_QC_population"])
            _, e0 = check_cached_pair(source, root / "uncut", expected_encoding=UNCLIPPED_ENCODING)
            check_unclipped_pair(root / "new", root / "uncut")
            e0_file = root / "uncut" / e0["records"][0]["file"]
            with np.load(e0_file) as a:
                bad = {k: a[k].copy() for k in a.files}
            bad["stats"][1] += .1
            np.savez_compressed(e0_file, **bad)
            e0["records"][0]["sha256"] = sha(e0_file)
            write(root / "uncut/cache.json", e0)
            with self.assertRaisesRegex(ValueError, "template clipping: stats"):
                check_unclipped_pair(root / "new", root / "uncut")
            row = new["records"][0]
            target = root / "new" / row["file"]
            with np.load(target) as a:
                arrays = {k: a[k].copy() for k in a.files}
            arrays["base_ii"][0] += 1
            np.savez_compressed(target, **arrays)
            row["sha256"] = sha(target)
            write(root / "new/cache.json", new)
            with self.assertRaisesRegex(ValueError, "changed base_ii"):
                check_cached_pair(source, root / "new")

    def test_training_pair_rejects_different_seed_budget_initial_weights_or_cache(self):
        c = {"protocol": BODY_PROTOCOL, "group": "context", "cache_sha256": "old",
            "initial_weights_sha256": "same", "epochs": 10, "seed": 42, "loss": "same"}
        e = {**c, "feature_encoding": SHAPE_ENCODING, "cache_sha256": "new"}
        check_training_pair(c, e, "old", "new")
        for key, value in (("seed", 43), ("epochs", 20), ("initial_weights_sha256", "other"),
                           ("cache_sha256", "wrong"), ("feature_encoding", LEGACY_ENCODING)):
            with self.assertRaises(ValueError):
                check_training_pair(c, {**e, key: value}, "old", "new")
        with self.assertRaises(ValueError):
            check_training_pair(c, e, "old", "new", preflight=True)
        e0 = {**e, "feature_encoding": UNCLIPPED_ENCODING, "cache_sha256": "uncut"}
        check_training_pair(c, e0, "old", "uncut", expected_encoding=UNCLIPPED_ENCODING)
        with self.assertRaises(ValueError):
            check_training_pair(c, e0, "old", "uncut")


class EncodingRouteTests(unittest.TestCase):
    def test_fourteen_groups_keep_task1_chest_and_d6_fixed(self):
        self.check_routes(False)

    def test_twenty_groups_keep_task1_chest_and_d6_fixed(self):
        self.check_routes(True)

    def check_routes(self, include_unclipped):
        class Fake:
            def __init__(self, gain):
                self.gain, self.mode = gain, "correct"
            def __call__(self, p, anchor, visible):
                p[1] += (0 if self.mode == "disabled" else self.gain) * anchor[0]
                return p, {}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            test_context.StudyTests().baseline(root / "base")
            engines = {"O": Fake(.01), "E": Fake(.02)}
            if include_unclipped:
                engines["E0"] = Fake(.03)
            def provider(row):
                return HistoryRecord(row["input_type"], row["subject_id"], history(.6).raw_uV,
                                     np.ones((6, 10000), bool))
            result = run_study(root / "base", root / "study", provider, None, None,
                case_specs=cases(include_unclipped),
                adapter_map={**engines, **{name + "_disabled": engine for name, engine in engines.items()}},
                additive_routes=[(x + "_both", "d6_only", x + "_body") for x in engines])
            self.assertEqual(len(result), 20 if include_unclipped else 14)
            values = {r["group"]: r for r in result}
            self.assertEqual(values["O_both"]["machine_r"], values["E_both"]["machine_r"])
            if include_unclipped:
                self.assertEqual(values["E0_both"]["machine_r"], values["E_both"]["machine_r"])
            self.assertTrue(all(r["task1_r"] == values["baseline"]["task1_r"] for r in result))
            self.assertTrue(read(root / "study/study.json")["device_additivity_checked"])


if __name__ == "__main__":
    unittest.main()
