"""New scoring defaults; old bundle references remain explicitly historical."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5_best import SAMPLING, SCORE_PROTOCOL as OLD_PROTOCOL
from baselines.B5_best.common import read
from baselines.B5_best.validate import evaluate, reference_settings
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best.select import select
from ecg12gen.evaluate import correlation_score


class ReliableHybridTests(unittest.TestCase):
    def datasets(self):
        raw = np.tile(np.sin(np.arange(60000) / 30).astype(np.float32), (12, 1))
        class Dataset:
            def __init__(self, task):
                devices = ["watch_ecg"] if task == "task1" else ["ecg_machine_d6", "body_scale_d6"]
                self.rows = [{"pair_id": dev + key, "target_record_id": key, "subject_id": key,
                              "input_type": dev, "start_sample_500hz": str(start), "expected_window_count": "12"}
                             for dev in devices for key in ("good", "bad") for start in range(0, 60000, 5000)]
                self.demographics = SimpleNamespace(get=lambda _: {"numeric": np.array([0.])})
                self.base = Base(self.rows)
            def __len__(self): return len(self.rows)
            def __getitem__(self, i):
                row = self.rows[i]
                start = int(row["start_sample_500hz"])
                mask = np.ones(12, bool)
                mask[0] = row["target_record_id"] == "good"
                return {"target_uV": raw[:, start:start + 5000], "quality_mask": mask, "evaluation_metadata": row}
        class Base:
            def __init__(self, rows): self.rows = rows
            def __getitem__(self, i):
                start = int(self.rows[i]["start_sample_500hz"])
                return SimpleNamespace(anchor_i_ecg=raw[:1, start:start + 5000])
        class Engine:
            def __init__(self, *unused): pass
            def predict_encoded(self, anchor, meta, key, batch_size):
                return np.tile(anchor, (12, 1)) * (1 if key == "good" else -1)
        return {task: Dataset(task) for task in ("task1", "task2")}, Engine

    def bundle(self):
        return {"sampling": SAMPLING, "slow_beta": .75, "score_protocol": OLD_PROTOCOL,
                "reference_qc_r": {"task1_r_missing11": 0., "task2_r_missing11": 0.},
                "reference_tolerance": 2e-5, "provenance": {}}

    def test_new_default_filters_bad_I_but_does_not_compare_old_reference(self):
        datasets, engine = self.datasets()
        with tempfile.TemporaryDirectory() as tmp, patch("baselines.B5_best.validate.Predictor", engine):
            output = Path(tmp) / "new"
            score = evaluate(self.bundle(), datasets, torch.device("cpu"), output, "original_bundle")
            self.assertEqual(score["score_protocol"], SCORE_PROTOCOL)
            self.assertAlmostEqual(score["main_correlation"], 1)
            report = read(output / "evaluation.json")
            self.assertIsNone(report["reference_matched"])
            self.assertFalse(report["reference_scope_matched"])
            self.assertTrue((output / "task2/input_qc_exclusions.csv").exists())

    def test_old_reference_requires_explicit_historical_mode(self):
        datasets, engine = self.datasets()
        with tempfile.TemporaryDirectory() as tmp, patch("baselines.B5_best.validate.Predictor", engine):
            with self.assertRaisesRegex(ValueError, "legacy-target-qc-only"):
                evaluate(self.bundle(), datasets, torch.device("cpu"), Path(tmp)/"refused", "old", verify_reference=True)
            self.assertFalse((Path(tmp)/"refused").exists())
            score = evaluate(self.bundle(), datasets, torch.device("cpu"), Path(tmp)/"old", "old",
                             verify_reference=True, legacy_target_qc_only=True)
            self.assertEqual(score["score_protocol"], OLD_PROTOCOL)
            self.assertAlmostEqual(score["main_correlation"], 0)
            self.assertTrue(read(Path(tmp)/"old/evaluation.json")["reference_matched"])

    def test_new_reference_can_verify_same_weights_under_new_scope(self):
        datasets, engine = self.datasets()
        reference = {"score_protocol": SCORE_PROTOCOL, "task1_r_missing11": 1., "task2_r_missing11": 1.}
        with tempfile.TemporaryDirectory() as tmp, patch("baselines.B5_best.validate.Predictor", engine):
            evaluate(self.bundle(), datasets, torch.device("cpu"), Path(tmp)/"new", "old_weights",
                     verify_reference=True, reference_report=reference)
            self.assertTrue(read(Path(tmp)/"new/evaluation.json")["reference_matched"])
        with self.assertRaisesRegex(ValueError, "different input-QC"):
            reference_settings(self.bundle(), reference_report={**reference, "score_protocol": OLD_PROTOCOL})

    def test_ranking_accepts_new_protocol_but_rejects_mixed_population_protocols(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, protocol, r in (("new_a", SCORE_PROTOCOL, .7), ("new_b", SCORE_PROTOCOL, .8),
                                      ("old", OLD_PROTOCOL, .9)):
                folder = root / name
                folder.mkdir()
                report = {"completed": True, "score_protocol": protocol, "population": {"QC": "same"},
                          "sampling": SAMPLING, "checkpoint_sha256": name,
                          "score": correlation_score(r, r, score_protocol=protocol)}
                (folder / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
            ranked = select([root / "new_a", root / "new_b"], root / "rank")
            self.assertEqual(ranked[0]["checkpoint_sha256"], "new_b")
            with self.assertRaises(ValueError):
                select([root / "new_a", root / "old"], root / "mixed")


if __name__ == "__main__":
    unittest.main()
