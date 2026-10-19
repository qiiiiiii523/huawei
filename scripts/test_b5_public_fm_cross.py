"""Synthetic cached-waveform integration checks; no real data/training/CUDA."""
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from baselines.B5.checkpoint import atomic_save, load_checkpoint
from baselines.B5_best.bundle import load_bundle
from baselines.B5_public_fm.common import read, write, sha, state_sha
from baselines.B5_public_fm.evaluation import HybridEvaluator
from baselines.B5_public_fm.tests.test_public_fm import prepared_fixture

spec = importlib.util.spec_from_file_location("cross", ROOT / "scripts/b5_public_fm_cross.py")
cross = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross)
torch.set_num_threads(2)


class CrossTests(unittest.TestCase):
    def test_condition_contract_rejects_different_scales(self):
        base = {k: "same" for k in ("architecture_hash", "condition_schema", "preprocessing_version", "scales_sha256")}
        changed = dict(base, scales_sha256="other")
        with self.assertRaises(ValueError):
            cross.same_condition_contract(base, changed)

    def test_summary_is_within_seed_and_does_not_select_seeds(self):
        rows = [{"candidate": n, "seed": s, "task1_r_missing11": .5 + j / 100,
                 "task2_r_missing11": .5 + j / 100, "main_correlation": .5 + j / 100}
                for s in (42, 43) for j, n in enumerate(cross.NAMES)]
        table, means = cross.summarize(rows)
        self.assertEqual(len(table), 6)
        self.assertAlmostEqual(table[2]["delta_vs_FM_C_P"], .01)
        self.assertAlmostEqual(means[2]["main_correlation"], .52)
        with self.assertRaises(ValueError):
            cross.summarize(rows[:-1])

    def test_six_cached_combinations_export_preserves_P_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, datasets = prepared_fixture(root)
            study = read(path)
            fixed = load_bundle(path.parent / "fixed_slow_bundle.pt")
            review = root / "review43"
            review.mkdir()
            write(review / "completed.json", {"completed": True})
            p_bundle = copy.deepcopy(fixed)
            p_bundle["slow"]["model"]["output.bias"] += .2
            p_bundle["slow_state_sha256"] = state_sha(p_bundle["slow"]["model"])
            p_bundle["reference_qc_r"] = None
            p_path = root / "P.pt"
            atomic_save(p_path, p_bundle)

            def fake_sample(model, inputs, keys, **kw):
                t = torch.arange(5000).float()
                # Visible temporal structure, deterministic different keyed seed results.
                y = torch.sin(t / 20) * (.2 if kw["seed"] == 42 else .21)
                return y[None, None].expand(len(keys), 12, -1).clone()

            with patch("baselines.B5_public_fm.evaluation.build_validation_datasets", return_value=datasets), \
                 patch("baselines.B5_public_fm.evaluation.sample", side_effect=fake_sample), \
                 patch("sys.stdout", io.StringIO()):
                engine = HybridEvaluator(path, study, torch.device("cpu"))
                checkpoint = load_checkpoint(path.parent / "public_initializer.pt")
                # Fake cached FM creation is strictly outside the code under test.
                from baselines.B5.checkpoint import model_from_checkpoint
                model = model_from_checkpoint(checkpoint, torch.device("cpu"))
                for name, group, epoch, legacy in (("control100", None, 0, "stable_baseline"),
                        ("FM_C_best100", "control", 100, "FM-C_best100"), ("FM_P_best45", "expanded", 45, "FM-P_best45")):
                    original = root / "baseline" if group is None else root / group / "huawei/evaluations" / f"epoch_{epoch:04d}"
                    score = engine.evaluate(model, original)
                    if group is None:
                        digest = sha(path.parent / "fixed_slow_bundle.pt")
                    else:
                        folder = root / group / "huawei"
                        cp = copy.deepcopy(checkpoint)
                        cp.update(epoch=epoch, validation_score=score)
                        candidate = folder / "candidates" / f"epoch_{epoch:04d}.pt"
                        atomic_save(candidate, cp)
                        digest = sha(candidate)
                        write(folder / "completed.json", {"completed": True, "study_sha256": sha(path),
                            "best": {"epoch": epoch, "path": str(candidate.relative_to(folder)), "sha256": digest, "score": score}})
                    engine.study["settings"]["sampling"]["seed"] = 43
                    engine.evaluate(model, review / legacy)
                    write(review / legacy / "source.json", {"source_sha256": digest})
                    engine.study["settings"]["sampling"]["seed"] = 42

                args = SimpleNamespace(run=root, seed43_review=review, slow_bundle=p_path, output=root / "cross", device="cpu")
                with patch.object(HybridEvaluator, "generate", side_effect=AssertionError("No FM regeneration allowed")):
                    cross.run(args)
                done = read(root / "cross/completed.json")
                self.assertTrue(done["completed"])
                self.assertFalse(done["FM_regenerated"])
                self.assertEqual(len(done["rows"]), 6)
                for name in cross.NAMES:
                    bundle = load_bundle(root / "cross/bundles" / (name + "__P.pt"))
                    self.assertEqual(bundle["slow_state_sha256"], p_bundle["slow_state_sha256"])
                    self.assertEqual(bundle["sampling"]["seed"], 42)
                    self.assertEqual(bundle["reference_qc_r"]["task1_r_missing11"],
                        read(root / "cross/seed42" / name / "correlation_score.json")["task1_r_missing11"])
                pop = read(root / "baseline/evaluation.json")["population"]
                with self.assertRaises(ValueError):
                    cross.check_cache(review / "stable_baseline", 42, sha(path.parent / "fixed_slow_bundle.pt"),
                        path, study, pop, fixed["slow_state_sha256"])
                tamper = review / "stable_baseline/task1/fm_prediction_uV.npy"
                x = np.load(tamper); x[0, 0, 0] += 1; np.save(tamper, x)
                with self.assertRaises(ValueError):
                    cross.check_cache(review / "stable_baseline", 43, sha(path.parent / "fixed_slow_bundle.pt"),
                        path, study, pop, fixed["slow_state_sha256"])


if __name__ == "__main__":
    unittest.main()
