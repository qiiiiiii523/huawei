"""Rate alignment, historical 5Hz parity, immutable caches and exact resume."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from baselines.B5.metadata import encode_demographics
from baselines.B5_best.common import state_sha, write, sha, read
from baselines.B5_final.profile import BASE_SHA, BODY_SHA, SAMPLING
from baselines.B5_slow120 import PROTOCOL as PARENT_PROTOCOL
from baselines.B5_slow120.common import code_sha as parent_code_sha
from baselines.B5_slow120.model import SlowCorrection, visible_inputs as old_visible, inputs as old_inputs
from baselines.B5_slow120.losses import objective
from baselines.B5_slowrate import RATES
from baselines.B5_slowrate.model import RateCorrection, visible_inputs, inputs, array_batch, positions, tokens_to_native
from baselines.B5_slowrate.common import load_study, load_correction
from baselines.B5_slowrate.prepare import prepare
from baselines.B5_slowrate import train
from baselines.B5_slowrate.summarize import patient_scores, summarize


def fixture():
    t = np.arange(60000) / 500
    anchor = (50 * np.sin(t / 6) + 20 * np.sin(2 * np.pi * 12 * t))[None].astype(np.float32)
    meta = encode_demographics({"age": 35, "sex": "male"})
    base = np.tile(anchor, (12, 1))
    return {**old_visible(anchor, 100., meta), "anchor_uV": anchor, "base_uV": base,
        "target_uV": base + 1, "base_slow": np.zeros((11, 6000), np.float32),
        "slow_target": np.tile((.3 * np.sin(np.arange(6000) / 300))[None], (11, 1)).astype(np.float32),
        "mask": np.ones(12, bool)}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_5Hz_exact_parity_with_nonzero_historical_G120(self):
        data = fixture()
        original = SlowCorrection("G120").eval()
        torch.nn.init.normal_(original.output.weight, std=.02)
        model = RateCorrection("R5").eval()
        model.load_state_dict(original.state_dict())
        fields = visible_inputs(data["anchor_uV"], 100., data, "R5")
        np.testing.assert_array_equal(fields["global_i"], data["global_i"])
        with torch.no_grad():
            old = original(**old_inputs({k: torch.from_numpy(np.asarray(v)[None].copy()) for k, v in data.items()}))
            new = model(**inputs(array_batch(fields, "cpu")))
        self.assertTrue(torch.equal(old, new))

    def test_matched_initialization_zero_output_and_training_all_rates(self):
        data, hashes = fixture(), []
        for group, hz in RATES.items():
            torch.manual_seed(43)
            model = RateCorrection(group)
            hashes.append(state_sha(model.state_dict()))
            self.assertEqual(sum(p.numel() for p in model.parameters()), 18999)
            fields = visible_inputs(data["anchor_uV"], 100., data, group)
            self.assertEqual(fields["global_i"].shape, (1, int(120 * hz)))
            batch = {k: torch.from_numpy(np.asarray(v)[None].copy()) for k, v in {**data, **fields}.items()}
            delta = model(**inputs(batch))
            self.assertEqual(tuple(delta.shape), (1, 11, 6000))
            self.assertEqual(torch.count_nonzero(delta).item(), 0)
            loss, _ = objective(delta, batch, torch.ones(12) * 100)
            loss.backward()
            self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
            with self.assertRaises(TypeError):
                model(**inputs(batch), hidden_target=batch["target_uV"])
        self.assertEqual(len(set(hashes)), 1)

    def test_interpolation_matches_physical_bin_centers_and_has_gradients(self):
        rng = np.random.default_rng(4)
        for hz in RATES.values():
            n = int(120 * hz)
            value = torch.tensor(rng.normal(size=(1, 11, n)), requires_grad=True)
            result = tokens_to_native(value, hz)
            query = (np.arange(6000) * 10 + 4.5) * hz / 500
            expected = np.stack([np.interp(query, np.arange(n), row) for row in value.detach().numpy()[0]])
            np.testing.assert_allclose(result.detach().numpy()[0], expected, atol=1e-11)
            result.sum().backward()
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertTrue((value.grad > 0).all())

    def test_position_encoding_uses_same_real_time(self):
        ref = positions(600, 5., dtype=torch.float32, device="cpu")
        low = positions(300, 2.5, dtype=torch.float32, device="cpu")
        high = positions(1200, 10., dtype=torch.float32, device="cpu")
        self.assertTrue(torch.equal(low, ref[::2]))
        self.assertTrue(torch.equal(high[::2], ref))

    def test_rate_specific_antialias_and_DC(self):
        t = np.arange(60000) / 500
        for group in RATES:
            fields = visible_inputs(2 + np.sin(2 * np.pi * 100 * t), 1., encode_demographics({}), group)
            interior = fields["global_i"][0, 20:-20]
            self.assertLess(float(interior.std()), .01)
            self.assertAlmostEqual(float(interior.mean()), 2., places=4)
        amplitudes = {}
        for group in RATES:
            fields = visible_inputs(np.sin(2 * np.pi * 2 * t), 1., encode_demographics({}), group)
            amplitudes[group] = float(fields["global_i"][0, 20:-20].std())
        self.assertLess(amplitudes["R2p5"], .05)
        self.assertGreater(amplitudes["R5"], .6)
        self.assertGreater(amplitudes["R10"], .6)


class FakeEvaluator:
    def __init__(self, *_): pass
    def close(self): pass
    def evaluate(self, model, output, group, epoch, **_):
        Path(output).mkdir(parents=True)
        return {"group": group, "epoch": epoch, "mean_r": .7 - .01 * epoch}


class WorkflowTests(unittest.TestCase):
    def parent(self, root):
        source = root / "parent"
        source.mkdir()
        data = fixture()
        np.savez(source / "record.npz", **data)
        entry = {"file": "record.npz", "sha256": sha(source / "record.npz"), "target_record_id": "record_a",
                 "subject_id": "train_a", "task": "train"}
        parent = {"protocol": PARENT_PROTOCOL, "completed": True, "code_sha256": parent_code_sha(),
            "base_sha256": BASE_SHA, "body_sha256": BODY_SHA, "sampling": SAMPLING,
            "scale_uV": [100.] * 12, "train": [entry], "validation": [], "model": {"channels": 8, "layers": 1, "heads": 2},
            "training": {"epochs": 2, "seed": 43, "batch_records": 1, "learning_rate": 1e-4,
                         "weight_decay": .01, "gradient_clip": 1., "validate_every": 1},
            "score_protocol": "synthetic", "base_checkpoint": "not_used", "body_checkpoint": "not_used",
            "config": "not_used", "config_sha256": "not_used", "baseline_evaluation": "not_used", "population": {},
            "slow_beta": .75, "reference": {k: .7 for k in ("mean_r", "task1_r", "machine_r", "body_r", "task2_r")},
            "validation_rows": {}, "train_subjects": 1,
            "validation_subjects": 0, "data_sources": {}, "loss": {}}
        write(source / "study.json", parent)
        return source

    def prepare_args(self, source, output):
        return argparse.Namespace(source_shared=str(source), output=str(output), epochs=None, validate_every=None,
                                  batch_records=None, learning_rate=None, training_seed=None)

    def test_real_sidecars_epoch0_preservation_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.parent(root)
            source_digest = sha(source / "record.npz")
            shared = root / "shared"
            prepare(self.prepare_args(source, shared))
            study = load_study(shared)
            self.assertEqual(sha(source / "record.npz"), source_digest)
            with np.load(shared / "train/00000.npz") as sidecar:
                self.assertEqual(set(sidecar.files), set(RATES))
            def args(name, resume=None):
                return argparse.Namespace(run=str(shared), output=str(root / name), group="R2p5", device="cpu", resume=resume)
            with patch.object(train, "Evaluator", FakeEvaluator):
                train.run(args("full"))
                real_save = train.atomic_save
                def interrupted(path, payload):
                    real_save(path, payload)
                    if Path(path).name == "last.pt" and payload["epoch"] == 1:
                        raise InterruptedError("Synthetic interruption after durable epoch1")
                with patch.object(train, "atomic_save", interrupted):
                    with self.assertRaises(InterruptedError):
                        train.run(args("resumed"))
                train.run(args("resumed", str(root / "resumed/last.pt")))
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(root / "resumed/last.pt", weights_only=True)
            self.assertEqual(state_sha(full["model"]), state_sha(resumed["model"]))
            self.assertEqual(full["updates"], resumed["updates"])
            best = torch.load(root / "resumed/best.pt", weights_only=True)
            self.assertEqual(best["epoch"], 0)
            self.assertEqual(best["sampling_hz"], 2.5)
            load_correction(root / "resumed/best.pt", study, shared / "study.json", "cpu")
            with (shared / "train/00000.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                load_study(shared)

    def test_patient_diagnostic_aggregates_records_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.csv"
            path.write_text("task_id,subject_id,target_record_id,pair_id,input_type,r_missing11\n"
                            "task1,p1,a,1,watch_ecg,0.5\ntask1,p1,b,2,watch_ecg,0.7\n", encoding="utf-8")
            scores, identities = patient_scores(path)
            self.assertAlmostEqual(scores[("task1", "p1", "watch_ecg")], .6)
            self.assertEqual(len(identities), 2)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("task1,p1,a,1,watch_ecg,0.5\n")
            with self.assertRaises(ValueError):
                patient_scores(path)

    def test_summary_uses_actual_scores_and_patient_contrasts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.parent(root)
            prepare(self.prepare_args(source, root / "shared"))
            for group, hz in RATES.items():
                r = {"R2p5": .71, "R5": .72, "R10": .73}[group]
                write(root / group / "completed.json", {"completed": True, "sampling_hz": hz, "updates": 2, "best_r": r})
                for name in ("best", "last"):
                    out = root / "verification" / group / name
                    row = {k: r for k in ("mean_r", "task1_r", "machine_r", "body_r", "task2_r")}
                    write(out / "evaluation.json", {"completed": True, "sampling_hz": hz, "population": {},
                          "summary": {**row, "group": group, "epoch": 2}})
                    (out / "pair_metrics.csv").write_text(
                        "task_id,subject_id,target_record_id,pair_id,input_type,r_missing11\n"
                        f"task1,p1,a,1,watch_ecg,{r}\n", encoding="utf-8")
            summarize(root, root / "comparison")
            report = read(root / "comparison/summary.json")
            self.assertEqual(report["winner_this_seed"], "R10")
            self.assertAlmostEqual(report["delta_matched_R5"]["R10"], .01)
            self.assertTrue((root / "comparison/patient_rate_contrasts.csv").exists())
            write(root / "R10/completed.json", {"completed": True, "sampling_hz": 10., "updates": 3, "best_r": .73})
            with self.assertRaises(ValueError):
                summarize(root, root / "mismatched_budget")


if __name__ == "__main__":
    unittest.main()
