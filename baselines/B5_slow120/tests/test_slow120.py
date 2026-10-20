"""Synthetic tests, no patient data or selected model assets needed."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from baselines.B5.metadata import encode_demographics
from baselines.B5_best.common import state_sha, write, sha
from baselines.B5_best.combine import upsample_native as numpy_upsample
from baselines.B5_final.profile import BASE_SHA, BODY_SHA, SAMPLING
from baselines.B5_slow120 import PROTOCOL
from baselines.B5_slow120.model import (SlowCorrection, visible_inputs, inputs, tokens_to_native,
    upsample_native, correct_base)
from baselines.B5_slow120.evaluate import array_batch
from baselines.B5_slow120.losses import objective
from baselines.B5_slow120.common import code_sha, load_study
from baselines.B5_slow120.prepare import group_train_rows
from baselines.B5_slow120 import train


def fixture():
    t = np.arange(60000) / 500
    anchor = (50 * np.sin(t / 6) + 20 * np.sin(2 * np.pi * 12 * t))[None].astype(np.float32)
    fields = visible_inputs(anchor, 100., encode_demographics({"age": 35, "sex": "male"}))
    base = np.tile(anchor, (12, 1))
    slow = np.tile((.3 * np.sin(np.arange(6000) / 300))[None], (11, 1)).astype(np.float32)
    return {**fields, "anchor_uV": anchor, "base_uV": base, "target_uV": base + 1,
            "base_slow": np.zeros((11, 6000), np.float32), "slow_target": slow,
            "mask": np.ones(12, bool)}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_matched_zero_init_and_unchanged_I(self):
        data = fixture()
        torch.manual_seed(43); local = SlowCorrection("G50")
        torch.manual_seed(43); full = SlowCorrection("G120")
        self.assertEqual(state_sha(local.state_dict()), state_sha(full.state_dict()))
        for model in (local, full):
            delta = model(**inputs(array_batch(data, "cpu")))
            self.assertEqual(tuple(delta.shape), (1, 11, 6000))
            self.assertEqual(torch.count_nonzero(delta).item(), 0)
            enhanced = correct_base(data["base_uV"], delta[0].detach().numpy(), [100.] * 12)
            np.testing.assert_array_equal(enhanced, data["base_uV"])

    def test_local_filter_has_no_outside50_leakage(self):
        raw = fixture()["anchor_uV"]
        modified = raw.copy(); modified[:, 40000:45000] += 1000  # 80..90s
        a = visible_inputs(raw, 100., encode_demographics({}))
        b = visible_inputs(modified, 100., encode_demographics({}))
        np.testing.assert_array_equal(a["local_i"][:5], b["local_i"][:5])  # through context ending70s
        self.assertGreater(np.max(np.abs(a["global_i"] - b["global_i"])), 1.)
        torch.manual_seed(9)
        model = SlowCorrection("G120").eval()
        torch.nn.init.normal_(model.output.weight, std=.02)
        with torch.no_grad():
            pa = model(**inputs(array_batch(a, "cpu")))
            pb = model(**inputs(array_batch(b, "cpu")))
        self.assertGreater(float((pa[..., :500] - pb[..., :500]).abs().max()), 1e-7)

    def test_temporal_gradient_dependency(self):
        data = fixture()
        for group in ("G50", "G120"):
            model = SlowCorrection(group)
            torch.nn.init.normal_(model.output.weight, std=.02)
            batch = array_batch(data, "cpu")
            batch["global_i"].requires_grad_(); batch["local_i"].requires_grad_()
            model(**inputs(batch))[..., :500].sum().backward()
            if group == "G120":
                self.assertGreater(float(batch["global_i"].grad[..., -1].abs().sum()), 0.)
                self.assertIsNone(batch["local_i"].grad)
            else:
                self.assertIsNone(batch["global_i"].grad)
                self.assertGreater(float(batch["local_i"].grad[:, 0].abs().sum()), 0.)
                self.assertEqual(float(batch["local_i"].grad[:, 1:].abs().sum()), 0.)

    def test_interpolation_alignment(self):
        rng = np.random.default_rng(42)
        for local, n in ((False, 600), (True, 250)):
            tokens = rng.normal(size=(2, 11, n))
            value = tokens_to_native(torch.tensor(tokens), local).numpy()
            query = (np.arange(500 if local else 6000) * 10 + 4.5) / 100 + (100 if local else 0)
            expected = np.stack([np.stack([np.interp(query, np.arange(n), row) for row in item]) for item in tokens])
            np.testing.assert_allclose(value, expected, atol=1e-12)
        native = rng.normal(size=(11, 6000))
        actual = upsample_native(torch.tensor(native)).numpy()
        np.testing.assert_allclose(actual, numpy_upsample(native), atol=1e-12)

    def test_antialias_and_DC(self):
        t = np.arange(60000) / 500
        raw = 2 + np.sin(2 * np.pi * 100 * t)
        data = visible_inputs(raw, 1., encode_demographics({}))
        self.assertLess(float(data["global_i"][0, 20:-20].std()), .01)
        self.assertAlmostEqual(float(data["global_i"][0, 20:-20].mean()), 2., places=4)

    def test_training_and_masked_target_invariance(self):
        data = fixture()
        batch = {k: torch.from_numpy(np.asarray(v)[None].copy()) for k, v in data.items()}
        batch["mask"][:, 4] = False
        model = SlowCorrection("G120")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        fixed = batch["base_uV"].clone()
        for _ in range(2):
            optimizer.zero_grad()
            delta = model(**inputs(batch))
            loss, _ = objective(delta, batch, torch.ones(12) * 100)
            loss.backward(); optimizer.step()
        self.assertGreater(float(model.output.weight.detach().abs().sum()), 0.)
        self.assertTrue(torch.equal(fixed, batch["base_uV"]))
        delta = model(**inputs(batch))
        before = objective(delta, batch, torch.ones(12) * 100)[0]
        batch["slow_target"][:, 3] += 10000
        batch["target_uV"][:, 4] += 10000
        after = objective(delta, batch, torch.ones(12) * 100)[0]
        self.assertEqual(before.item(), after.item())
        with self.assertRaises(TypeError):
            model(**inputs(batch), hidden_target=batch["target_uV"])

    def test_real_chronological_train_windows_only(self):
        rows = [{"target_record_id": "a", "subject_id": "p", "start_sample_500hz": str(s)}
                for s in range(0, 60000, 5000)]
        self.assertEqual(len(group_train_rows(rows)), 1)
        with self.assertRaises(ValueError):
            group_train_rows(rows[:-1])
        with self.assertRaises(ValueError):
            group_train_rows(rows + [rows[0]])


class FakeEvaluator:
    """Only external patient evaluation is replaced in the resume integration test."""
    def __init__(self, *_): pass
    def close(self): pass
    def evaluate(self, model, output, group, epoch, **_):
        Path(output).mkdir(parents=True)
        return {"group": group, "epoch": epoch, "mean_r": .7 - .01 * epoch}


class TrainingIntegrationTests(unittest.TestCase):
    def test_epoch0_preserved_and_resume_is_bit_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); shared = root / "shared"; shared.mkdir()
            data = fixture()
            np.savez(shared / "record.npz", **data)
            torch.manual_seed(43)
            model = SlowCorrection("G120")
            study = {"protocol": PROTOCOL, "completed": True, "base_sha256": BASE_SHA, "body_sha256": BODY_SHA,
                "sampling": SAMPLING, "code_sha256": code_sha(), "model": {"channels": 32, "layers": 2, "heads": 4},
                "initial_weights_sha256": state_sha(model.state_dict()), "scale_uV": [100.] * 12,
                "training": {"epochs": 2, "seed": 43, "batch_records": 1, "learning_rate": 1e-4,
                    "weight_decay": .01, "gradient_clip": 1., "validate_every": 1},
                "train": [{"file": "record.npz", "sha256": sha(shared / "record.npz")}], "validation": []}
            write(shared / "study.json", study)
            def args(name, resume=None):
                return argparse.Namespace(run=str(shared), output=str(root / name), group="G120", device="cpu", resume=resume)
            with patch.object(train, "Evaluator", FakeEvaluator):
                train.run(args("full"))
                real_save = train.atomic_save
                def interrupted(path, payload):
                    real_save(path, payload)
                    if Path(path).name == "last.pt" and payload["epoch"] == 1:
                        raise InterruptedError("synthetic interruption after durable epoch1 checkpoint")
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
            self.assertEqual(best["best_epoch"], 0)
            self.assertEqual(state_sha(best["model"]), study["initial_weights_sha256"])
            with (shared / "record.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                load_study(shared)


if __name__ == "__main__":
    unittest.main()
