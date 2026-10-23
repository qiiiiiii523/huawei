import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from baselines.B5.metadata import encode_demographics
from baselines.B5_best.common import write, state_sha
from baselines.B5_best.slow import predict_native
from baselines.B5_slow120.model import visible_inputs, inputs
from baselines.B5_slowlead.model import JointSlow, LeadDecoder, visible_batch
from baselines.B5_slowlead.losses import objective
from baselines.B5_slowlead import train


def fixture():
    t = np.arange(60000) / 500
    raw = (15 + 100 * np.sin(t / 4))[None].astype(np.float32)
    fields = visible_inputs(raw, 100., encode_demographics({"age": 40, "sex": "male"}))
    return dict(fields, anchor_uV=raw, base_slow=np.zeros((11, 6000), np.float32),
        slow_target=np.tile(raw.reshape(1, 6000, 10).mean(-1) / 100, (11, 1)),
        mask=np.ones(12, bool), base_uV=np.tile(raw, (12, 1)), target_uV=np.tile(raw, (12, 1)))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_abc_zero_init_reproduces_original_full_record(self):
        data = fixture()
        b = visible_batch(data, "cpu", 100.)
        m = JointSlow(dict(channels=8, heads=2, layers=1), window_batch=3).eval()
        nn.init.normal_(m.global_model.output.weight, std=.01)
        with torch.no_grad():
            expected = predict_native(m.local, {"anchor": b["anchor"],
                **{k: b[k] for k in ("numeric", "sex", "field_mask", "age_topcoded")}})
            expected = expected + m.global_model(**inputs(b))
            for group in ("A", "B", "C"):
                m.set_group(group)
                torch.testing.assert_close(m(**b), expected, atol=2e-6, rtol=2e-6)
        changed = dict(data, target_uV=np.full((12, 60000), 1e9),
                       slow_target=np.full((11, 6000), 1e9), mask=np.zeros(12, bool))
        for k, v in b.items():
            self.assertTrue(torch.equal(v, visible_batch(changed, "cpu", 100.)[k]))

    def test_lead_specific_decoder_and_ii_stop_gradient(self):
        d = LeadDecoder(channels=8)
        h = torch.randn(1, 8, 100, requires_grad=True)
        meta = torch.randn(1, 32)
        ii = torch.randn(1, 1, 100)
        self.assertEqual(float(d(h, meta, ii).detach().abs().sum()), 0.)
        nn.init.normal_(d.output.weight, std=.1)
        out = d(h, meta, ii)
        self.assertGreater(float((out[:, 0] - out[:, 5]).detach().abs().sum()), 0.)
        out.square().mean().backward()
        self.assertGreater(float(d.embedding.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(d.ii_encoder[0].weight.grad.abs().sum()), 0.)
        with self.assertRaises(ValueError):
            d(h, meta, ii.requires_grad_())

    def test_frozen_global_and_trainable_sets(self):
        m = JointSlow(dict(channels=8, heads=2, layers=1))
        cfg = dict(cnn_learning_rate=1e-5, decoder_learning_rate=1e-4, weight_decay=.01)
        global_ids = {id(p) for p in m.global_model.parameters()}
        for group in ("A", "B", "C"):
            m.set_group(group).train()
            self.assertFalse(m.global_model.training)
            self.assertTrue(all(not p.requires_grad for p in m.global_model.parameters()))
            opt = train.optimizer_for(m, cfg)
            active = {id(p) for g in opt.param_groups for p in g["params"]}
            self.assertFalse(active & global_ids)
            ii_ids = {id(p) for p in m.local.lead_decoder.ii_encoder.parameters()}
            self.assertEqual(bool(active & ii_ids), group == "C")
            self.assertEqual(active, {id(p) for p in m.parameters() if p.requires_grad})

    def test_c_full_forward_detaches_ii_and_updates_p_decoder_only(self):
        b = visible_batch(fixture(), "cpu", 100.)
        m = JointSlow(dict(channels=8, heads=2, layers=1), window_batch=3).set_group("C").train()
        nn.init.normal_(m.local.lead_decoder.output.weight, std=.01)
        before = state_sha(m.global_model.state_dict())
        seen = []
        hook = m.local.lead_decoder.ii_encoder.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].requires_grad))
        opt = train.optimizer_for(m, dict(cnn_learning_rate=1e-5, decoder_learning_rate=1e-4, weight_decay=.01))
        out = m(**b)
        out.square().mean().backward()
        hook.remove()
        self.assertTrue(seen)
        self.assertFalse(any(seen))
        self.assertGreater(float(m.local.input.conv.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(m.local.lead_decoder.ii_encoder[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in m.global_model.parameters()))
        opt.step()
        self.assertEqual(before, state_sha(m.global_model.state_dict()))

    def test_masked_targets_do_not_affect_training_loss(self):
        data = fixture()
        b = {k: torch.from_numpy(np.asarray(v)[None].copy()) for k, v in data.items()}
        b["mask"][:, 4] = False
        p = torch.zeros(1, 11, 6000, requires_grad=True)
        scales = torch.ones(12) * 100
        before = objective(p, b, scales)[0]
        b["slow_target"][:, 3] += 99999
        b["target_uV"][:, 4] += 99999
        self.assertEqual(before.item(), objective(p, b, scales)[0].item())


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Module()
        self.local.base = nn.Linear(1, 1)
        self.local.lead_decoder = nn.Linear(1, 1)
        self.global_model = nn.Linear(1, 1).requires_grad_(False)
        self.set_group("B")

    def set_group(self, group):
        self.group = group
        self.local.lead_decoder.requires_grad_(group != "A")
        return self

    def forward(self, anchor, **kwargs):
        x = anchor.transpose(1, 2)
        y = self.local.base(x) + self.global_model(x)
        if self.group != "A":
            y = y + self.local.lead_decoder(x)
        return y.transpose(1, 2).expand(-1, 11, -1)


class FakeEvaluator:
    def __init__(self, *args):
        pass

    def close(self):
        pass

    def evaluate(self, model, output, group, epoch, **kwargs):
        Path(output).mkdir(parents=True)
        return dict(mean_r=.7 - .01 * epoch, task1_r=.7, task2_r=.7)


class ResumeTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_and_retains_epoch_zero_best(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shared, records = root / "shared", root / "records"
            shared.mkdir()
            records.mkdir()
            for i in range(2):
                data = fixture()
                data["global_i"] *= i + 1
                data["anchor_uV"] *= i + 1
                np.savez(records / f"{i}.npz", **data)
            study = dict(training=dict(seed=43, epochs=2, batch_records=1, cnn_learning_rate=1e-5,
                decoder_learning_rate=1e-4, weight_decay=.01, gradient_clip=1., validate_every=1),
                scale_uV=[100.] * 12, source_run=str(records), train=[dict(file=f"{i}.npz") for i in range(2)],
                source_g120_sha256="source", source_g120_epoch=20, reference=dict(mean_r=.7))
            write(shared / "study.json", study)

            def load(path, *args):
                cp = torch.load(path, weights_only=True)
                m = Tiny().set_group(cp["group"])
                m.load_state_dict(cp["model"])
                return m, cp

            def args(name, resume=None):
                return argparse.Namespace(run=str(shared), output=str(root / name), group="C", device="cpu", resume=resume)

            with patch.object(train, "load_study", return_value=study), \
                 patch.object(train, "initial_model", side_effect=lambda *args: Tiny()), \
                 patch.object(train, "load_model", side_effect=load), patch.object(train, "Evaluator", FakeEvaluator):
                train.run(args("full"))
                real_save = train.atomic_save

                def interrupt(path, cp):
                    real_save(path, cp)
                    if Path(path).name == "last.pt" and cp["epoch"] == 1:
                        raise InterruptedError("test epoch boundary")

                with patch.object(train, "atomic_save", side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        train.run(args("resumed"))
                train.run(args("resumed", str(root / "resumed/last.pt")))
                with self.assertRaises(ValueError):
                    a = args("wrong", str(root / "resumed/last.pt"))
                    train.run(a)
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(root / "resumed/last.pt", weights_only=True)
            self.assertEqual(full["updates"], 4)
            self.assertEqual(full["updates"], resumed["updates"])
            self.assertEqual(state_sha(full["model"]), state_sha(resumed["model"]))
            self.assertEqual(resumed["global_optimizer_updates"], 0)
            best = torch.load(root / "resumed/best_r.pt", weights_only=True)
            self.assertEqual(best["epoch"], 0)


if __name__ == "__main__":
    unittest.main()
