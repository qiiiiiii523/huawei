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
from baselines.B5_slowjoint.model import JointSlow, visible_batch
from baselines.B5_slowjoint.losses import objective
from baselines.B5_slowjoint.public import pack_record, patient_split
from baselines.B5_slowjoint import train


def fixture():
    t = np.arange(60000) / 500
    raw = (15 + 100 * np.sin(t / 4))[None].astype(np.float32)
    fields = visible_inputs(raw, 100., encode_demographics({"age": 40, "sex": "male"}))
    return dict(fields, anchor_uV=raw, base_slow=np.zeros((11, 6000), np.float32),
        slow_target=np.tile(raw.reshape(1, 6000, 10).mean(-1) / 100, (11, 1)),
        mask=np.ones(12, bool), base_uV=np.tile(raw, (12, 1)), target_uV=np.tile(raw, (12, 1)))


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_inference_crop_and_both_gradients(self):
        data = fixture()
        b = visible_batch(data, "cpu", 100.)
        model = JointSlow(dict(channels=8, heads=2, layers=1), window_batch=3).eval()
        torch.nn.init.normal_(model.global_model.output.weight, std=.01)
        with torch.no_grad():
            old = predict_native(model.local, {"anchor": b["anchor"],
                **{k: b[k] for k in ("numeric", "sex", "field_mask", "age_topcoded")}})
            expected = old + model.global_model(**inputs(b))
            actual = model(**b)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        model.train()
        model(**b).square().mean().backward()
        self.assertGreater(float(model.local.input.conv.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.global_model.input.weight.grad.abs().sum()), 0.)
        # Hidden Y/quality cannot become inference inputs.
        changed = dict(data, target_uV=np.full((12, 60000), 99999), slow_target=np.full((11, 6000), 99999))
        for k, value in b.items():
            self.assertTrue(torch.equal(value, visible_batch(changed, "cpu", 100.)[k]))

    def test_loss_mask_and_real_duration(self):
        data = fixture()
        b = {k: torch.from_numpy(np.asarray(v)[None].copy()) for k, v in data.items()}
        b["mask"][:, 4] = False
        p = torch.zeros(1, 11, 6000, requires_grad=True)
        scales = torch.ones(12) * 100
        for public in (True, False):
            before = objective(p, b, scales, public)[0]
            b["slow_target"][:, 3] += 12345
            b["target_uV"][:, 4] += 99999
            after = objective(p, b, scales, public)[0]
            self.assertEqual(before.item(), after.item())
        meta = encode_demographics({})
        with self.assertRaises(ValueError):
            pack_record(np.zeros((12, 57600)), np.ones(12, bool), meta, np.ones(12))
        packed = pack_record(np.ones((12, 60000)) * 17, np.ones(12, bool), meta, np.ones(12))
        np.testing.assert_allclose(packed["slow_target"], 17.)
        split = patient_split(["patient" + str(i) for i in range(10)] * 2)
        self.assertEqual(split, patient_split(list(reversed(list(split)))))


class TinyJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.local, self.global_model = nn.Linear(1, 1), nn.Linear(1, 1)

    def forward(self, anchor, **_):
        x = self.local(anchor.transpose(1, 2)) + self.global_model(anchor.transpose(1, 2))
        return x.transpose(1, 2).expand(-1, 11, -1)


class FakeEvaluator:
    def __init__(self, *_):
        pass

    def close(self):
        pass

    def evaluate(self, model, folder, group, epoch, **_):
        Path(folder).mkdir(parents=True)
        return dict(mean_r=.7 - .01 * epoch, task1_r=.7, task2_r=.7)


class ResumeTests(unittest.TestCase):
    def test_public_to_huawei_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shared, records, public = root / "shared", root / "records", root / "public"
            for d in (shared, records, public):
                d.mkdir()
            data = fixture()
            for d in (records, public):
                np.savez(d / "one.npz", **data)
            entries = [dict(file="one.npz")]
            write(public / "public.json", dict(train=entries, validation=entries))
            study = dict(training=dict(seed=43, epochs=2, public_epochs=2, batch_records=1,
                cnn_learning_rate=1e-5, global_learning_rate=1e-4, weight_decay=.01,
                gradient_clip=1., validate_every=1), scale_uV=[100.] * 12,
                source_run=str(records), train=entries, source_g120_sha256="source", source_g120_epoch=20,
                public=dict(root=str(public)), reference=dict(mean_r=.7))
            write(shared / "study.json", study)

            def load(path, *_):
                cp = torch.load(path, weights_only=True)
                m = TinyJoint()
                m.load_state_dict(cp["model"])
                return m, cp

            def args(name, resume=None):
                return argparse.Namespace(run=str(shared), output=str(root / name), group="P", device="cpu", resume=resume)

            with patch.object(train, "load_study", return_value=study), patch.object(train, "initial_model", side_effect=lambda *_, **__: TinyJoint()), \
                 patch.object(train, "load_model", side_effect=load), patch.object(train, "Evaluator", FakeEvaluator):
                train.run(args("full"))
                real_save = train.atomic_save

                def interrupted(path, cp):
                    real_save(path, cp)
                    if Path(path).name == "last.pt" and cp["stage"] == "public" and cp["epoch"] == 1:
                        raise InterruptedError("synthetic interruption")

                with patch.object(train, "atomic_save", side_effect=interrupted):
                    with self.assertRaises(InterruptedError):
                        train.run(args("resumed"))
                train.run(args("resumed", str(root / "resumed/last.pt")))
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(root / "resumed/last.pt", weights_only=True)
            self.assertEqual(full["updates"], dict(public=2, huawei=2))
            self.assertEqual(full["updates"], resumed["updates"])
            self.assertEqual(state_sha(full["model"]), state_sha(resumed["model"]))
            best = torch.load(root / "resumed/best_r.pt", weights_only=True)
            self.assertEqual(best["stage"], "initial")


if __name__ == "__main__":
    unittest.main()
