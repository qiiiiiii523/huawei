import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn

from baselines.B5_best.common import write, sha
from baselines.B5_slow120 import FIELDS
from baselines.B5_slow120.model import inputs
from baselines.B5_slowjoint import PROTOCOL
from baselines.B5_slowjoint.model import JointSlow, local_native, global_forward, visible_batch
from baselines.B5_slowjoint.common import initial_model, load_model
from baselines.B5_slowjoint.train import optimizer_for
from .test_joint import fixture


class Capture(nn.Module):
    def __init__(self):
        super().__init__()
        self.values = []

    def forward(self, local, features):
        self.values.append(features.detach().clone())
        return local


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def models(self):
        torch.manual_seed(43)
        add = JointSlow(dict(channels=8, heads=2, layers=1), window_batch=3).eval()
        torch.nn.init.normal_(add.global_model.output.weight, std=.01)
        mid = JointSlow(dict(channels=8, heads=2, layers=1), window_batch=3, fusion_mode="mid").eval()
        missing = mid.load_state_dict(add.state_dict(), strict=False)
        self.assertTrue(all(k.startswith("fusion.") for k in missing.missing_keys))
        return add, mid

    def test_pretrained_output_exact_and_new_branch_receives_gradients(self):
        add, mid = self.models()
        b = visible_batch(fixture(), "cpu", 100.)
        fields = {k: b[k] for k in FIELDS}
        with torch.no_grad():
            old_g = add.global_model(**inputs(b))
            new_g, _ = global_forward(mid.global_model, b["global_i"], fields)
            self.assertTrue(torch.equal(old_g, new_g))
            self.assertTrue(torch.equal(add(**b), mid(**b)))
        mid.train()
        mid(**b).square().mean().backward()
        self.assertGreater(float(mid.fusion.output.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(mid.local.input.conv.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(mid.global_model.output.weight.grad.abs().sum()), 0.)
        # Once the zero projection opens, a LOCAL loss also updates the global encoder.
        mid.zero_grad(set_to_none=True)
        torch.nn.init.normal_(mid.fusion.output.weight, std=.01)
        _, features = global_forward(mid.global_model, b["global_i"], fields)
        p = local_native(mid.local, b["anchor"], fields, 3, True, mid.fusion, features)
        p.square().mean().backward()
        self.assertGreater(float(mid.global_model.blocks[0].qkv.weight.grad.abs().sum()), 0.)

    def test_features_align_with_twelve_windows_and_do_not_mix_patients(self):
        model, _ = self.models()
        b = visible_batch(fixture(), "cpu", 100.)
        anchor = b["anchor"].repeat(2, 1, 1)
        fields = {k: b[k].repeat(2, *([1] * (b[k].ndim - 1))) for k in FIELDS}
        ramp = torch.arange(6000, dtype=torch.float32).reshape(1, 1, 6000)
        features = torch.cat((ramp, ramp + 10000), 0)
        capture = Capture()
        with torch.no_grad():
            local_native(model.local, anchor, fields, 5, False, capture, features)
        seen = torch.cat(capture.values)
        self.assertEqual(tuple(seen.shape), (24, 1, 2500))
        for patient in range(2):
            for window in range(12):
                expected = torch.arange(window * 500, window * 500 + 500) + 10000 * patient
                torch.testing.assert_close(seen[patient * 12 + window, 0, 1000:1500], expected.float())
        self.assertTrue(torch.all(seen[0, 0, :1000] == 0))
        self.assertTrue(torch.all(seen[-1, 0, 1500:] == 15999))

    def test_initializer_checkpoint_mode_and_optimizer_cover_all_parameters(self):
        add, mid = self.models()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = add.state_dict()
            from baselines.B5_best.common import state_sha
            torch.save(dict(protocol=PROTOCOL, model=state), root / "initializer.pt")
            study = dict(model=dict(channels=8, heads=2, layers=1), training=dict(window_batch=3),
                         source_g120_sha256="source", initial_weights_sha256=state_sha(state))
            initialized = initial_model(root, study, "cpu", fusion_mode="mid")
            for key, tensor in add.state_dict().items():
                self.assertTrue(torch.equal(tensor, initialized.state_dict()[key]))
            torch.nn.init.normal_(mid.fusion.output.weight, std=.02)
            write(root / "study.json", study)
            torch.save(dict(protocol=PROTOCOL, study_sha256=sha(root / "study.json"),
                source_g120_sha256="source", fusion_mode="mid", model=mid.state_dict()), root / "model.pt")
            restored, _ = load_model(root / "model.pt", study, root / "study.json", "cpu")
            self.assertEqual(restored.fusion_mode, "mid")
            self.assertEqual(state_sha(restored.state_dict()), state_sha(mid.state_dict()))
        opt = optimizer_for(mid, dict(cnn_learning_rate=1e-5, global_learning_rate=1e-4, weight_decay=.01))
        actual = [id(p) for group in opt.param_groups for p in group["params"]]
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), {id(p) for p in mid.parameters()})


if __name__ == "__main__":
    unittest.main()
