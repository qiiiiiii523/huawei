import importlib.util
import unittest
import tempfile
import argparse
import json
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class RuntimeTests(unittest.TestCase):
    def test_v6_load_forward_and_mask_fallback(self):
        import torch
        from baselines.B5.model import B5UNet
        from baselines.B5.config import ModelConfig
        from baselines.context_d6_v6 import ContextD6V6UNet, load_b5_weights
        torch.set_num_threads(2)
        cfg = ModelConfig(channels=(8, 16, 32), time_dim=8, metadata_dim=8, metadata_dropout=0., metadata_field_dropout=0.)
        base, v6 = B5UNet(cfg).eval(), ContextD6V6UNet(cfg).eval()
        load_b5_weights(v6, base.state_dict())
        anchor = torch.randn(1, 1, 64)
        args = (anchor, torch.zeros(1, 3), torch.tensor([2]), torch.zeros(1, 4, dtype=torch.bool), torch.zeros(1, 1))
        kwargs = dict(context=torch.randn(1, 2, 6, 5000), context_time_mask=torch.ones(1, 2, 6, 5000, dtype=torch.bool),
                      context_window_mask=torch.ones(1, 2, dtype=torch.bool), context_window_weights=torch.ones(1, 2) / 2)
        state, t = torch.randn(1, 11, 64), torch.tensor([.5])
        with torch.no_grad():
            before = base.velocity(state, t, base.encode_conditions(*args))
            after = v6.velocity(state, t, v6.encode_conditions(*args, **kwargs))
        self.assertTrue(torch.equal(before, after))
        # Enable learned branches to ensure missing-history masking, not just zero init, protects fallback.
        with torch.no_grad():
            v6.context_cross_gate.fill_(.2)
            for x in v6.context_projection: x.bias.fill_(.1)
        kwargs["context_time_mask"].zero_(); kwargs["context_window_mask"].zero_(); kwargs["context_window_weights"].zero_()
        with torch.no_grad():
            empty = v6.velocity(state, t, v6.encode_conditions(*args, **kwargs))
        self.assertTrue(torch.allclose(before, empty, atol=1e-6))
        v6.train(); kwargs["context_time_mask"].fill_(True); kwargs["context_window_mask"].fill_(True); kwargs["context_window_weights"].fill_(.5)
        v6.velocity(state, t, v6.encode_conditions(*args, **kwargs)).square().mean().backward()
        self.assertIsNotNone(v6.stem.weight.grad)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in v6.context_encoder.parameters() if p.grad is not None), 0.)

    def test_history_mask_scaling_and_target_free_model_inputs(self):
        import numpy as np
        from baselines.B5_context_public.data import encode_history, model_inputs
        raw = np.tile(np.arange(5001, dtype=np.float32), (6, 1))
        mask = np.ones_like(raw, bool); mask[:, 20] = False; raw[:, 20] = np.nan
        d = encode_history(raw, mask, np.ones(12))
        self.assertTrue(np.isfinite(d["context"]).all())
        self.assertFalse(d["context_time_mask"][0, :, 20].any())
        self.assertFalse(d["context_time_mask"][1, :, 1:].any())
        self.assertAlmostEqual(float(d["context_window_weights"].sum()), 1.)
        self.assertNotIn("target", model_inputs(dict(anchor=raw, target=raw, quality_mask=mask)))

    def test_two_stage_resume_matches_uninterrupted(self):
        import torch
        import numpy as np
        from baselines.B5_context_public import train
        from baselines.B5_context_public.sampling import SAMPLING
        from baselines.B5_best.common import write, sha
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([.1]))
        def fake_loss(m, batch, *_):
            loss = (m.weight - torch.randn_like(m.weight)).square().mean()
            return loss, {}
        def fake_eval(m, root, study, output, *_):
            Path(output).mkdir(parents=True)
            return dict(machine_r=-float(m.weight.detach().abs().sum()), mean_r=0.)
        record = dict(anchor_uV=np.ones((1, 5000), np.float32), target_uV=np.ones((12, 5000), np.float32),
            mask=np.ones(12, bool), numeric=np.zeros(3, np.float32), sex=np.array(2), field_mask=np.zeros(4, bool), age_topcoded=np.zeros(1, np.float32))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); shared = root / "shared"; shared.mkdir(); write(shared / "study.json", {})
            ref = dict(completed=True, route="ecg_machine_d6", full_record_seconds=120, slow_beta=.75, sampling=SAMPLING, machine_r=-.1)
            for k in ("population_sha256", "record_keys_sha256", "scales_sha256", "slow_assets_sha256", "terminal_rules_sha256", "checkpoint_sha256"): ref[k] = 'a' * 64
            references = root / "references.json"
            write(references, dict(completed=True, study_sha256=sha(shared / "study.json"), reports={k:ref for k in
                ("public_initial", "control_initial", "context_public_initial", "stable_machine")}))
            study = dict(scale_uV=[1.] * 12, loss={}, records={k:[{}, {}, {}] for k in ("public_train", "huawei_train")},
                training=dict(seed=43, public_epochs=2, huawei_epochs=2, validate_every=1, learning_rate=.01,
                    weight_decay=.01, ema_decay=.9, batch_records=2, gradient_clip=1.))
            def args(name, resume=None): return argparse.Namespace(run=str(shared), output=str(root/name), references=str(references),
                arm="control", device="cpu", resume=resume)
            with patch.object(train, "load_study", return_value=study), patch.object(train, "create_model", side_effect=lambda *args: Tiny()), \
                 patch.object(train, "read_record", return_value=record), patch.object(train, "flow_loss", side_effect=fake_loss), \
                 patch.object(train, "evaluate", side_effect=fake_eval):
                train.run(args("full"))
                original_save = train.atomic_save
                def stop(path, value):
                    original_save(path, value)
                    if Path(path).name == 'last.pt' and value['stage']=='public' and value['epoch']==1:
                        raise InterruptedError('synthetic durable epoch interruption')
                with patch.object(train, "atomic_save", side_effect=stop):
                    with self.assertRaises(InterruptedError): train.run(args("resume"))
                train.run(args("resume", str(root / 'resume/last.pt')))
            full = torch.load(root / 'full/last.pt', weights_only=True)
            resumed = torch.load(root / 'resume/last.pt', weights_only=True)
            self.assertEqual(full['updates'], resumed['updates'])
            self.assertTrue(torch.equal(full['model']['weight'], resumed['model']['weight']))
            self.assertTrue(torch.equal(full['ema']['weight'], resumed['ema']['weight']))


if __name__ == "__main__": unittest.main()
