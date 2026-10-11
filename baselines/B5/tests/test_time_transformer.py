from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from baselines.B5.checkpoint import atomic_save, initialize_model, load_checkpoint, model_from_checkpoint
from baselines.B5.config import ModelConfig, load_config
from baselines.B5.flow import sample
from baselines.B5.model import B5UNet
from baselines.B5.tests.test_b5 import checkpoint, inputs, preprocessor, tiny_model
from baselines.B5.time_transformer import TemporalResidual, resize_linear_1d

ROOT = Path(__file__).resolve().parents[3]


def temporal_model():
    base = tiny_model()
    cfg = replace(base.config, temporal_enabled=True, temporal_tokens=8, temporal_dimension=16,
                  temporal_heads=4, temporal_layers=2, temporal_ffn=32)
    return base, B5UNet(cfg).eval()


class TemporalTests(unittest.TestCase):
    def test_legacy_hash_and_config_preserved(self):
        cfg = ModelConfig()
        legacy = cfg.core_dict
        expected = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
        self.assertEqual(cfg.fingerprint, expected)
        self.assertEqual(cfg.fingerprint, '637df1317e18638abc52330edb754308bebd802803ad5bb349a853d2f52afd46')
        self.assertEqual(cfg.checkpoint_dict, legacy)
        self.assertNotEqual(replace(cfg, temporal_enabled=True).fingerprint, expected)
        for overrides in [{'temporal_dimension': 15}, {'temporal_tokens': 0},
                          {'temporal_layers': True}, {'temporal_dropout': 1.}]:
            with self.assertRaises(ValueError):
                replace(cfg, **overrides)

    def test_zero_projection_preserves_backbone_outputs_and_sampling(self):
        base, model = temporal_model()
        audit = initialize_model(model, checkpoint(base, preprocessor()))
        self.assertEqual(audit['mode'], 'B5_to_B5T_zero_residual')
        self.assertTrue(all(key.startswith('temporal_branch.') for key in audit['new_keys']))
        x = torch.randn(1, 11, 64); c = inputs(1, 64); t = torch.tensor([.3])
        with torch.no_grad():
            first = base(x, t, **c); second = model(x, t, **c)
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(sample(base, c, ['record:0'], steps=2, samples=2),
                                       sample(model, c, ['record:0'], steps=2, samples=2), rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_missing_extra_or_wrong_backbone_weights_rejected(self):
        base, model = temporal_model()
        cp = checkpoint(base, preprocessor())
        key = next(iter(cp['ema']))
        wrong = {**cp, 'ema': {k:v for k,v in cp['ema'].items() if k != key}}
        with self.assertRaisesRegex(ValueError, 'source keys'):
            initialize_model(model, wrong)
        wrong = {**cp, 'ema': {**cp['ema'], 'unrelated.weight': torch.zeros(1)}}
        with self.assertRaisesRegex(ValueError, 'source keys'):
            initialize_model(model, wrong)
        changed = B5UNet(replace(model.config, metadata_dropout=.1))
        with self.assertRaisesRegex(ValueError, 'identical original backbone'):
            initialize_model(changed, cp)
        with self.assertRaises(ValueError):
            initialize_model(base, {**cp, 'architecture_id': 'B5-T-v1'})

    def test_projection_learns_first_then_attention_has_gradients(self):
        base, model = temporal_model()
        initialize_model(model, checkpoint(base, preprocessor()))
        c = inputs(1, 64); x = torch.randn(1, 11, 64)
        model(x, torch.tensor([.4]), **c)[0].square().mean().backward()
        self.assertGreater(float(model.temporal_branch.output_projection.weight.grad.abs().max()), 0.)
        self.assertEqual(float(model.temporal_branch.blocks[0].attention.in_proj_weight.grad.abs().max()), 0.)
        model.zero_grad(set_to_none=True)
        # Static nonzero projection for a derivative check; no optimizer or training.
        with torch.no_grad():
            model.temporal_branch.output_projection.weight.normal_(std=.01)
        model(x, torch.tensor([.4]), **c)[0].square().mean().backward()
        self.assertGreater(float(model.temporal_branch.blocks[0].attention.in_proj_weight.grad.abs().max()), 0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_temporal_tokens_and_arbitrary_length_resize(self):
        branch = TemporalResidual(32, 8, 16, 4, 2, 32, 0.)
        lengths = []
        hook = branch.blocks[0].register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
        for length in [40, 37, 5]:
            x = torch.randn(1, 32, length, requires_grad=True)
            out = branch(x)
            torch.testing.assert_close(out, x, rtol=0, atol=0)
            out.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        hook.remove()
        self.assertEqual(lengths, [8, 8, 5])
        for source, target in [(9, 57), (57, 9), (1, 8)]:
            x = torch.randn(2, 3, source, requires_grad=True)
            y = resize_linear_1d(x, target)
            torch.testing.assert_close(y, F.interpolate(x, size=target, mode='linear', align_corners=False), rtol=1e-5, atol=1e-5)
            y.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_full_window_and_rng_constructor_isolation(self):
        torch.manual_seed(19)
        base = B5UNet(ModelConfig())
        rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        model = B5UNet(replace(ModelConfig(), temporal_enabled=True)).eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        seen = []
        hook = model.temporal_branch.blocks[0].register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape)))
        with torch.no_grad():
            velocity, anchor = model(torch.zeros(1, 11, 5000), torch.tensor([.2]), **inputs(1, 5000))
        hook.remove()
        self.assertEqual(seen, [(1, 250, 128)])
        self.assertEqual(velocity.shape, (1, 11, 5000))
        self.assertEqual(anchor.shape, (1, 1, 5000))

    def test_trained_temporal_checkpoint_strict_roundtrip(self):
        base, model = temporal_model()
        initialize_model(model, checkpoint(base, preprocessor()))
        with torch.no_grad():
            model.temporal_branch.output_projection.weight.fill_(.01)
        cp = checkpoint(model, preprocessor())
        cp['architecture_id'] = model.config.architecture_id
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'temporal.pt'; atomic_save(path, cp)
            loaded = load_checkpoint(path)
            rebuilt = model_from_checkpoint(loaded, torch.device('cpu'))
            torch.testing.assert_close(rebuilt.temporal_branch.output_projection.weight,
                                       model.temporal_branch.output_projection.weight, rtol=0, atol=0)
            self.assertEqual(initialize_model(model, loaded)['mode'], 'strict_same_architecture')
            with self.assertRaises(ValueError):
                initialize_model(base, loaded)

    def test_paired_configs_only_differ_in_temporal_enable_and_output(self):
        control = load_config(ROOT/'configs/experiments/b5_t_control.yaml')
        treatment = load_config(ROOT/'configs/experiments/b5_t_finetune_meta.yaml')
        for section in ['stage', 'training', 'validation', 'sampling', 'loss']:
            self.assertEqual(control[section], treatment[section])
        a, b = dict(control['model']), dict(treatment['model'])
        self.assertFalse(a.pop('temporal_enabled')); self.assertTrue(b.pop('temporal_enabled'))
        self.assertEqual(a, b)
        self.assertEqual(control['sampling']['samples'], 16)
        self.assertTrue(control['training']['validate_initial'])

    def test_cpu_initialization_preflight_without_real_data_or_optimizer(self):
        from baselines.B5.check_time_init import inspect_initialization
        base, model = temporal_model()
        processor = preprocessor()
        config = {'model': model.config.checkpoint_dict}
        with patch('baselines.B5.data.load_preprocessor', return_value=processor), \
             patch('baselines.B5.data.preprocessing_config', return_value=processor.config):
            report = inspect_initialization(config, checkpoint(base, processor))
        self.assertTrue(report['ready'])
        self.assertFalse(report['training_started'])
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(report['velocity_max_abs_difference'], 0)
        self.assertGreater(report['new_branch_parameters'], 0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; run these tests on the server')
class CUDATemporalTests(unittest.TestCase):
    def test_strict_deterministic_attention_and_resize_backward(self):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            branch = TemporalResidual(32, 8, 16, 4, 2, 32, 0.).cuda()
            with torch.no_grad():
                branch.output_projection.weight.fill_(.01)
            x = torch.randn(1, 32, 37, device='cuda', requires_grad=True)
            branch(x).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(torch.isfinite(branch.blocks[0].attention.in_proj_weight.grad).all())
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_full_model_cuda_backward(self):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            _, model = temporal_model(); model = model.cuda()
            c = {key: value.cuda() for key, value in inputs(1, 5000).items()}
            with torch.no_grad():
                model.temporal_branch.output_projection.weight.normal_(std=.01)
            model(torch.randn(1,11,5000,device='cuda'), torch.tensor([.4],device='cuda'), **c)[0].square().mean().backward()
            self.assertTrue(torch.isfinite(model.temporal_branch.output_projection.weight.grad).all())
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == '__main__':
    unittest.main()
