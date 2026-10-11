from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from baselines.B5.checkpoint import atomic_save, initialize_model, load_checkpoint, model_from_checkpoint
from baselines.B5.config import ModelConfig, load_config
from baselines.B5.decoder_fusion import WeightedScaleResidual
from baselines.B5.flow import sample
from baselines.B5.model import B5UNet
from baselines.B5.tests.test_b5 import checkpoint, inputs, preprocessor, tiny_model

ROOT = Path(__file__).resolve().parents[3]


def paired_models():
    base = tiny_model()
    return base, B5UNet(replace(base.config, fusion_enabled=True, fusion_dimension=8)).eval()


class FusionTests(unittest.TestCase):
    def test_legacy_hash_config_and_state_keys_unchanged(self):
        config = ModelConfig()
        self.assertEqual(config.fingerprint, '637df1317e18638abc52330edb754308bebd802803ad5bb349a853d2f52afd46')
        self.assertEqual(config.checkpoint_dict, config.core_dict)
        base, model = paired_models()
        self.assertIsNone(base.fusion_branch)
        self.assertFalse(any(k.startswith('fusion_branch.') for k in base.state_dict()))
        self.assertNotEqual(base.config.fingerprint, model.config.fingerprint)
        self.assertEqual(model.config.architecture_id, 'B5-F-v1')
        for overrides in [{'fusion_enabled': 1}, {'fusion_dimension': 0}, {'fusion_dimension': True}]:
            with self.assertRaises(ValueError):
                replace(config, **overrides)

    def test_zero_initialization_exact_forward_and_target_free_sampling(self):
        base, model = paired_models()
        audit = initialize_model(model, checkpoint(base, preprocessor()))
        self.assertEqual(audit['mode'], 'B5_to_B5F_zero_residual')
        self.assertTrue(all(k.startswith('fusion_branch.') for k in audit['new_keys']))
        for length in [8, 63, 64, 65]:
            data = inputs(1, length)
            state = torch.randn(1, 11, length)
            with torch.no_grad():
                for old, new in zip(base(state, torch.tensor([.3]), **data), model(state, torch.tensor([.3]), **data)):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
        with torch.no_grad():
            data = inputs(1, 64)
            torch.testing.assert_close(sample(base, data, ['record:0'], steps=2, samples=2),
                                       sample(model, data, ['record:0'], steps=2, samples=2), rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_weighted_fusion_uses_all_three_scales(self):
        branch = WeightedScaleResidual((8, 16, 32), 8, 16)
        with torch.no_grad():
            branch.output_projection.weight.normal_(std=.05)
        features = tuple(torch.randn(1, c, n, requires_grad=True) for c, n in [(8, 65), (16, 32), (32, 16)])
        result = branch(features, 32)
        self.assertEqual(result.shape, (1, 16, 32))
        torch.testing.assert_close(branch.scale_logits.softmax(0), torch.full((3,), 1/3))
        result.square().mean().backward()
        for feature in features:
            self.assertGreater(float(feature.grad.abs().sum()), 0.)
        self.assertGreater(float(branch.scale_logits.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(branch.scale_logits.grad).all())

    def test_projection_receives_gradient_then_scale_weights_receive_gradient(self):
        base, model = paired_models()
        initialize_model(model, checkpoint(base, preprocessor()))
        data, state = inputs(1, 64), torch.randn(1, 11, 64)
        model(state, torch.tensor([.4]), **data)[0].square().mean().backward()
        for branch in [model.fusion_branch.at_half, model.fusion_branch.at_full]:
            self.assertGreater(float(branch.output_projection.weight.grad.abs().sum()), 0.)
            self.assertEqual(float(branch.scale_logits.grad.abs().sum()), 0.)
            self.assertEqual(float(branch.projections[0].weight.grad.abs().sum()), 0.)
        model.zero_grad(set_to_none=True)
        # Derivative check with static nonzero parameters; no optimizer updates.
        with torch.no_grad():
            for branch in [model.fusion_branch.at_half, model.fusion_branch.at_full]:
                branch.output_projection.weight.normal_(std=.02)
        model(state, torch.tensor([.4]), **data)[0].square().mean().backward()
        for branch in [model.fusion_branch.at_half, model.fusion_branch.at_full]:
            self.assertGreater(float(branch.scale_logits.grad.abs().sum()), 0.)
            for projection in branch.projections:
                self.assertGreater(float(projection.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_transfer_refuses_missing_extra_wrong_shape_and_changed_conditions(self):
        base, model = paired_models()
        cp = checkpoint(base, preprocessor())
        key = next(iter(cp['ema']))
        for state in [{k:v for k,v in cp['ema'].items() if k != key},
                      {**cp['ema'], 'extra.weight': torch.zeros(1)}]:
            with self.assertRaisesRegex(ValueError, 'source keys'):
                initialize_model(model, {**cp, 'ema': state})
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            initialize_model(model, {**cp, 'ema': {**cp['ema'], key: torch.zeros(1)}})
        with self.assertRaisesRegex(ValueError, 'identical original backbone'):
            initialize_model(B5UNet(replace(model.config, metadata_dropout=.1)), cp)

    def test_learned_fusion_checkpoint_strict_roundtrip_and_reject_reverse_transfer(self):
        base, model = paired_models()
        initialize_model(model, checkpoint(base, preprocessor()))
        with torch.no_grad():
            model.fusion_branch.at_half.output_projection.weight.fill_(.02)
            model.fusion_branch.at_full.scale_logits.copy_(torch.tensor([.2, -.3, .7]))
        cp = checkpoint(model, preprocessor())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fusion.pt'
            atomic_save(path, cp)
            restored = model_from_checkpoint(load_checkpoint(path), torch.device('cpu'))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            data = inputs(1, 64)
            torch.testing.assert_close(sample(model, data, ['a:0'], steps=2, samples=2),
                                       sample(restored, data, ['a:0'], steps=2, samples=2), rtol=0, atol=0)
            self.assertEqual(initialize_model(restored, cp)['mode'], 'strict_same_architecture')
            with self.assertRaises(ValueError):
                initialize_model(base, cp)
            atomic_save(path, {**cp, 'architecture_id': 'B5-U-v1'})
            with self.assertRaisesRegex(ValueError, 'architecture ID'):
                load_checkpoint(path)

    def test_full_window_shapes_and_constructor_rng_isolation(self):
        torch.manual_seed(19)
        base = B5UNet(ModelConfig()).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        model = B5UNet(replace(ModelConfig(), fusion_enabled=True)).eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        data = inputs(1, 5000)
        state = torch.randn(1, 11, 5000)
        with torch.no_grad():
            old = base(state, torch.tensor([.2]), **data)
            new = model(state, torch.tensor([.2]), **data)
        self.assertEqual(new[0].shape, (1, 11, 5000))
        for a,b in zip(old,new):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_paired_configs_only_change_fusion_and_output(self):
        control = load_config(ROOT/'configs/experiments/b5_f_control.yaml')
        treatment = load_config(ROOT/'configs/experiments/b5_f_finetune_meta.yaml')
        for section in ['stage', 'training', 'validation', 'sampling', 'loss']:
            self.assertEqual(control[section], treatment[section])
        a,b = dict(control['model']),dict(treatment['model'])
        self.assertFalse(a.pop('fusion_enabled'))
        self.assertTrue(b.pop('fusion_enabled'))
        self.assertEqual(a,b)
        paths_a, paths_b = dict(control['paths']), dict(treatment['paths'])
        paths_a.pop('output_dir'); paths_b.pop('output_dir')
        self.assertEqual(paths_a, paths_b)
        self.assertEqual(control['sampling'], {'steps':16,'samples':16,'solver':'heun','seed':42})
        self.assertEqual(control['training']['epochs'],20)
        self.assertTrue(control['training']['validate_initial'])

    def test_cpu_preflight_has_zero_updates_and_finite_identity(self):
        from baselines.B5.check_fusion_init import inspect_initialization
        base, model = paired_models()
        processor = preprocessor()
        with patch('baselines.B5.data.load_preprocessor', return_value=processor), \
             patch('baselines.B5.data.preprocessing_config', return_value=processor.config):
            report = inspect_initialization({'model': model.config.checkpoint_dict}, checkpoint(base, processor))
        self.assertTrue(report['ready'])
        self.assertFalse(report['training_started'])
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual(report['velocity_max_abs_difference'], 0.)
        self.assertGreater(report['new_branch_parameters'], 0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; run on the training server')
class CUDAFusionTests(unittest.TestCase):
    def test_full_window_strict_deterministic_backward(self):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            _, model = paired_models()
            model = model.cuda()
            with torch.no_grad():
                for branch in [model.fusion_branch.at_half, model.fusion_branch.at_full]:
                    branch.output_projection.weight.normal_(std=.02)
            data = {k:v.cuda() for k,v in inputs(1,5000).items()}
            model(torch.randn(1,11,5000,device='cuda'), torch.tensor([.4],device='cuda'), **data)[0].square().mean().backward()
            for branch in [model.fusion_branch.at_half, model.fusion_branch.at_full]:
                self.assertGreater(float(branch.scale_logits.grad.abs().sum()),0.)
                self.assertTrue(torch.isfinite(branch.output_projection.weight.grad).all())
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == '__main__':
    unittest.main()
