from __future__ import annotations

from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from baselines.B5.config import load_config, ModelConfig
from baselines.B5.sampling_settings import DEFAULT_SAMPLING_CONFIG, inference_sampling

ROOT = Path(__file__).resolve().parents[3]
LEGACY = {'seed': 42, 'solver': 'heun', 'steps': 16, 'samples': 1}


class InferenceSamplingTests(unittest.TestCase):
    def test_default_final_profile_and_training_config_separate(self):
        self.assertEqual(inference_sampling(), {**LEGACY, 'samples': 16})
        self.assertEqual(load_config(ROOT/'configs/experiments/b5_finetune_meta.yaml')['sampling'], LEGACY)

    def test_explicit_overrides_and_none_preserves_profile(self):
        self.assertEqual(inference_sampling(overrides={'steps': 32, 'samples': 4, 'seed': None}),
                         {**LEGACY, 'steps': 32, 'samples': 4})
        self.assertEqual(inference_sampling(overrides={'samples': 1})['samples'], 1)

    def test_alternate_profile_and_invalid_settings(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sampling.yaml'
            path.write_text(yaml.safe_dump({'sampling': {**LEGACY, 'samples': 8}}))
            self.assertEqual(inference_sampling(path)['samples'], 8)
            for field, value in [('samples', 0), ('steps', -1), ('samples', True),
                                 ('steps', 16.5), ('seed', -1), ('solver', 'rk4')]:
                path.write_text(yaml.safe_dump({'sampling': {**LEGACY, field: value}}))
                with self.assertRaises(ValueError):
                    inference_sampling(path)
            path.write_text(yaml.safe_dump({'sampling': {'samples': 16}}))
            with self.assertRaises(ValueError):
                inference_sampling(path)
            path.write_text('sampling: []\n')
            with self.assertRaises(ValueError):
                inference_sampling(path)
            with self.assertRaises(FileNotFoundError):
                inference_sampling(Path(tmp)/'missing.yaml')

    def test_legacy_checkpoint_predict_defaults_and_sidecar(self):
        from baselines.B5 import predict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root/'visible_I.npy'; np.save(anchor, np.linspace(-30, 40, 100, dtype=np.float32))
            output = root/'output.npy'
            legacy = {'config': {'sampling': dict(LEGACY)}, 'condition_schema': 'fixture'}
            argv = ['predict', '--checkpoint', str(root/'old.pt'), '--anchor', str(anchor),
                    '--record-id', 'visible-record', '--output', str(output), '--device', 'cpu']
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(patch('baselines.B5.checkpoint.load_checkpoint', return_value=legacy))
                stack.enter_context(patch('baselines.B5.checkpoint.model_from_checkpoint', return_value=object()))
                stack.enter_context(patch('baselines.B5.checkpoint.checkpoint_preprocessor', return_value=object()))
                stack.enter_context(patch('baselines.B5.runtime.seed_all'))
                mocked = stack.enter_context(patch.object(predict, 'predict_record', return_value=np.zeros((12,100), np.float32)))
                stack.enter_context(patch('sys.stdout', new=io.StringIO()))
                predict.main()
            self.assertEqual(mocked.call_args.args[6], {**LEGACY, 'samples': 16})
            self.assertEqual(legacy['config']['sampling'], LEGACY)
            sidecar = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(sidecar['sampling']['samples'], 16)
            self.assertEqual(Path(sidecar['sampling_config']), DEFAULT_SAMPLING_CONFIG)
            self.assertFalse(sidecar['copied_observed_i'])
            self.assertEqual(np.load(output).shape, (12, 100))

    def test_validate_uses_final_profile_and_supports_override(self):
        from baselines.B5 import validate
        for explicit, expected in [([], 16), (['--samples', '1'], 1)]:
            with self.subTest(samples=expected), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = load_config(ROOT/'configs/experiments/b5_finetune_meta.yaml')
                legacy = {'architecture_hash': ModelConfig.from_dict(config['model']).fingerprint,
                          'config': {'sampling': dict(LEGACY)}}
                captured = {}
                def evaluate(model, datasets, cfg, preprocessor, device, output):
                    captured.update(copy.deepcopy(cfg['sampling']))
                    return {'task1': {'r_missing11': .5},
                            'task2': {'r_missing11': .5, 'task2_missing_lead_mean_rmse_uV': 700.}}
                argv = ['validate', '--config', str(ROOT/'configs/experiments/b5_finetune_meta.yaml'),
                        '--checkpoint', str(root/'old.pt'), '--device', 'cpu', '--output-dir', str(root/'out'), *explicit]
                with ExitStack() as stack:
                    stack.enter_context(patch.object(sys, 'argv', argv))
                    stack.enter_context(patch('baselines.B5.checkpoint.load_checkpoint', return_value=legacy))
                    stack.enter_context(patch('baselines.B5.checkpoint.model_from_checkpoint', return_value=object()))
                    stack.enter_context(patch('baselines.B5.checkpoint.checkpoint_preprocessor', return_value=object()))
                    stack.enter_context(patch('baselines.B5.runtime.build_validation_datasets', return_value={}))
                    stack.enter_context(patch('baselines.B5.runtime.seed_all'))
                    stack.enter_context(patch('baselines.B5.runtime.validate', side_effect=evaluate))
                    stack.enter_context(patch('sys.stdout', new=io.StringIO()))
                    validate.main()
                self.assertEqual(captured, {**LEGACY, 'samples': expected})
                saved = json.loads((root/'out'/'inference_sampling.json').read_text())
                self.assertEqual(saved['sampling'], captured)
                self.assertEqual(legacy['config']['sampling'], LEGACY)

    def test_k16_mean_and_heun_network_call_count(self):
        import torch
        from baselines.B5.flow import keyed_noise, sample
        class ZeroVelocity:
            training = False
            calls = 0
            def encode_conditions(self, anchor, *unused):
                return SimpleNamespace(anchor_prediction=anchor)
            def velocity(self, state, time, condition):
                self.calls += 1
                return torch.zeros_like(state)
        model = ZeroVelocity()
        anchor = torch.zeros(1, 1, 8)
        inputs = {'anchor': anchor, 'numeric': None, 'sex': None, 'field_mask': None, 'age_topcoded': None}
        settings = inference_sampling()
        result = sample(model, inputs, ['record:0'], **settings)
        expected = torch.stack([keyed_noise(['record:0'], 8, 42, i, torch.device('cpu')) for i in range(16)]).mean(0)
        torch.testing.assert_close(result[:, 1:], expected)
        self.assertEqual(model.calls, 512)
        torch.testing.assert_close(result[:, :1], anchor)


if __name__ == '__main__':
    unittest.main()
