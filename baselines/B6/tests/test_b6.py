from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from baselines.B5 import ARCHITECTURE_ID as B5_ID
from baselines.B5.checkpoint import load_checkpoint as load_b5_checkpoint
from baselines.B5.config import load_config as load_b5_config
from baselines.B5.predict import canonical_anchor, restore_rate
from baselines.B5.runtime import collate
from baselines.B6 import ARCHITECTURE_ID, CONDITION_SCHEMA, PREPROCESSING_VERSION
from baselines.B6.checkpoint import load_checkpoint, model_from_checkpoint, scales_digest
from baselines.B6.config import ModelConfig, load_config
from baselines.B6.flow import keyed_noise, sample
from baselines.B6.losses import flow_loss
from baselines.B6.model import B6AxialFlow
from baselines.B6.predict import predict_record

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[3]


def tiny_config(**overrides):
    values = dict(cnn_channels=(8, 16, 32), decoder_channels=(16, 8), state_channels=(8, 16),
                  state_fine_channels=8, d_model=16, num_blocks=1, num_heads=2, ffn_dim=32,
                  time_dim=16, metadata_dim=16, dropout=0., metadata_dropout=0., metadata_field_dropout=0.)
    values.update(overrides)
    return ModelConfig(**values)


def inputs(batch=2, length=100):
    return {'anchor':torch.randn(batch, 1, length), 'numeric':torch.rand(batch, 3),
            'sex':torch.arange(batch) % 2, 'field_mask':torch.ones(batch, 4),
            'age_topcoded':torch.zeros(batch, 1)}


def call(model, state, time, data):
    return model(state, time, **data)


class B6Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_default_5000_contract_and_parameter_count(self):
        model = B6AxialFlow().eval()
        with torch.no_grad():
            velocity, anchor = call(model, torch.randn(1, 11, 5000), torch.tensor([.4]), inputs(1, 5000))
        self.assertEqual(tuple(velocity.shape), (1, 11, 5000))
        self.assertEqual(tuple(anchor.shape), (1, 1, 5000))
        self.assertTrue(torch.isfinite(velocity).all())
        self.assertEqual(model.parameter_count, 3425799)

    def test_noise_and_time_reach_velocity(self):
        model = B6AxialFlow(tiny_config()).eval()
        data = inputs(1)
        state = torch.randn(1, 11, 100)
        with torch.no_grad():
            a, _ = call(model, state, torch.tensor([.1]), data)
            b, _ = call(model, state + .1, torch.tensor([.1]), data)
            c, _ = call(model, state, torch.tensor([.7]), data)
        self.assertGreater(float((a - b).abs().max()), 1e-5)
        self.assertGreater(float((a - c).abs().max()), 1e-5)

    def test_both_attention_axes_and_ablation(self):
        for axes in ('both', 'time_only', 'lead_only'):
            model = B6AxialFlow(tiny_config(attention_axes=axes)).eval()
            v, _ = call(model, torch.randn(1, 11, 100), torch.tensor([.4]), inputs(1))
            self.assertEqual(tuple(v.shape), (1, 11, 100))
            block = model.axial_blocks[0]
            self.assertEqual(block.time_attention is not None, axes != 'lead_only')
            self.assertEqual(block.lead_attention is not None, axes != 'time_only')

    def test_gradient_reaches_state_attention_and_condition(self):
        model = B6AxialFlow(tiny_config()).train()
        data = inputs(2)
        data['anchor'].requires_grad_(True)
        state = torch.randn(2, 11, 100, requires_grad=True)
        v, anchor = call(model, state, torch.tensor([.3, .7]), data)
        (v.square().mean() + anchor.square().mean()).backward()
        for grad in (state.grad, data['anchor'].grad, model.state_projection.weight.grad,
                     model.axial_blocks[0].time_attention.in_proj_weight.grad,
                     model.axial_blocks[0].lead_attention.in_proj_weight.grad,
                     model.conditioners[0].film.weight.grad,
                     model.raw_state_skip.weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)

    def test_missing_metadata_nan_is_sanitized(self):
        model = B6AxialFlow(tiny_config()).eval()
        data = inputs(1)
        data['numeric'].fill_(float('nan'))
        data['field_mask'].zero_()
        v, _ = call(model, torch.randn(1, 11, 100), torch.tensor([.5]), data)
        self.assertTrue(torch.isfinite(v).all())

    def test_i_only_ignores_demographics(self):
        model = B6AxialFlow(tiny_config(metadata_enabled=False)).eval()
        data = inputs(1)
        state, time = torch.randn(1, 11, 100), torch.tensor([.5])
        a, _ = call(model, state, time, data)
        data['numeric'].add_(100)
        data['sex'] = 1 - data['sex']
        b, _ = call(model, state, time, data)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_fm_loss_backward_and_invalid_lead_mask(self):
        model = B6AxialFlow(tiny_config()).train()
        data = inputs(2)
        data['target'] = torch.randn(2, 12, 100)
        data['quality_mask'] = torch.ones(2, 12, dtype=torch.bool)
        data['quality_mask'][0, 7] = False
        data['target'][0, 7].fill_(float('nan'))
        settings = dict(huber=.1, pcc=.1, anchor=.02, physiology=0., huber_delta=1., slow_trend=0.)
        loss, parts = flow_loss(model, data, settings, torch.ones(12))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(parts['slow_trend']), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.local_head.weight.grad).all())

    def test_sampling_target_free_cache_and_batch_stability(self):
        model = B6AxialFlow(tiny_config()).eval()
        data = inputs(2)
        keys = ['record:a:0', 'record:b:0']
        calls = []
        hook = model.anchor_encoder.register_forward_hook(lambda *args:calls.append(1))
        both = sample(model, data, keys, steps=2, samples=2)
        hook.remove()
        self.assertEqual(len(calls), 1)
        singles = torch.cat([sample(model, {k:v[i:i+1] for k,v in data.items()}, keys[i:i+1],
                                    steps=2, samples=2) for i in range(2)])
        torch.testing.assert_close(both, singles, rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.isfinite(both).all())
        self.assertEqual(tuple(both.shape), (2, 12, 100))

    def test_sampling_does_not_copy_visible_i(self):
        model = B6AxialFlow(tiny_config()).eval()
        data = inputs(1)
        p = sample(model, data, ['record:0'], steps=1)
        self.assertGreater(float((p[:, :1] - data['anchor']).abs().max()), 1e-5)
        model.train()
        with self.assertRaises(ValueError):
            sample(model, data, ['record:0'])

    def test_shared_stable_noise(self):
        a = keyed_noise(['x', 'y'], 100, 42, 0, torch.device('cpu'), torch.float32)
        b = keyed_noise(['y'], 100, 42, 0, torch.device('cpu'), torch.float32)
        torch.testing.assert_close(a[1:], b, rtol=0, atol=0)

    def test_checkpoint_identity_roundtrip_and_cross_baseline_rejection(self):
        config = tiny_config()
        model = B6AxialFlow(config).eval()
        scales = {'d12':[1.] * 12, 'ecg_machine_i':[1.]}
        checkpoint = {'architecture_id':ARCHITECTURE_ID, 'architecture_hash':config.fingerprint,
                      'model_config':asdict(config), 'condition_schema':CONDITION_SCHEMA,
                      'preprocessing_version':PREPROCESSING_VERSION, 'scales':scales,
                      'scales_sha256':scales_digest(scales), 'model':model.state_dict(), 'ema':model.state_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'b6.pt'
            torch.save(checkpoint, path)
            restored = model_from_checkpoint(load_checkpoint(path), torch.device('cpu'))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'B5-U'):
                load_b5_checkpoint(path)
            checkpoint['architecture_id'] = B5_ID
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'B6-M1-CFM'):
                load_checkpoint(path)

    def test_config_variants_and_shared_comparison_contract(self):
        b5 = load_b5_config(ROOT / 'configs/experiments/b5_local_meta.yaml')
        b6 = load_config(ROOT / 'configs/experiments/b6_local_meta.yaml')
        for section in ('sampling', 'validation'):
            a,b = dict(b5[section]), dict(b6[section])
            a.pop('batch_size',None); b.pop('batch_size',None)
            self.assertEqual(a,b)
        self.assertEqual(b6['paths']['scales'], b5['paths']['scales'])
        self.assertEqual(b6['training']['batch_size'] * b6['training']['gradient_accumulation'], 8)
        for name in ('local_meta','public_meta','finetune_meta','local_i','public_i','finetune_i','local_time_only','local_lead_only'):
            c = load_config(ROOT / f'configs/experiments/b6_{name}.yaml')
            ModelConfig.from_dict(c['model'])

    def test_invalid_architecture_settings(self):
        for args in ({'d_model':17}, {'attention_axes':'bad'}, {'time_tokens':100}, {'dropout':1.}, {'num_blocks':0}):
            with self.assertRaises(ValueError):
                tiny_config(**args)

    def test_cli_opt_in_before_loading_or_training(self):
        for command in ('train_public','train_huawei','overfit'):
            p = subprocess.run([sys.executable,'-m',f'baselines.B6.{command}','--config','missing.yaml'],
                               cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Training was not started',p.stderr)
            self.assertNotIn('Traceback',p.stderr)

    def test_target_free_tail_inference(self):
        class Processor:
            scale_uV_by_source = {'d12':np.ones(12,dtype=np.float32)}
            def transform_observed_record(self,raw,source):
                class Result: pass
                value=Result(); value.model_signal=raw
                return value
        model = B6AxialFlow(tiny_config()).eval()
        metadata = {'numeric':np.zeros(3,dtype=np.float32),'sex':np.asarray(2,dtype=np.int64),
                    'field_mask':np.zeros(4,dtype=np.float32),'age_topcoded':np.zeros(1,dtype=np.float32)}
        raw = np.sin(np.arange(5017,dtype=np.float32)/10)[None]
        p = predict_record(model, Processor(), raw, metadata, 'visible-only', torch.device('cpu'),
                           {'seed':42,'steps':1,'samples':1,'solver':'euler'},1)
        self.assertEqual(p.shape,(12,5017))
        self.assertTrue(np.isfinite(p).all())

    def test_record_evaluation_reports_b6_and_chest_rmse(self):
        from baselines.B6.runtime import validate
        config = load_config(ROOT / 'configs/experiments/b6_local_meta.yaml')
        config['sampling'].update(steps=1, solver='euler')
        config['validation']['batch_size'] = 1
        model = B6AxialFlow(tiny_config()).eval()
        values = inputs(2, 5000)
        dataset = []
        for i in range(2):
            item = {k:v[i].numpy() for k,v in values.items()}
            item.update(key=f'target:{i}', target_uV=np.random.default_rng(i).normal(size=(12,5000)).astype(np.float32),
                        evaluation_metadata={'pair_id':'same-record','target_record_id':'target',
                                             'subject_id':'subject','start_sample_500hz':str(i*5000),
                                             'expected_window_count':'2'})
            dataset.append(item)
        class Processor:
            scale_uV_by_source = {'d12':np.ones(12,dtype=np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            summaries = validate(model, {'task1':dataset,'task2':dataset}, config, Processor(),
                                 torch.device('cpu'), Path(tmp))
            self.assertEqual(summaries['task2']['architecture_id'], ARCHITECTURE_ID)
            self.assertEqual(summaries['task1']['n_records'],1)
            self.assertEqual(summaries['task2']['task2_rmse_scored_leads'],'V1,V2,V3,V4,V5,V6')
            self.assertTrue((Path(tmp)/'competition_score.csv').is_file())
            self.assertTrue(np.isfinite(summaries['task2']['task2_missing_lead_mean_rmse_uV']))

    def test_predict_cli_with_untrained_fixture_checkpoint(self):
        config = load_config(ROOT / 'configs/experiments/b6_local_meta.yaml')
        architecture = tiny_config()
        config['model'] = asdict(architecture)
        config['sampling'].update(steps=1, solver='euler')
        model = B6AxialFlow(architecture).eval()
        scales = {'d12':[1.] * 12, 'ecg_machine_i':[1.]}
        checkpoint = {'architecture_id':ARCHITECTURE_ID, 'architecture_hash':architecture.fingerprint,
                      'model_config':asdict(architecture), 'condition_schema':CONDITION_SCHEMA,
                      'preprocessing_version':PREPROCESSING_VERSION, 'scales':scales,
                      'scales_sha256':scales_digest(scales), 'model':model.state_dict(),
                      'ema':model.state_dict(), 'config':config}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            torch.save(checkpoint,folder/'untrained.pt')
            np.save(folder/'visible.npy',np.sin(np.arange(5017,dtype=np.float32)/20))
            command=[sys.executable,'-m','baselines.B6.predict','--checkpoint',str(folder/'untrained.pt'),
                     '--anchor',str(folder/'visible.npy'),'--record-id','fixture-visible-I',
                     '--output',str(folder/'prediction.npy'),'--device','cpu']
            p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(np.load(folder/'prediction.npy').shape,(12,5017))
            sidecar=json.loads((folder/'prediction.json').read_text())
            self.assertEqual(sidecar['unit'],'uV')
            self.assertFalse(sidecar['copied_observed_i'])


if __name__ == '__main__':
    unittest.main()
