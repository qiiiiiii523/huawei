from argparse import Namespace
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from baselines.B7 import adapt120, evaluate_modes, train as trainer
from baselines.B7.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.data import DATA_PROTOCOL, WindowDataset, read_manifest
from baselines.B7.evaluate import SCORE_PROTOCOL, aggregate
from baselines.B7.inference import predict_record
from baselines.B7.losses import core_slice, reconstruction_losses
from baselines.B7.model import B7Transformer
from baselines.B7.reschedule import assert_same
from baselines.B7.tests.test_b7 import payload, small_config


def manifest_fixture(root):
    t = np.arange(60000, dtype='float32')/500
    target = np.stack([100*np.sin(t*(2+j/10))+j for j in range(12)])
    rows = []
    for split in ('train', 'validation'):
        filename = split+'.npz'
        np.savez(root/filename, anchor_uV=target[:1], target_uV=target,
                 lead_mask=np.ones(12, dtype=bool))
        for group in ('task1', 'machine', 'body'):
            name = split+':'+group
            rows.append(dict(record_id=name, target_record_id=name, subject_id=name,
                source='huawei', group=group, path=filename, split=split, input_reliable=True))
    path = root/'manifest.json'
    path.write_text(json.dumps({'protocol': DATA_PROTOCOL, 'sample_rate': 500,
                               'scales_uV': [100.]*12, 'records': rows}), encoding='utf-8')
    return path


def metrics_fixture(value):
    rows = [{'record_id': group, 'target_record_id': group, 'group': group,
             'leads': {str(i): {'r': value, 'rmse_uV': 1.} for i in range(1, 12)}}
            for group in ('task1', 'machine', 'body')]
    return aggregate(rows, 'huawei120'), rows


class RecordingVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(.1))
        self.lengths = []
    def forward(self, state, time, anchor, metadata=None):
        self.lengths.append(state.shape[-1])
        return state*self.marker + anchor*.02


class Full120Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def training_fixture(self, root, updates=1):
        manifest_path = manifest_fixture(root)
        manifest = read_manifest(manifest_path)
        old_cfg = {'stage': 'huawei40', 'model': small_config().to_dict(), 'loss': asdict(LossConfig()),
            'sampling': asdict(SamplingConfig()), 'validation': {'batch_size': 1, 'initial': False},
            'training': {'seed': 7, 'updates': updates, 'validate_every': updates, 'checkpoint_every': 1,
                'batch_size': 1, 'learning_rate': .00001, 'weight_decay': .01, 'ema_decay': .9,
                'gradient_clip': 1., 'boundary_probability': .15}}
        old_cfg = json.loads(json.dumps(old_cfg))
        value = payload(B7Transformer(small_config()))
        value.update(stage='huawei40', score_protocol=SCORE_PROTOCOL, update=updates, best_r=.4,
            identity={'config': old_cfg, 'code_sha256': 'prior40-code',
                      **evaluate_modes.data_identity(manifest_path, manifest)})
        source = root/'source/A'
        source.mkdir(parents=True)
        save_checkpoint(source/'best_r.pt', value)
        (source/'config.json').write_text(json.dumps(old_cfg), encoding='utf-8')
        (source/'completed.json').write_text(json.dumps({'completed': True, 'update': updates, 'best_r': .4}))
        cfg = copy.deepcopy(old_cfg)
        cfg['stage'] = 'huawei120'
        cfg['loss']['pcc_core_seconds'] = 90
        cfg['sampling'] = asdict(SamplingConfig(window_seconds=120, core_seconds=120))
        cfg['training']['boundary_probability'] = 0.
        config = root/'config120.json'
        config.write_text(json.dumps(cfg), encoding='utf-8')
        args = Namespace(config=str(config), manifest=str(manifest_path), output=str(root/'run'),
                         device='cpu', resume=False, init_checkpoint=str(source/'best_r.pt'))
        return cfg, args

    def test_recipes_preserve_model_channels_losses_K16_and_final_only(self):
        for preset in adapt120.PRESET_NAMES:
            args = Namespace(preset=preset, batch_size=4, updates=1000, eval_batch_size=4)
            cfg = adapt120.recipe(args)
            self.assertEqual(cfg['loss']['pcc_core_seconds'], 90)
            self.assertEqual(cfg['loss']['ii_weight'], 5.)
            self.assertEqual(cfg['sampling'], asdict(SamplingConfig(window_seconds=120, core_seconds=120)))
            self.assertFalse(cfg['validation']['initial'])
            self.assertEqual(cfg['training']['validate_every'], 1000)
            self.assertEqual(cfg['training']['checkpoint_every'], 500)
            self.assertEqual(cfg['training']['boundary_probability'], 0.)

    def test_PCC_core90_excludes_edges_and_MAE_uses_whole120(self):
        self.assertEqual(core_slice(60000, core_seconds=90), slice(7500, 52500))
        self.assertEqual(core_slice(20000), slice(7500, 12500))
        y = torch.sin(torch.arange(60000)/21)[None, None].repeat(1, 7, 1)
        prediction = y.clone()
        prediction[:, :, :7500] += 3
        prediction[:, :, 52500:] -= 3
        parts = reconstruction_losses(prediction, y, torch.ones_like(y, dtype=torch.bool), LossConfig(pcc_core_seconds=90))
        self.assertGreater(parts['mae'].item(), .7)
        self.assertLess(abs(parts['pcc'].item()), 1e-5)
        for value in (0, 121, True):
            with self.assertRaises(ValueError):
                core_slice(60000, core_seconds=value)

    def test_full120_dataset_never_crops_or_pads_and_preserves_raw_I(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = read_manifest(manifest_fixture(root))
            dataset = WindowDataset(manifest, 120, 0.)
            with np.load(root/'train.npz') as saved:
                expected = torch.from_numpy(saved['anchor_uV']/100.)
            for _ in range(3):
                batch = dataset[0]
                self.assertEqual(batch['target'].shape, (7, 60000))
                self.assertTrue(torch.equal(batch['anchor'], expected))
                self.assertTrue(batch['time_mask'].all())
            with self.assertRaisesRegex(ValueError, 'disable padded'):
                WindowDataset(manifest, 120, .15)
            with np.load(root/'train.npz') as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            arrays['anchor_uV'] = arrays['anchor_uV'][:, :-1]
            arrays['target_uV'] = arrays['target_uV'][:, :-1]
            np.savez(root/'train.npz', **arrays)
            with self.assertRaisesRegex(ValueError, 'exactly 60000'):
                WindowDataset(manifest, 120, 0.)

    def test_inference_mode_window_counts_and_K16_averaging_preserve_I(self):
        q = torch.sin(torch.arange(60000)/23)[None]
        for window, core, expected_calls in ((120, 120, 32), (40, 10, 384), (40, 20, 192)):
            model = RecordingVelocity().eval()
            sc = SamplingConfig(window_seconds=window, core_seconds=core, steps=1)
            output = predict_record(model, q, [100.]*12, 'record', config=sc, batch_size=1)
            self.assertEqual(len(model.lengths), expected_calls)
            self.assertEqual(set(model.lengths), {window*500})
            self.assertEqual(output.shape, (12, 60000))
            self.assertTrue(torch.equal(output[:1], q))
            self.assertTrue(torch.isfinite(output).all())

    def test_new_huawei_stage_preserves_original_group_aggregation(self):
        _, rows = metrics_fixture(.5)
        for row in rows:
            for values in row['leads'].values():
                values['r'] = {'task1': .8, 'machine': .2, 'body': .4}[row['group']]
        metrics = aggregate(rows, 'huawei120')
        self.assertEqual(metrics, aggregate(rows, 'huawei40'))
        self.assertAlmostEqual(metrics['r'], .55)

    def test_real120_backward_initializes_same_data_40EMA_and_validates_only_final(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, args = self.training_fixture(root)
            source_before = Path(args.init_checkpoint).read_bytes()
            with patch.object(trainer, 'evaluate', return_value=metrics_fixture(.5)) as evaluation:
                trainer.train(args)
                self.assertEqual(evaluation.call_count, 1)
                self.assertEqual(evaluation.call_args.args[2].window_seconds, 120)
                self.assertEqual(evaluation.call_args.args[3], 'huawei120')
            checkpoint = load_checkpoint(root/'run/last.pt')
            self.assertEqual(checkpoint['update'], 1)
            self.assertEqual(checkpoint['last_validated_update'], 1)
            self.assertEqual(checkpoint['loss']['pcc_core_seconds'], 90)
            self.assertFalse(checkpoint['pending_validation'])
            self.assertEqual(checkpoint['initialization']['sha256'], file_sha256(args.init_checkpoint))
            self.assertEqual(Path(args.init_checkpoint).read_bytes(), source_before)
            self.assertTrue(all(torch.isfinite(v).all() for v in checkpoint['ema'].values()))

    def test_comparison_reuses_full120_and_evaluates_other_modes_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, args = self.training_fixture(root)
            with patch.object(trainer, 'evaluate', return_value=metrics_fixture(.5)):
                trainer.train(args)
            compare_args = Namespace(checkpoint=root/'run/best_r.pt', manifest=Path(args.manifest),
                                     output=root/'comparison', batch_size=1, device='cpu')
            def evaluate(model, manifest, sampling, stage, **kwargs):
                return metrics_fixture(.4 if sampling.core_seconds == 10 else .6)
            with patch.object(evaluate_modes, 'evaluate', side_effect=evaluate) as evaluator:
                result = evaluate_modes.compare(compare_args)
                self.assertEqual(evaluator.call_count, 2)
                for mode, expected in {'full120': .5, 'window40_core10': .4, 'window40_core20': .6}.items():
                    self.assertAlmostEqual(result[mode], expected)
            with patch.object(evaluate_modes, 'evaluate') as evaluator:
                self.assertEqual(evaluate_modes.compare(compare_args), result)
                evaluator.assert_not_called()

    def test_source_40_code_hash_may_differ_but_model_and_data_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, train_args = self.training_fixture(root)
            args = Namespace(source_run=root/'source', manifest=Path(train_args.manifest))
            info = adapt120.verify_source(args, 'A', cfg)
            self.assertEqual(info['source_code_sha256'], 'prior40-code')
            self.assertEqual(info['best_r_40s'], .4)
            changed = copy.deepcopy(cfg)
            changed['model']['embed_dims'] = (32, 64, 96)
            with self.assertRaisesRegex(ValueError, 'identity/model'):
                adapt120.verify_source(args, 'A', changed)

    def test_full120_resume_preserves_model_EMA_optimizer_and_random_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, args = self.training_fixture(root, updates=2)
            original = trainer.balanced_indices
            count = 0
            def stop(*a, **kw):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError('pause after saved first update')
                return original(*a, **kw)
            with patch.object(trainer, 'balanced_indices', side_effect=stop):
                with self.assertRaisesRegex(RuntimeError, 'pause after saved'):
                    trainer.train(args)
            self.assertEqual(load_checkpoint(root/'run/last.pt')['update'], 1)
            source = args.init_checkpoint
            args.resume, args.init_checkpoint = True, None
            with patch.object(trainer, 'evaluate', return_value=metrics_fixture(.5)):
                trainer.train(args)
            actual = load_checkpoint(root/'run/last.pt')
            args.output, args.resume, args.init_checkpoint = str(root/'reference'), False, source
            with patch.object(trainer, 'evaluate', return_value=metrics_fixture(.5)):
                trainer.train(args)
            expected = load_checkpoint(root/'reference/last.pt')
            for key in ('model', 'ema', 'optimizer', 'rng'):
                assert_same(actual[key], expected[key])

    def test_adapter_end_to_end_reuses_source_and_resumes_completed_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, training_args = self.training_fixture(root)
            args = Namespace(output=root/'adapt', source_run=root/'source', manifest=Path(training_args.manifest),
                preset='original_t_15m', routes=['A'], batch_size=1, eval_batch_size=1, updates=1,
                gpu='0', wait_for_source=False, dry_run=False)
            source_before = (root/'source/A/best_r.pt').read_bytes()
            def execute(command, log_path, gpu):
                def option(key):
                    return command[command.index(key)+1]
                self.assertEqual(gpu, '0')
                if 'baselines.B7.train' in command:
                    train_args = Namespace(config=option('--config'), manifest=option('--manifest'),
                        output=option('--output'), init_checkpoint=option('--init-checkpoint'), resume=False, device='cpu')
                    with patch.object(trainer, 'evaluate', return_value=metrics_fixture(.5)):
                        trainer.train(train_args)
                else:
                    compare_args = Namespace(checkpoint=Path(option('--checkpoint')), manifest=Path(option('--manifest')),
                        output=Path(option('--output')), device='cpu', batch_size=1)
                    with patch.object(evaluate_modes, 'evaluate', return_value=metrics_fixture(.6)):
                        evaluate_modes.compare(compare_args)
            with patch.object(adapt120, 'recipe', return_value=cfg), \
                 patch.object(adapt120, 'execute', side_effect=execute) as executor:
                self.assertEqual(adapt120.run(args), 0)
                self.assertEqual(executor.call_count, 2)
            status = json.loads((args.output/'pipeline_status.json').read_text())
            self.assertTrue(status['success'])
            self.assertEqual(status['results']['A120']['source_40s_r'], .4)
            with patch.object(adapt120, 'recipe', return_value=cfg), patch.object(adapt120, 'execute') as executor:
                self.assertEqual(adapt120.run(args), 0)
                executor.assert_not_called()
            self.assertEqual((root/'source/A/best_r.pt').read_bytes(), source_before)


if __name__ == '__main__':
    unittest.main()
