from argparse import Namespace
from dataclasses import asdict, replace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B7 import direct_suite as suite, train as trainer
from baselines.B7.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.data import DATA_PROTOCOL, WindowDataset, read_manifest
from baselines.B7.evaluate import SCORE_PROTOCOL, evaluate
from baselines.B7.losses import reconstruction_losses, flow_loss
from baselines.B7.model import B7Transformer
from baselines.B7.tests.test_b7 import payload, small_config, RecordingVelocity
from baselines.B7.tests.test_120 import manifest_fixture, metrics_fixture


def public_manifest(root):
    rows = []
    for split in ('train', 'validation'):
        for length in (35000, 37500, 51000):
            t = np.arange(length, dtype='float32') / 500
            target = np.stack([np.sin(t * (2+j/10))*100+j for j in range(12)])
            filename = f'{split}{length}.npz'
            np.savez(root/filename, anchor_uV=target[:1], target_uV=target,
                     lead_mask=np.ones(12, dtype=bool))
            name = f'ptbdb:{split}{length}'
            rows.append(dict(record_id=name, subject_id=name, target_record_id=name,
                source='ptbdb', group='ptbdb', path=filename, split=split, input_reliable=True,
                real_samples=length))
    path = root/'ptb.json'
    path.write_text(json.dumps(dict(protocol=DATA_PROTOCOL, sample_rate=500,
                                   scales_uV=[100.]*12, records=rows)))
    return path


class DirectSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_recipes_are_separate_final_only_and_correct_supervision(self):
        for preset in suite.PRESETS:
            cfgs = suite.recipes(preset)
            self.assertEqual(set(cfgs), {'Q75', 'H120', 'H10', 'H40C10'})
            for cfg in cfgs.values():
                trainer.validate_settings(cfg)
                self.assertEqual(cfg['training']['batch_size'], 16)
                self.assertEqual(cfg['training']['checkpoint_every'], 500)
                self.assertEqual(cfg['training']['validate_every'], 1000)
                self.assertEqual(cfg['validation'], {'batch_size': 4, 'initial': False})
                self.assertEqual(cfg['sampling']['samples'], 16)
                self.assertEqual(cfg['sampling']['steps'], 16)
            self.assertEqual(cfgs['H40C10']['loss']['mae_core_seconds'], 10)
            self.assertEqual(cfgs['H120']['loss']['pcc_core_seconds'], 90)
            self.assertEqual(cfgs['Q75']['sampling']['core_seconds'], 75)

    def test_central_mae_pcc_but_full_window_fm(self):
        t = torch.linspace(0, 40, 20000)
        target = torch.sin(t)[None, None].expand(1, 7, -1).clone()
        prediction = target.clone().requires_grad_()
        mask = torch.ones_like(target, dtype=torch.bool)
        prediction.data[:, :, :7500] += 10
        prediction.data[:, :, 12500:] += 10
        lc = LossConfig(mae_core_seconds=10)
        losses = reconstruction_losses(prediction, target, mask, lc)
        self.assertEqual(float(losses['mae']), 0)
        self.assertLess(float(losses['pcc']), 1e-6)
        losses['mae'].backward()
        self.assertEqual(float(prediction.grad.abs().sum()), 0)
        batch = dict(anchor=target[:, :1], target=target, lead_mask=torch.ones(1, 7, dtype=torch.bool))
        # No endpoint loss weights: FM must still supervise edges.
        model = RecordingVelocity(.1)
        loss, parts = flow_loss(model, batch, replace(lc, mae_weight=0, pcc_weight=0),
                               noise=torch.zeros_like(target), time=torch.tensor([.5]))
        expected = ((.1-target)**2 * torch.tensor([5, 1, 1, 1, 1, 1, 1])[None,:,None]).sum() / (11*20000)
        torch.testing.assert_close(parts['fm'], expected)
        loss.backward()
        self.assertTrue(torch.isfinite(model.marker.grad))

    def test_q75_real_windows_and_full75_evaluation_no_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = public_manifest(root)
            value = suite.ptb75_manifest(original)
            self.assertEqual(value['window_selection']['counts'], {'train': 2, 'validation': 2})
            self.assertEqual(len(value['window_selection']['excluded_short']), 2)
            path = root/'ptb75.json'
            suite.write(path, value)
            manifest = read_manifest(path)
            ds = WindowDataset(manifest, 75, 0.)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0]['anchor'].shape[-1], 37500)
            self.assertTrue(ds[0]['time_mask'].all())
            lengths = []
            def predict(model, anchors, scales, keys, metadata, config, **kw):
                lengths.extend(a.shape[-1] for a in anchors)
                self.assertEqual(config.window_seconds, 75)
                return [torch.as_tensor(a).repeat(12, 1) for a in anchors]
            with patch('baselines.B7.evaluate.predict_records', side_effect=predict):
                evaluate(None, manifest, SamplingConfig(window_seconds=75, core_seconds=75), 'public75')
            self.assertEqual(lengths, [37500, 37500])

    def test_seven_job_end_to_end_public_parents_and_idempotent_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            huawei_path = manifest_fixture(root)
            ptb_path = public_manifest(root)
            mc = replace(small_config(), patch_size=500)
            cfgs = suite.recipes('g6_19m', 1, 1, 1, 1)
            for cfg in cfgs.values():
                cfg['model'] = json.loads(json.dumps(mc.to_dict()))
            source = root/'source'
            for name, stage in (('P', 'public10'), ('Q', 'public40')):
                directory = source/name
                directory.mkdir(parents=True)
                value = payload(B7Transformer(mc))
                old = copy.deepcopy(cfgs['H10'])
                old['stage'] = stage
                old['sampling'] = asdict(SamplingConfig(window_seconds=10 if name == 'P' else 40))
                value.update(stage=stage, update=1, best_r=.5, score_protocol=SCORE_PROTOCOL,
                             identity={'config': old})
                if name == 'Q':
                    value['initialization'] = {'kind': 'previous_B7T_stage',
                        'sha256': file_sha256(source/'P/best_r.pt')}
                save_checkpoint(directory/'best_r.pt', value)
                suite.write(directory/'config.json', old)
                suite.write(directory/'completed.json', {'completed': True, 'update': 1, 'best_r': .5})
            args = Namespace(preset='g6_19m', source_run=source, huawei_manifest=huawei_path,
                ptb_manifest=ptb_path, output=root/'out', gpu='2', batch_size=1,
                eval_batch_size=1, updates=1, q_updates=1, dry_run=False)
            def execute(command, log_path, gpu):
                option = lambda key: command[command.index(key)+1]
                a = Namespace(config=option('--config'), manifest=option('--manifest'), output=option('--output'),
                    device='cpu', resume='--resume' in command,
                    init_checkpoint=option('--init-checkpoint') if '--init-checkpoint' in command else None)
                with patch.object(trainer, 'evaluate', return_value=metrics_fixture(.5)):
                    trainer.train(a)
            with patch.object(suite, 'recipes', return_value=cfgs), patch.object(suite, 'execute', side_effect=execute) as call:
                self.assertEqual(suite.run(args), 0)
                self.assertEqual(call.call_count, 7)
            status = read_manifest(args.output/'ptb75_manifest.json')
            self.assertEqual(len(status['records']), 4)
            status = json.loads((args.output/'pipeline_status.json').read_text())
            self.assertTrue(status['success'])
            for name, report in status['results'].items():
                expected = 'public10' if name == 'Q75' else ('public40' if name.startswith('Q40_') else 'public75')
                self.assertEqual(report['initialization']['source_stage'], expected)
                self.assertEqual(report['updates'], 1)
            with patch.object(suite, 'recipes', return_value=cfgs), patch.object(suite, 'execute') as call:
                self.assertEqual(suite.run(args), 0)
                call.assert_not_called()
            # A wrong public checkpoint hash must be rejected before loading it as initialization.
            config = args.output/'configs/Q40_H10.json'
            cfg = json.loads(config.read_text())
            cfg['initialization']['checkpoint_sha256'] = '0'*64
            config.write_text(json.dumps(cfg))
            a = Namespace(config=str(config), manifest=str(huawei_path), output=str(root/'bad'),
                device='cpu', resume=False, init_checkpoint=str(source/'Q/best_r.pt'))
            with self.assertRaisesRegex(ValueError, 'pinned public EMA'):
                trainer.train(a)


if __name__ == '__main__':
    unittest.main()
