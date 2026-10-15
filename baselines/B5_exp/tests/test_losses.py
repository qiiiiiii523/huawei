"""Synthetic forward/backward and workflow checks; optimizer updates are forbidden."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5_exp import PROTOCOL
from baselines.B5_exp.core import SlowCNN, sha, json_write, csv_write, objective
from baselines.B5_exp.losses import auxiliary_terms, paired_objective, smooth_native, LOSS_PROTOCOL
from baselines.B5_exp.check_loss import check, initializer
from baselines.B5_exp import train_loss
from baselines.B5_exp.loss_metrics import Reference
from baselines.B5_exp.evaluate import run as evaluate
from baselines.B5_exp.train import state_hash
from baselines.B5_exp.compare_loss import compare
from ecg12gen.evaluate import competition_score


def fixture(root):
    prepared = root / 'prepared'
    prepared.mkdir()
    t = np.arange(6000) / 50
    arrays = {'anchor': np.stack([np.sin(t)[None], np.cos(t)[None]]).astype(np.float32),
              'target': np.stack([np.stack([.2 * np.sin(t / (j + 1)) + .1 * j for j in range(11)])] * 2).astype(np.float32),
              'mask': np.ones((2, 11), bool), 'numeric': np.zeros((2, 3), np.float32),
              'sex': np.full(2, 2, np.int64), 'field_mask': np.zeros((2, 4), bool),
              'age_topcoded': np.zeros((2, 1), np.float32)}
    for split in ('train', 'validation'):
        np.savez_compressed(prepared / (split + '.npz'), **arrays)
    source = root / 'source.pt'
    source.write_bytes(b'synthetic frozen FM reference')
    original = root / 'original_evaluation'
    full_t = np.arange(60000) / 500
    wave = np.stack([150 * np.sin(full_t / (j + 1)) + 30 * np.sin(full_t * 20) + j * 10 for j in range(12)]).astype(np.float32)
    metadata = {'train': [{'target_record_id': f'train-{i}'} for i in range(2)],
                'validation': [{'target_record_id': name, 'subject_id': name, 'shared_with_task1': name == 'a', 'input_I_good': True} for name in ('a', 'b')],
                'pairs': {}}
    specs = {'task1': [('watch_ecg', 'a')], 'task2': [('ecg_machine_d6', 'a'), ('ecg_machine_d6', 'b'), ('body_scale_d6', 'a'), ('body_scale_d6', 'b')]}
    files = {}
    for task, pairs in specs.items():
        folder = original / task
        folder.mkdir(parents=True)
        rows, blocks, pair_metadata = [], [], []
        for n, (device, name) in enumerate(pairs):
            pair = f'{task}-{n}'
            ids = list(range(n * 12, (n + 1) * 12))
            pair_metadata.append({'pair_id': pair, 'target_record_id': name, 'input_type': device, 'indices': ids})
            for w in range(12):
                rows.append({'pair_id': pair, 'target_record_id': name, 'subject_id': name,
                             'start_sample_500hz': str(w * 5000), 'expected_window_count': '12', 'input_type': device})
                blocks.append(wave[:, w * 5000:(w + 1) * 5000])
        metadata['pairs'][task] = pair_metadata
        np.save(folder / 'target_uV.npy', np.stack(blocks))
        np.save(folder / 'prediction_uV.npy', np.stack(blocks) * 1.02)
        csv_write(folder / 'window_metadata.csv', rows)
        for name in ('target_uV.npy', 'prediction_uV.npy', 'window_metadata.csv'):
            files[task + '/' + name] = sha(folder / name)
    json_write(prepared / 'records.json', metadata)
    info = {'protocol': PROTOCOL, 'completed': True, 'scale_uV': [100.] * 12,
            'applied_score_protocol': 'correlation50_bonus10_target_record_v2',
            'checkpoint': str(source), 'checkpoint_sha256': sha(source),
            'evaluation_dir': str(original), 'evaluation_files': files,
            'files': {name: sha(prepared / name) for name in ('train.npz', 'validation.npz', 'records.json')}}
    json_write(prepared / 'prepared.json', info)
    torch.manual_seed(42)
    model = SlowCNN()
    init = root / 'S30_best.pt'
    config = {'context_seconds': 30, 'model_channels': 48, 'prepared_sha256': sha(prepared / 'prepared.json')}
    torch.save({'protocol': PROTOCOL, 'config': config, 'epoch': 50, 'model': model.state_dict()}, init)
    return prepared, info, init


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mean_and_residual_separate_offset_from_target_matched_oscillation(self):
        y = torch.sin(torch.arange(6000.) * .15)[None, None].expand(1, 11, -1)
        mask = torch.ones(1, 11, dtype=torch.bool)
        exact = auxiliary_terms(y, y, mask)
        self.assertEqual([float(v) for v in exact], [0., 0.])
        mean, residual = auxiliary_terms(y + 2., y, mask)
        self.assertAlmostEqual(float(mean), 1.5, places=5)
        self.assertLess(float(residual), 1e-10)
        perturbation = .2 * torch.sin(torch.arange(6000.) * 1.2)
        _, residual = auxiliary_terms(y + perturbation, y, mask)
        self.assertGreater(float(residual), .005)
        # Nonzero oscillations in the target are preserved, not suppressed toward zero.
        self.assertGreater(float((y - smooth_native(y)).square().mean()), .1)

    def test_mask_nan_and_gradient_and_exact_control(self):
        y = torch.linspace(-1, 1, 6000)[None, None].expand(1, 11, -1).clone()
        mask = torch.ones(1, 11, dtype=torch.bool)
        mask[:, -1] = False
        y[:, -1] = float('nan')
        p = torch.zeros_like(y, requires_grad=True)
        weights = {'mean': .4, 'residual': .7}
        base, _ = objective(p, y, mask)
        control, _ = paired_objective(p, y, mask, 'C', weights)
        self.assertTrue(torch.equal(control, base))
        for group in ('M', 'F'):
            value, parts = paired_objective(p, y, mask, group, weights)
            grad = torch.autograd.grad(value, p, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertEqual(float(grad[:, -1].abs().sum()), 0.)
            self.assertEqual(parts['weighted_residual' if group == 'M' else 'weighted_mean'], 0.)
        with self.assertRaises(ValueError):
            smooth_native(p, 50)

    def test_control_mean_residual_configuration_only_changes_one_loss(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prepared, info, init = fixture(root)
            report = root / 'calibration.json'
            with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('Forbidden local training')):
                check(SimpleNamespace(prepared=str(prepared), init_checkpoint=str(init), report=str(report),
                                      device='cpu', residual_width=51, gradient_fraction=.1))
            calibration = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(calibration['optimizer_updates'], 0)
            self.assertEqual(calibration['prediction_shape'], [2, 11, 6000])
            configs = []
            for group in ('C', 'M', 'F'):
                args = SimpleNamespace(group=group, epochs=20, batch_records=2, validate_every=5,
                                       calibration=str(report), init_checkpoint=str(init))
                configs.append(train_loss.configuration(args, prepared))
            ignore = {'group', 'mean_weight', 'residual_weight'}
            self.assertEqual([{k: v for k, v in c.items() if k not in ignore} for c in configs],
                             [{k: v for k, v in configs[0].items() if k not in ignore}] * 3)
            self.assertEqual(configs[0]['seed'], 42)  # Backward-compatible default.
            args.seed = 43
            changed_seed = train_loss.configuration(args, prepared)
            self.assertEqual(changed_seed['seed'], 43)
            self.assertEqual({k:v for k,v in changed_seed.items() if k != 'seed'},
                             {k:v for k,v in configs[-1].items() if k != 'seed'})
            for invalid in (-1, 2**32, True):
                args.seed = invalid
                with self.assertRaisesRegex(ValueError, 'seed'):
                    train_loss.configuration(args, prepared)
            wrong = root / 'wrong.pt'
            torch.save({'protocol': 'FM', 'config': {'context_seconds': 30}}, wrong)
            with self.assertRaises(ValueError):
                initializer(wrong, prepared)
            with self.assertRaises(FileExistsError):
                check(SimpleNamespace(report=str(report)))

    def test_initialize_validate_save_resume_without_optimizer_updates(self):
        class StopBeforeTraining(RuntimeError):
            pass

        class NoTrainingLoader:
            def __iter__(self):
                raise StopBeforeTraining()

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prepared, info, init = fixture(root)
            report = root / 'calibration.json'
            check(SimpleNamespace(prepared=str(prepared), init_checkpoint=str(init), report=str(report),
                                  device='cpu', residual_width=51, gradient_fraction=.1))
            args = SimpleNamespace(group='C', epochs=20, batch_records=2, validate_every=5, calibration=str(report),
                                   init_checkpoint=str(init), prepared=str(prepared), output=str(root / 'C'), device='cpu', resume=None, seed=43)
            original_loader = train_loss.DataLoader

            def loader(dataset, **kwargs):
                if kwargs.get('shuffle'):
                    self.assertEqual(kwargs['generator'].initial_seed(), args.seed)
                return NoTrainingLoader() if kwargs.get('shuffle') else original_loader(dataset, **kwargs)

            source_hash = sha(init)
            with patch.object(train_loss, 'DataLoader', side_effect=loader), patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('Forbidden local training')):
                with self.assertRaises(StopBeforeTraining):
                    train_loss.run(args)
                last = torch.load(root / 'C/last.pt', map_location='cpu', weights_only=True)
                self.assertEqual(last['epoch'], 0)
                self.assertEqual(last['config']['seed'], 43)
                self.assertTrue(torch.equal(last['loader_rng'], torch.Generator().manual_seed(43).get_state()))
                first_record = json.loads((root / 'C/history.jsonl').read_text(encoding='utf-8').splitlines()[0])
                self.assertEqual(first_record['seed'], 43)
                self.assertEqual(last['optimizer']['state'], {})
                self.assertEqual(state_hash(last['model']), state_hash(torch.load(init, weights_only=True)['model']))
                self.assertEqual(last['best']['slow_r']['epoch'], 0)
                self.assertEqual(last['best']['score']['epoch'], 0)
                args.resume = str(root / 'C/last.pt')
                # Simulate an interrupted alias save; last.pt repairs best files.
                (root / 'C/best_score.pt').write_bytes(b'interrupted candidate')
                with self.assertRaises(StopBeforeTraining):
                    train_loss.run(args)
                repaired = torch.load(root / 'C/best_score.pt', weights_only=True)
                self.assertEqual(repaired['epoch'], 0)
                args.seed = 44
                # This fails before reading the training iterator or updating weights.
                with self.assertRaisesRegex(ValueError, 'seed'):
                    train_loss.run(args)
                args.seed = 43
                args.group = 'M'
                with self.assertRaises(ValueError):
                    train_loss.run(args)
            self.assertEqual(sha(init), source_hash)
            evaluate(SimpleNamespace(prepared=str(prepared), checkpoint=str(root / 'C/best_score.pt'),
                                     output=str(root / 'C/evaluation_best_score'), device='cpu', no_plots=True))
            final = json.loads((root / 'C/evaluation_best_score/evaluation.json').read_text(encoding='utf-8'))
            initial = json.loads((root / 'C/initial_evaluation.json').read_text(encoding='utf-8'))
            for key in ('competition_total_score', 'task1_r_missing11', 'task2_r_missing11'):
                self.assertAlmostEqual(final['scores'][1][key], initial['hybrid_score'][key], places=6)
            self.assertTrue((root / 'C/evaluation_best_score/wave_record_lead.csv').is_file())

    def test_comparison_audits_budget_and_current_candidate_hashes(self):
        # Pure synthetic report fixtures; no optimizer or training loop is run.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = []
            score = competition_score(.65, .60, 650, 600)
            for group in ('C', 'M', 'F'):
                path = root / group
                path.mkdir()
                paths.append(path)
                config = {'group': group, 'experiment_protocol': LOSS_PROTOCOL, 'epochs': 20,
                          'calibrated_weights': {'mean': .2, 'residual': .3},
                          'mean_weight': .2 if group == 'M' else 0., 'residual_weight': .3 if group == 'F' else 0.,
                          'prepared_sha256': 'same data', 'initial_weights_sha256': 'same initialization',
                          'calibration_sha256': 'same train-only calibration', 'context_seconds': 30}
                json_write(path / 'resolved_config.json', config)
                json_write(path / 'completed.json', {'completed': True, 'additional_epochs': 20})
                json_write(path / 'initial_evaluation.json', {'hybrid_score': score})
                (path / 'history.jsonl').write_text(''.join(json.dumps({'epoch': epoch}) + '\n' for epoch in range(21)), encoding='utf-8')
                for candidate in ('best_slow_r', 'best_score', 'last'):
                    checkpoint = path / (candidate + '.pt')
                    checkpoint.write_bytes(b'synthetic checkpoint hash fixture')
                    folder = path / ('evaluation_' + candidate)
                    folder.mkdir()
                    json_write(folder / 'evaluation.json', {'completed': True, 'group': group,
                              'experiment_protocol': LOSS_PROTOCOL, 'epoch': 20 if candidate == 'last' else 0,
                              'probe_checkpoint_sha256': sha(checkpoint), 'prepared_sha256': 'same data',
                              'scores': [{'method': 'learned_slow_hybrid', **score}]})
                    csv_write(folder / 'wave_summary.csv', [{'cohort': 'all_unique', 'quality': 'good', 'method': 'hybrid', 'raw_r': .6}])
            compare(paths, root / 'comparison')
            self.assertTrue((root / 'comparison/report.md').is_file())
            (paths[1] / 'last.pt').write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'Stale'):
                compare(paths, root / 'stale')
            config = json.loads((paths[2] / 'resolved_config.json').read_text(encoding='utf-8'))
            config['epochs'] = 25
            json_write(paths[2] / 'resolved_config.json', config)
            with self.assertRaisesRegex(ValueError, 'more than'):
                compare(paths, root / 'unpaired')

    def test_candidate_selection_keeps_initializer_and_separates_objectives(self):
        best = {k: {'value': -float('inf'), 'epoch': None} for k in ('slow_r', 'score')}
        def scores(r, score):
            return {'slow_r_unique_quality_good_50Hz': r, 'hybrid_score': {'competition_total_score': score}}
        self.assertEqual(train_loss.update_bests(best, scores(.5, 35), 0), ['slow_r', 'score'])
        self.assertEqual(train_loss.update_bests(best, scores(.4, 34), 5), [])
        self.assertEqual(train_loss.update_bests(best, scores(.6, 34), 10), ['slow_r'])
        self.assertEqual(train_loss.update_bests(best, scores(.55, 36), 15), ['score'])
        self.assertEqual(best['slow_r']['epoch'], 10)
        self.assertEqual(best['score']['epoch'], 15)


if __name__ == '__main__':
    unittest.main()
