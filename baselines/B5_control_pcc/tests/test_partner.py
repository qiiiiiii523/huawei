import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from baselines.B5_best import SAMPLING
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_best.combine import combine
from baselines.B5.metadata import encode_demographics
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_control_pcc.prepare import check_train_pair, validation_audit, prepare
from baselines.B5_control_pcc.core import generate, model_inputs
from baselines.B5_control_pcc.losses import objective
from baselines.B5_control_pcc.evaluate import evaluate
from baselines.B5_control_pcc import GROUPS, LEGACY_CACHE_PROTOCOL
from baselines.B5_control_pcc.common import load_study, load_cache
from baselines.B5_best.tests.test_hybrid import fixture
from baselines.B5_best.bundle import export_bundle
from baselines.B5_control_pcc.export import export
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.validate import reference_settings

torch.set_num_threads(2)


def packed(bad_i=False):
    raw = np.tile(np.sin(np.arange(60000) / 30).astype(np.float32), (12, 1))
    mask = np.ones(12, bool)
    mask[0] = not bad_i
    slow = np.zeros((11, 6000), np.float32)
    target = combine(raw, np.zeros((11, 60000)), .75)
    if bad_i:
        target[1:] *= -1
    return {'anchor': raw[:1].reshape(1, 6000, 10).mean(-1), 'target_uV': target,
            'fm_uV': raw, 'mask': mask, 'slow_target': slow, **encode_demographics({'age': 40})}


def entry(root, name, values, subject=None):
    relative = name + '.npz'
    np.savez(root / relative, **values)
    return {'file': relative, 'sha256': sha(root / relative), 'data_sha256': name,
            'target_record_id': name, 'subject_id': subject or name}


def validation_fixture(root):
    entries, rows = [], {}
    for task, devices in (('task1', ['watch_ecg']), ('task2', ['ecg_machine_d6', 'body_scale_d6'])):
        rows[task] = []
        for dev in devices:
            for name in ('good', 'bad'):
                values = packed(name == 'bad')
                item = entry(root, task + dev + name, values, subject=name)
                item.update(target_record_id=name, data_sha256=name, pair_id=dev + name,
                            cache_key=task + ':' + dev + name)
                entries.append(item)
                rows[task] += [{'pair_id': dev + name, 'target_record_id': name, 'subject_id': name,
                               'input_type': dev, 'start_sample_500hz': str(s), 'expected_window_count': '12'}
                              for s in range(0, 60000, 5000)]
    return {'validation': entries, 'validation_rows': rows, 'validation_root': str(root)}


class PartnerTests(unittest.TestCase):
    def test_export_keeps_fm_and_scopes_new_reference_without_training_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp, sp, ip, profile = fixture(root)
            initial = export_bundle(fp, sp, ip, root / 'initial.pt', profile)
            # Use the exact deployable S50 state; no optimizer or training step.
            from baselines.B5_best.slow import SlowCNN
            model = SlowCNN()
            model.load_state_dict(initial['slow']['model'])
            score = {'task1_r_missing11': .71, 'task2_r_missing11': .70, 'main_correlation': .705}
            cp = {'group': GROUPS[1], 'epoch': 5, 'validation_score': score}
            study = {'beta': .75, 'sampling': dict(SAMPLING), 'score_protocol': SCORE_PROTOCOL}
            (root / 'study.json').write_text('{}', encoding='utf-8')
            (root / 'checkpoint.pt').write_bytes(b'fixture')
            with patch('baselines.B5_control_pcc.export.load_checkpoint', return_value=(cp, study, initial, model)):
                exported = export(root / 'study.json', root / 'checkpoint.pt', root / 'export.pt')
            actual = load_bundle(root / 'export.pt')
            self.assertEqual(actual['fm_state_sha256'], initial['fm_state_sha256'])
            self.assertEqual(actual['reference_score_protocol'], SCORE_PROTOCOL)
            self.assertTrue(reference_settings(actual)[2])
            self.assertFalse(reference_settings(actual, legacy_target_qc_only=True)[2])
            self.assertFalse(actual['prepared_data_required_at_inference'])

    def test_only_train_fm_may_change_and_input_qc_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = entry(root, 'a', packed())
            changed = packed()
            changed['fm_uV'][1:] *= 2
            b = entry(root, 'b', changed)
            b.update(target_record_id='a', subject_id='a', data_sha256='a')
            self.assertGreater(check_train_pair(root, a, root, b), 0)
            changed['slow_target'][0, 0] = 2
            np.savez(root / b['file'], **changed)
            with self.assertRaisesRegex(ValueError, 'outside FM'):
                check_train_pair(root, a, root, b)
            a = entry(root, 'a', packed(True))
            b = dict(a)
            with self.assertRaisesRegex(ValueError, 'reliable I'):
                check_train_pair(root, a, root, b)

    def test_cached_bad_i_is_excluded_per_device_without_deleting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = validation_fixture(root)
            before = {e['file']: sha(root / e['file']) for e in manifest['validation']}
            audit = validation_audit(manifest, root, {'train_patient'})
            self.assertEqual(audit['task2']['body_scale_d6']['eligible_unique_targets'], 1)
            study = {'training': {'batch_records': 2}, 'sampling': dict(SAMPLING), 'beta': .75,
                     'reference_score': {'task1_r_missing11': 1., 'task2_r_missing11': 1., 'main_correlation': 1.}}
            bundle = {'fm': {'scales': {'d12': [1.] * 12}}, 'reference_tolerance': 2e-5}
            with patch('baselines.B5_control_pcc.evaluate.generate',
                       side_effect=lambda model, batch: torch.zeros(len(batch['anchor']), 11, 6000)):
                score = evaluate(torch.nn.Identity(), bundle, study, None, manifest, torch.device('cpu'),
                                 root / 'evaluation', GROUPS[1], 0, initial=True)
            self.assertAlmostEqual(score['main_correlation'], 1, places=6)
            self.assertEqual(score['score_protocol'], SCORE_PROTOCOL)
            self.assertTrue(read(root / 'evaluation/evaluation.json')['reference_matched'])
            self.assertTrue((root / 'evaluation/task2/input_qc_exclusions.csv').exists())
            self.assertEqual(before, {e['file']: sha(root / e['file']) for e in manifest['validation']})

    def test_full_record_pcc_gradient_depends_on_fm_partner_only_cnn_has_gradient(self):
        p = torch.sin(torch.arange(6000) / 20).reshape(1, 1, 6000).repeat(1, 11, 1).requires_grad_()
        wave = torch.sin(torch.arange(60000) / 15)[None, None].repeat(1, 12, 1)
        batch = {'mask': torch.ones(1, 12, dtype=torch.bool), 'slow_target': torch.zeros_like(p),
                 'fm_uV': wave, 'target_uV': wave + .1 * torch.cos(torch.arange(60000) / 100)}
        a, parts = objective(p, batch, torch.ones(12), .5, .1, beta=.75)
        b, other = objective(p, {**batch, 'fm_uV': -wave}, torch.ones(12), .5, .1, beta=.75)
        torch.testing.assert_close(parts['original_F_loss'], other['original_F_loss'])
        self.assertNotAlmostEqual(float(a.detach()), float(b.detach()), places=5)
        parts['final_hybrid_pcc'].backward()
        self.assertGreater(float(p.grad.abs().sum()), 0)
        self.assertIsNone(batch['fm_uV'].grad)
        self.assertNotIn('target_uV', model_inputs({**batch, 'anchor': 0, 'numeric': 0,
                                                 'sex': 0, 'field_mask': 0, 'age_topcoded': 0}))

    def test_prepare_imports_sources_without_mutating_legacy_study_or_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp, sp, ip, profile = fixture(root)
            bundle = export_bundle(fp, sp, ip, root / 'initial.pt', profile)
            runs = [root / 'old', root / 'new']
            for i, run in enumerate(runs):
                (run / 'shared').mkdir(parents=True)
                (run / 'cache').mkdir()
                source_bundle = copy.deepcopy(bundle)
                if i == 0:
                    key = next(iter(source_bundle['fm']['ema']))
                    source_bundle['fm']['ema'][key] = source_bundle['fm']['ema'][key] + .01
                    source_bundle['fm_state_sha256'] = state_sha(source_bundle['fm']['ema'])
                torch.save(source_bundle, run / 'shared/components.pt')
                source_study = {'bundle_sha256': sha(run / 'shared/components.pt'), 'manifests': {'train': 'same'}, 'train_records': 1,
                                'source_slow_config': {'same': True}, 'combo_weight': .1, 'residual_weight': .5,
                                'mean_weight': 0., 'beta': .75, 'residual_width_native': 51,
                                'training': {'epochs': 20, 'batch_records': 1, 'seed': 42, 'validate_every': 5,
                                             'learning_rate': 1e-4, 'weight_decay': 0., 'gradient_clip': 1.}}
                write(run / 'shared/study.json', source_study)
                values = packed()
                values['fm_uV'][1:] *= (i + 1)
                train = entry(run / 'cache', 'train', values, 'train_patient')
                manifest = validation_fixture(run / 'cache')
                manifest.update(protocol=LEGACY_CACHE_PROTOCOL, completed=True, FM_optimizer_updates=0,
                                sampling=dict(SAMPLING), study_sha256=sha(run / 'shared/study.json'), train=[train])
                write(run / 'cache/cache.json', manifest)
            reference = {'score_protocol': SCORE_PROTOCOL, 'task1_r_missing11': 1.,
                         'task2_r_missing11': 1., 'main_correlation': 1.}
            write(root / 'reference.json', reference)
            files = [p for run in runs for p in run.rglob('*') if p.is_file()]
            before = {str(p): sha(p) for p in files}
            args = SimpleNamespace(old_run=str(runs[0]), control100_run=str(runs[1]),
                                   reference_report=str(root / 'reference.json'), epochs=20, output=str(root / 'out'))
            with patch('baselines.B5_control_pcc.prepare.verified_source', return_value=(root / 'initial.pt', bundle)):
                status = prepare(args)
            self.assertFalse(status['cache_regenerated'])
            self.assertEqual(before, {str(p): sha(p) for p in files})
            study, _ = load_study(root / 'out/study.json')
            view = load_cache(root / 'out/cache_view.json', root / 'out/study.json')
            self.assertEqual(study['score_protocol'], SCORE_PROTOCOL)
            self.assertEqual(set(view['train_sources']), set(GROUPS))
            with self.assertRaises(FileExistsError):
                prepare(args)


if __name__ == '__main__':
    unittest.main()
