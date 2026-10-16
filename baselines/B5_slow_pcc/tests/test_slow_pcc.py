"""CPU synthetic checks, without a training loop or optimizer steps."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5.checkpoint import atomic_save
from baselines.B5.metadata import encode_demographics
from baselines.B5_best import SAMPLING, SCORE_PROTOCOL
from baselines.B5_best.bundle import export_bundle, load_bundle
from baselines.B5_best.combine import upsample_native, combine
from baselines.B5_best.common import write, read, sha, state_sha, csv_write
from baselines.B5_best.predict import Predictor
from baselines.B5_best.slow import SlowCNN, predict_native
from baselines.B5_best.tests.test_hybrid import fixture
from baselines.B5_slow_pcc import PROTOCOL, CACHE_PROTOCOL
from baselines.B5_slow_pcc.cache import generate_fm, build
from baselines.B5_slow_pcc.common import code_digest, load_cache, Records
from baselines.B5_slow_pcc.core import generate, upsample, model_inputs
from baselines.B5_slow_pcc.evaluate import evaluate
from baselines.B5_slow_pcc.export import export
from baselines.B5_slow_pcc.losses import old_f_objective, objective
from baselines.B5_slow_pcc.compare import compare

torch.set_num_threads(2)


class SlowPCCTests(unittest.TestCase):
    def test_native_inference_exactly_matches_original_frozen_predictor(self):
        model = SlowCNN().eval()
        fields = {k: torch.from_numpy(v[None]) for k, v in encode_demographics({"age": 40}).items()}
        inputs = {"anchor": torch.sin(torch.arange(6000) / 50).reshape(1, 1, 6000), **fields}
        with torch.no_grad():
            torch.testing.assert_close(generate(model, inputs), predict_native(model, inputs), atol=0, rtol=0)

    def test_interpolation_matches_deployment_coordinates_and_has_backward(self):
        values = torch.randn(1, 11, 6000, dtype=torch.float64, requires_grad=True)
        out = upsample(values)
        expected = upsample_native(values[0].detach().numpy())
        np.testing.assert_allclose(out[0].detach().numpy(), expected, atol=1e-11, rtol=1e-11)
        out.sum().backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, 10.), atol=1e-12, rtol=1e-12)

    def test_F_loss_matches_actual_legacy_reference_with_flagged_NaN(self):
        # Golden values computed by the existing S50-F B5_exp.losses.paired_objective.
        v = torch.arange(2 * 11 * 6000, dtype=torch.float32).reshape(2, 11, 6000)
        p = .7 * torch.sin(v / 37) + .2 * torch.cos(v / 119)
        y = .9 * torch.sin(v / 41) + .1
        mask = torch.ones(2, 11, dtype=torch.bool); mask[0, 7] = False; mask[1, 2] = False
        y[~mask] = float('nan')
        loss, parts = old_f_objective(p, y, mask, 2.5890171538829843, 51)
        self.assertAlmostEqual(float(loss), .42847129702568054, places=6)
        self.assertAlmostEqual(float(parts['slow_huber']), .3235340416431427, places=6)
        self.assertAlmostEqual(float(parts['slow_pcc']), 1.0089232921600342, places=6)
        self.assertAlmostEqual(float(parts['F_residual']), .0015623365761712193, places=7)

    def test_extra_term_backpropagates_only_to_slow_prediction(self):
        pred = torch.randn(1, 11, 6000, requires_grad=True)
        batch = {'mask': torch.ones(1, 12, dtype=torch.bool), 'fm_uV': torch.randn(1, 12, 60000),
                 'target_uV': torch.randn(1, 12, 60000), 'slow_target': torch.randn(1, 11, 6000)}
        base, parts0 = objective(pred, batch, torch.ones(12), 2.5890171538829843, 0.)
        extra, parts = objective(pred, batch, torch.ones(12), 2.5890171538829843, .1)
        torch.testing.assert_close(extra - base, .1 * parts['final_hybrid_pcc'])
        parts['final_hybrid_pcc'].backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0.)
        self.assertFalse(batch['fm_uV'].requires_grad)
        batch['fm_uV'].requires_grad_(True)
        with self.assertRaises(ValueError):
            objective(pred, batch, torch.ones(12), 2.5890171538829843, .1)

    def test_combo_loss_uses_the_same_beta075_as_deployment(self):
        pred = torch.randn(1, 11, 6000)
        batch = {'mask': torch.ones(1, 12, dtype=torch.bool), 'fm_uV': torch.randn(1, 12, 60000),
                 'target_uV': torch.randn(1, 12, 60000), 'slow_target': torch.randn(1, 11, 6000)}
        _, parts = objective(pred, batch, torch.ones(12), 2.5890171538829843, .1, beta=.75)
        output = combine(batch['fm_uV'][0].numpy(), upsample_native(pred[0].double().numpy()), .75)
        expected = 1 - np.mean([np.corrcoef(output[i], batch['target_uV'][0, i].numpy())[0, 1]
                                for i in range(1, 12)])
        self.assertAlmostEqual(float(parts['final_hybrid_pcc']), float(expected), places=7)

    def test_CNN_input_allow_list_excludes_labels_and_fixed_FM(self):
        fields = {'anchor': 1, 'numeric': 2, 'sex': 3, 'field_mask': 4, 'age_topcoded': 5,
                  'target_uV': object(), 'fm_uV': object(), 'slow_target': object(), 'mask': object()}
        self.assertEqual(set(model_inputs(fields)), {'anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded'})

    def test_full_generate_to_combo_loss_graph_reaches_CNN_weights(self):
        class TinySlow(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.head = torch.nn.Conv1d(1, 11, 3, padding=1)
            def forward(self, anchor, **metadata): return self.head(anchor)
        model = TinySlow()
        fields = {k: torch.from_numpy(v[None]) for k, v in encode_demographics({}).items()}
        batch = {'anchor': torch.randn(1, 1, 6000), **fields,
                 'mask': torch.ones(1, 12, dtype=torch.bool), 'fm_uV': torch.randn(1, 12, 60000),
                 'target_uV': torch.randn(1, 12, 60000), 'slow_target': torch.randn(1, 11, 6000)}
        prediction = generate(model, model_inputs(batch))
        _, parts = objective(prediction, batch, torch.ones(12), 2.5890171538829843, .1)
        parts['final_hybrid_pcc'].backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.)

    def test_fixed_FM_sampler_receives_only_visible_inputs_and_correct_noise_keys(self):
        model = torch.nn.Conv1d(1, 1, 1).eval().requires_grad_(False)
        meta = encode_demographics({})
        seen = []
        def fake_sample(model, inputs, keys, **settings):
            self.assertEqual(settings, dict(SAMPLING))
            self.assertEqual(set(inputs), {'anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded'})
            seen.extend(keys)
            return torch.ones(len(keys), 12, 5000)
        with patch('baselines.B5_slow_pcc.cache.sample', side_effect=fake_sample):
            actual = generate_fm(model, np.zeros((1, 60000), np.float32), meta, 'train_record',
                                 np.ones(12), dict(SAMPLING), torch.device('cpu'), 4)
        self.assertEqual(seen, ['train_record:' + str(s) for s in range(0, 60000, 5000)])
        self.assertEqual(actual.shape, (12, 60000))

    def test_cache_scoring_export_and_paired_comparison_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp, sp, ip, profile = fixture(root)
            bundle = export_bundle(fp, sp, ip, root / 'components.pt', profile)
            study_path = root / 'study.json'
            study = {'protocol': PROTOCOL, 'sampling': dict(SAMPLING), 'score_protocol': SCORE_PROTOCOL,
                     'beta': .75, 'code_sha256': code_digest(), 'bundle_sha256': sha(root / 'components.pt'),
                     'training': {'batch_records': 1}}
            write(study_path, study)
            cache = root / 'cache'; cache.mkdir()
            phase = np.arange(60000)
            raw = np.tile((100 * np.sin(phase / 25)).astype(np.float32), (12, 1))
            data = {'anchor': raw[:1].reshape(1, 6000, 10).mean(-1) / 100,
                    'target_uV': raw, 'fm_uV': raw, 'mask': np.ones(12, bool),
                    'slow_target': raw[1:].reshape(11, 6000, 10).mean(-1) / 100,
                    **encode_demographics({})}
            np.savez(cache / 'record.npz', **data)
            entry = {'target_record_id': 'record', 'subject_id': 'patient', 'file': 'record.npz',
                     'sha256': sha(cache / 'record.npz')}
            rows = {}
            for task, devices in [('task1', ['watch_ecg']), ('task2', ['ecg_machine_d6', 'body_scale_d6'])]:
                rows[task] = [{'pair_id': d, 'target_record_id': 'record', 'subject_id': 'patient',
                               'input_type': d, 'start_sample_500hz': str(s), 'expected_window_count': '12'}
                              for d in devices for s in range(0, 60000, 5000)]
            manifest = {'protocol': CACHE_PROTOCOL, 'completed': True, 'study_sha256': sha(study_path),
                        'train': [], 'validation': [entry], 'validation_rows': rows}
            write(cache / 'cache.json', manifest)
            load_cache(cache, study_path)
            self.assertEqual(Records(cache, [entry])[0]['target_uV'].shape, (12, 60000))
            model = SlowCNN().eval()
            run = root / 'run'
            for group in ('control', 'pcc'):
                folder = run / group / 'evaluation'
                score = evaluate(model, bundle, study, cache, manifest, torch.device('cpu'), folder, group, 0)
                write(run / group / 'initialization.json', {'study_sha256': sha(study_path)})
                for which in ('best', 'last'):
                    (run / group / f'latest_{which}_evaluation.txt').write_text(str(folder))
            compare(run, run / 'comparison', plots=False)
            self.assertAlmostEqual(read(run / 'comparison/comparison.json')['best_delta_mean_r'], 0.)
            cp_path = root / 'slow.pt'
            atomic_save(cp_path, {'protocol': PROTOCOL, 'study_sha256': sha(study_path), 'group': 'pcc',
                        'sampling_config': dict(SAMPLING), 'model': model.state_dict(), 'epoch': 20,
                        'validation_score': score})
            export(study_path, cp_path, root / 'portable.pt')
            portable = load_bundle(root / 'portable.pt')
            self.assertEqual(portable['fm_state_sha256'], bundle['fm_state_sha256'])
            self.assertEqual(portable['slow_beta'], .75)
            self.assertEqual(portable['slow_state_sha256'], state_sha(model.state_dict()))
            engine = Predictor(portable, torch.device('cpu'))
            with patch('baselines.B5_best.predict.sample', return_value=torch.zeros(8, 12, 5000)) as sampler:
                # Use a batch size dividing 12 so the cheap sampler mock has consistent shape.
                sampler.side_effect = lambda m, x, keys, **kw: torch.zeros(len(keys), 12, 5000)
                result = engine.predict(raw[:1], {}, 'new_patient', batch_size=4)
            self.assertEqual(result.shape, (12, 60000))
            self.assertTrue(np.isfinite(result).all())
            # Fixed-cache corruption must be detected before either arm can train.
            with (cache / 'record.npz').open('ab') as handle: handle.write(b'changed')
            with self.assertRaises(ValueError): load_cache(cache, study_path)

    def test_cache_build_keeps_train_and_validation_separate_and_resumes_per_record(self):
        phase = np.arange(60000)
        train_raw = np.tile(np.sin(phase / 20).astype(np.float32), (12, 1))
        val_raw = np.tile(np.cos(phase / 23).astype(np.float32), (12, 1))
        class Base:
            def __init__(self, rows, raw): self.rows, self.raw = rows, raw
            def __getitem__(self, i):
                s = int(self.rows[i]['start_sample_500hz'])
                raw = self.raw[:, s:s + 5000]
                return SimpleNamespace(Y_12lead=raw, anchor_i_ecg=raw[:1], target_quality_mask=np.ones(12, bool))
        class Adapter:
            def __init__(self, task):
                training = task == 'train'
                self.raw = train_raw if training else val_raw
                self.manifest_digest = task
                key = 'train_record' if training else 'validation_record'
                patient = 'train_patient' if training else 'validation_patient'
                devices = ['train'] if training else (['watch_ecg'] if task == 'task1' else ['ecg_machine_d6', 'body_scale_d6'])
                self.rows = [{'pair_id': d, 'target_record_id': key, 'subject_id': patient,
                              'input_type': d, 'start_sample_500hz': str(s), 'expected_window_count': '12'}
                             for d in devices for s in range(0, 60000, 5000)]
                self.demographics = SimpleNamespace(get=lambda _: encode_demographics({}))
                self.base = Base(self.rows, self.raw)
            def __len__(self): return len(self.rows)
            def __getitem__(self, i):
                s = int(self.rows[i]['start_sample_500hz'])
                return {'target_uV': self.raw[:, s:s + 5000], 'quality_mask': np.ones(12, bool),
                        'evaluation_metadata': self.rows[i]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); study_path = root / 'study.json'; write(study_path, {'fixture': True})
            train, validation = Adapter('train'), {t: Adapter(t) for t in ('task1', 'task2')}
            source = root / 'historical_validation'; checks = {}
            for task, ds in validation.items():
                folder = source / task; folder.mkdir(parents=True)
                values = np.stack([ds[i]['target_uV'] for i in range(len(ds))])
                np.save(folder / 'prediction_uV.npy', values)
                np.save(folder / 'target_uV.npy', values)
                csv_write(folder / 'window_metadata.csv', ds.rows)
                for path in folder.iterdir(): checks[str(path)] = sha(path)
            study = {'data_config': {}, 'manifests': {'train': 'train', 'task1': 'task1', 'task2': 'task2'},
                     'sampling': dict(SAMPLING), 'train_records': 1, 'validation_unique_targets': 1,
                     'source_evaluation_checks': checks, 'source_validation_evaluation': str(source)}
            bundle = {'fm': {'scales': {'d12': [1.] * 12}}}
            with patch('baselines.B5_slow_pcc.cache.load_study', return_value=(study, bundle)), \
                 patch('baselines.B5_slow_pcc.cache.preprocessing_config'), \
                 patch('baselines.B5_slow_pcc.cache.checkpoint_preprocessor'), \
                 patch('baselines.B5_slow_pcc.cache.build_datasets', return_value=(train, validation)), \
                 patch('baselines.B5_slow_pcc.cache.model_from_checkpoint',
                       side_effect=lambda *a: torch.nn.Conv1d(1, 1, 1)), \
                 patch('baselines.B5_slow_pcc.cache.generate_fm', return_value=train_raw) as inference:
                build(study_path, root / 'cache', 'cpu')
                self.assertEqual(inference.call_count, 1)  # TRAIN only; validation reuses old predictions.
                manifest = load_cache(root / 'cache', study_path)
                self.assertEqual(manifest['train'][0]['target_record_id'], 'train_record')
                self.assertEqual(manifest['validation'][0]['target_record_id'], 'validation_record')
                # Simulate interruption before the final completion marker, retaining per-record progress.
                (root / 'cache/cache.json').rename(root / 'cache/completed_before_test.json')
                inference.reset_mock()
                build(study_path, root / 'cache', 'cpu', resume=True)
                self.assertEqual(inference.call_count, 0)
                self.assertTrue(read(root / 'cache/cache.json')['completed'])


if __name__ == '__main__':
    unittest.main()
