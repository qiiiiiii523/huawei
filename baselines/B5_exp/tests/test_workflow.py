"""Prepare/check/evaluate/compare on synthetic adapters. No optimizer/training."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5_exp import PROTOCOL
from baselines.B5_exp import prepare as preparation
from baselines.B5_exp.check import check
from baselines.B5_exp.evaluate import run as evaluate
from baselines.B5_exp.compare import compare
from baselines.B5_exp.core import SlowCNN, sha, json_write, csv_write, LEADS
from baselines.B5.metadata import encode_demographics


class WorkflowTest(unittest.TestCase):
    def test_synthetic_readonly_workflow_latest_scoring(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'metadata').mkdir()
            run, source_eval = repo / 'original', repo / 'source_evaluation'
            run.mkdir()
            (run / 'best.pt').write_bytes(b'Synthetic reference SHA only; checkpoint loader mocked')
            json_write(run / 'resolved_config.json', {'paths': {}})
            paths = {}
            for key in ('subject_split_csv', 'device_interpretation_qc_csv', 'demographics_csv',
                        'task1_pair_manifest_csv', 'task2_pair_manifest_csv'):
                paths[key] = repo / 'metadata' / (key + '.csv')
                paths[key].write_text('fixture\n', encoding='utf-8')
            strict = repo / 'metadata/d12_strict_pretrain_index.csv'
            strict.write_text('synthetic index\n', encoding='utf-8')
            manifests = {'strict_index': sha(strict), 'demographics_csv': sha(paths['demographics_csv']),
                         'train': 'train-manifest', 'task1': 'val-task1', 'task2': 'val-task2'}
            for key in ('subject_split_csv', 'device_interpretation_qc_csv'):
                manifests[key] = sha(paths[key])
            for task in ('task1', 'task2'):
                cache = repo / (task + '_cache')
                cache.mkdir()
                paths[task + '_output'] = cache
                for suffix in ('window_metadata.csv', 'train_target.npy', 'validation_target.npy'):
                    (cache / (task + '_' + suffix)).write_bytes(b'synthetic source fingerprint')
                manifests[task + '_pair_manifest'] = sha(paths[task + '_pair_manifest_csv'])
                manifests[task + '_window_metadata'] = sha(cache / (task + '_window_metadata.csv'))
                for split in ('train', 'validation'):
                    manifests[task + '_' + split + '_target'] = sha(cache / (task + '_' + split + '_target.npy'))
            time = np.arange(60000) / 500
            waves = {}
            for name, offset in (('a', 100), ('b', -200)):
                waves[name] = np.stack([(offset + lead * 20 + 100 * np.sin(2 * np.pi * time / (5 + lead)) +
                                         40 * np.sin(2 * np.pi * time * 3)).astype(np.float32) for lead in range(12)])
            adapter = {}
            specs = {'task1': [('watch_ecg', 'a1', 'a'), ('watch_ecg', 'a2', 'a')],
                     'task2': [('ecg_machine_d6', 'm1', 'a'), ('ecg_machine_d6', 'm2', 'b'),
                               ('body_scale_d6', 'b1', 'a'), ('body_scale_d6', 'b2', 'b')]}
            for task, pairs in specs.items():
                folder = source_eval / task
                folder.mkdir(parents=True)
                rows, samples, arrays = [], [], []
                for device, pair, target in pairs:
                    for w in range(12):
                        raw = waves[target][:, w * 5000:(w + 1) * 5000]
                        row = {'pair_id': pair, 'target_record_id': target, 'subject_id': 'patient-' + target,
                               'start_sample_500hz': str(w * 5000), 'input_type': device, 'expected_window_count': '12'}
                        rows.append(row)
                        arrays.append(raw)
                        samples.append(SimpleNamespace(Y_12lead=raw, anchor_i_ecg=raw[:1], target_quality_mask=np.ones(12, bool)))
                np.save(folder / 'target_uV.npy', np.stack(arrays))
                np.save(folder / 'prediction_uV.npy', np.stack(arrays) * 1.05)
                csv_write(folder / 'window_metadata.csv', rows)
                csv_write(folder / 'overall_metrics.csv', [{'evaluation_view': 'raw_uV', 'prediction_unit': 'uV'}])
                adapter[task] = SimpleNamespace(rows=rows, base=samples, manifest_digest='val-' + task)
            json_write(source_eval / 'inference_sampling.json', {'checkpoint': str(run / 'best.pt'), 'context_used': False,
                                                               'sampling': {'seed': 42, 'solver': 'heun', 'steps': 16, 'samples': 16}})
            train_rows = [{'target_record_id': 'training', 'subject_id': 'train-patient',
                           'start_sample_500hz': str(w * 5000)} for w in range(12)]
            # Strict ECGSample has subject ID in row/meta, not a .subject_id attribute.
            train_samples = [SimpleNamespace(Y_12lead=waves['a'][:, w * 5000:(w + 1) * 5000],
                                              target_quality_mask=np.ones(12, bool)) for w in range(12)]
            training = SimpleNamespace(rows=train_rows, base=train_samples, manifest_digest='train-manifest')
            demo = SimpleNamespace(get=lambda _: encode_demographics({'age': 40, 'gender': 'male'}), audit=lambda: {'subjects': 3})
            checkpoint = {'cache_protocol': 'independent-record-context-v1',
                          'evaluation_protocol': 'machine_body_equal_weight_score_v1', 'manifests': manifests}
            prepared = repo / 'prepared'
            with ExitStack() as stack:
                overrides = {'__file__': str(repo / 'baselines/B5_exp/prepare.py'),
                             'load_checkpoint': lambda _: checkpoint,
                             'common_config': lambda _: SimpleNamespace(path=lambda key: paths[key]),
                             'demographics_table': lambda _: demo, 'demographics_path': lambda _: paths['demographics_csv'],
                             'preprocessing_config': lambda _: None,
                             'checkpoint_preprocessor': lambda *_: SimpleNamespace(scale_uV_by_source={'d12': np.ones(12)}),
                             'HuaweiTrainDataset': lambda *_: training,
                             'HuaweiValidationDataset': lambda _, task, *__: adapter[task]}
                for key, value in overrides.items():
                    stack.enter_context(patch.object(preparation, key, value))
                preparation.prepare(run / 'best.pt', source_eval, prepared)
            check(prepared, repo / 'preflight.json')
            torch.manual_seed(42)
            model = SlowCNN()
            for parameter in model.parameters():
                parameter.data.zero_()
            for seconds in (10, 30):
                output = repo / f'S{seconds}'
                output.mkdir()
                config = {'context_seconds': seconds, 'prepared_sha256': sha(prepared / 'prepared.json'),
                          'initial_weights_sha256': 'same synthetic model', 'epochs': 50}
                json_write(output / 'resolved_config.json', config)
                torch.save({'protocol': PROTOCOL, 'config': config, 'epoch': 5, 'model': model.state_dict()}, output / 'best.pt')
                evaluate(SimpleNamespace(prepared=str(prepared), checkpoint=str(output / 'best.pt'),
                                         output=str(output / 'evaluation_best'), device='cpu', no_plots=seconds == 10))
                result = json.loads((output / 'evaluation_best/evaluation.json').read_text(encoding='utf-8'))
                self.assertFalse(result['target_used_at_inference'])
                self.assertEqual(result['scores'][0]['competition_score_protocol'], 'correlation50_bonus10_target_record_v2')
                if seconds == 30:
                    self.assertTrue(list((output / 'evaluation_best/plots').glob('*_V4.png')))
            compare(repo / 'S10/evaluation_best', repo / 'S30/evaluation_best', repo / 'comparison')
            self.assertTrue((repo / 'comparison/report.md').is_file())


if __name__ == '__main__':
    unittest.main()
