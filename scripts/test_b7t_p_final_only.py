"""Run with: python -m unittest scripts.test_b7t_p_final_only -v"""
from argparse import Namespace
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import b7t_p_final_only as tool
from baselines.B7.checkpoint import file_sha256, load_checkpoint, pack_rng, save_checkpoint
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.model import B7Transformer
from baselines.B7.reschedule import assert_same, cache_snapshot
from baselines.B7.tests.test_b7 import make_manifest, small_config
from baselines.B7.train import code_sha


class FinalOnlyTests(unittest.TestCase):
    def fixture(self, root, pending=True):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        source, cache_root = root / 'old', root / 'caches'
        source.mkdir()
        (source / '.pipeline.lock').write_bytes(b'0')
        caches = {}
        for stage, name in (('public10', 'public10'), ('public40', 'ptb40'), ('huawei40', 'huawei')):
            directory = cache_root / 'data' / name
            directory.mkdir(parents=True)
            caches[stage] = cache_snapshot(make_manifest(directory))
        configs = {}
        for stage in caches:
            configs[stage] = json.loads(json.dumps({
                'stage': stage, 'model': small_config().to_dict(), 'loss': asdict(LossConfig()),
                'sampling': asdict(SamplingConfig(window_seconds=10 if stage == 'public10' else 40)),
                'validation': {'batch_size': 32},
                'training': {'seed': 42, 'updates': 10000 if stage == 'public10' else 1000,
                    'validate_every': {'public10': 2000, 'public40': 1000, 'huawei40': 500}[stage],
                    'checkpoint_every': 500, 'batch_size': 32, 'learning_rate': 1e-4,
                    'weight_decay': .01, 'ema_decay': .999, 'gradient_clip': 1., 'boundary_probability': 0.}}))
        model = B7Transformer(small_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        state = model.state_dict()
        cache = caches['public10']
        identity = {'config': configs['public10'], 'code_sha256': code_sha(),
            'manifest_sha256': cache['manifest_sha256'], 'record_sha256': cache['record_sha256']}
        from baselines.B7 import ARCHITECTURE_ID, CHECKPOINT_VERSION, GENERATED_LEADS
        best = {'architecture_id': ARCHITECTURE_ID, 'format_version': CHECKPOINT_VERSION,
            'generated_leads': GENERATED_LEADS, 'model_config': model.config.to_dict(),
            'model': state, 'ema': state, 'optimizer': optimizer.state_dict(), 'rng': pack_rng(),
            'scales_uV': cache['scales_uV'], 'stage': 'public10', 'best_r': .7, 'update': 5500,
            'initialization': {'kind': 'random_B7T', 'B5_weights_used': False},
            'identity': identity, 'last_validated_update': 5500, 'pending_validation': False}
        last = dict(best, update=6000, last_validated_update=5500 if pending else 6000,
                    pending_validation=pending, pending_train_parts={'total': .2} if pending else None)
        p = source / 'P'
        p.mkdir()
        for name, value in (('last.pt', last), ('best_r.pt', best), ('best.pt', best)):
            save_checkpoint(p / name, value)
        (p / 'config.json').write_text(json.dumps(configs['public10']))
        (p / 'history.jsonl').write_text('{"update":5500,"validation":{"r":0.7}}\n')
        (source / 'migration.json').write_text(json.dumps({'kind': 'B7-T-sparse-validation-v1',
            'complete': True, 'target_code_sha256': code_sha(), 'configs': configs,
            'caches': caches, 'source': str(cache_root)}))
        return source, configs

    def test_explicit_skip_preserves_all_training_state_and_real_validation_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, configs = self.fixture(root)
            before = load_checkpoint(source / 'P/last.pt')
            snapshots = {str(p.relative_to(source)): (file_sha256(p), p.stat().st_mtime_ns)
                         for p in source.rglob('*') if p.is_file()}
            target = root / 'new'
            with patch.object(tool, 'project_configs', return_value=configs):
                report = tool.migrate(source, target, skip_pending=True)
            after = load_checkpoint(target / 'P/last.pt')
            for key in ('model', 'ema', 'optimizer', 'rng', 'initialization', 'update', 'best_r'):
                assert_same(before[key], after[key])
            self.assertFalse(after['pending_validation'])
            self.assertEqual(after['last_validated_update'], 5500)
            self.assertEqual(report['skipped_validation'], 6000)
            self.assertEqual(after['identity']['config']['training']['validate_every'], 10000)
            self.assertEqual(after['identity']['config']['training']['checkpoint_every'], 500)
            for stage in ('public40', 'huawei40'):
                self.assertEqual(report['configs'][stage], configs[stage])
            for name, snapshot in snapshots.items():
                p = source / name
                self.assertEqual((file_sha256(p), p.stat().st_mtime_ns), snapshot)

    def test_completed_6000_validation_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, configs = self.fixture(root, pending=False)
            with patch.object(tool, 'project_configs', return_value=configs):
                result = tool.migrate(source, root / 'new', skip_pending=True)
            after = load_checkpoint(root / 'new/P/last.pt')
            self.assertIsNone(result['skipped_validation'])
            self.assertEqual(after['last_validated_update'], 6000)
            self.assertFalse(after['pending_validation'])

    def test_pending_requires_explicit_flag_and_original_config_cannot_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, configs = self.fixture(root)
            with patch.object(tool, 'project_configs', return_value=configs):
                with self.assertRaisesRegex(ValueError, 'explicit --skip-pending'):
                    tool.migrate(source, root / 'new')
            changed = copy.deepcopy(configs)
            changed['public10']['training']['batch_size'] = 64
            with patch.object(tool, 'project_configs', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'Shared configs changed'):
                    tool.migrate(source, root / 'new', skip_pending=True)
            self.assertFalse((root / 'new').exists())

    def test_driver_uses_private_configs_correct_GPU_assignment_and_original_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, configs = self.fixture(root)
            target = root / 'new'
            with patch.object(tool, 'project_configs', return_value=configs):
                tool.migrate(source, target, skip_pending=True)
            args = Namespace(output=target, data_root=root / 'raw', metadata=root / 'metadata',
                ptb_root=root / 'ptb', gpu_p='0', gpu_a='1', gpu_b='1', max_train=2, dry_run=True)
            original_plan, original_fingerprint = tool.pipeline.plan, tool.pipeline.fingerprint
            def inspect(value):
                jobs = {job.name: job for job in tool.pipeline.plan(value)}
                self.assertEqual(set(jobs), {'P', 'A', 'Q', 'B'})
                self.assertEqual(jobs['P'].gpu, '0')
                for name in ('A', 'Q', 'B'):
                    self.assertEqual(jobs[name].gpu, '1')
                self.assertEqual(jobs['A'].dependencies, ('P',))
                self.assertEqual(jobs['Q'].dependencies, ('P',))
                self.assertEqual(jobs['B'].dependencies, ('Q',))
                for name, job in jobs.items():
                    cfg_path = job.arguments[job.arguments.index('--config') + 1]
                    self.assertEqual(Path(cfg_path).parent, target / 'configs')
                self.assertNotEqual(tool.pipeline.fingerprint(value), original_fingerprint(value))
                return 0
            with patch.object(tool.pipeline, 'run', side_effect=inspect):
                self.assertEqual(tool.run(args), 0)
            self.assertIs(tool.pipeline.plan, original_plan)
            self.assertIs(tool.pipeline.fingerprint, original_fingerprint)

    def test_incomplete_report_and_started_long_window_stage_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, configs = self.fixture(root)
            (source / 'Q').mkdir()
            with self.assertRaisesRegex(ValueError, 'A/Q/B not started'):
                tool.migrate(source, root / 'new', skip_pending=True)


if __name__ == '__main__':
    unittest.main()
