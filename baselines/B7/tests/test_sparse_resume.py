"""Sparse validation must not reset optimization or lose historical best weights."""
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

from baselines.B7 import reschedule, train as trainer
from baselines.B7.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.pipeline import Job, plan, validate_artifact
from baselines.B7.tests.test_b7 import make_manifest, small_config


class SparseResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.baseline_package = reschedule.package_at_revision(reschedule.BASELINE_REVISION)
        # The old migration whitelist intentionally rejects the compact branch.
        # Exercise its legacy migration tests against the approved schedule-only
        # package; compact experiment rejection is covered in test_compact.py.
        cls.approved_schedule_package = reschedule.package_at_revision('5cffb8e')

    def setUp(self):
        self.current_package = copy.deepcopy(self.approved_schedule_package)
        self.worktree_guard = patch.object(reschedule, 'working_package', side_effect=lambda: self.current_package)
        self.worktree_guard.start()
        self.addCleanup(self.worktree_guard.stop)
        self.code_guard = patch.object(reschedule, 'package_at_revision', side_effect=lambda revision:
            self.current_package if revision == 'HEAD' else self.baseline_package)
        self.code_guard.start()
        self.addCleanup(self.code_guard.stop)

    def fixture(self, root):
        source = root / 'old'
        manifests = {stage: make_manifest(self.mkdir(source / 'data' / cache))
                     for stage, cache in reschedule.CACHES.items()}
        (source / '.pipeline.lock').write_bytes(b'0')
        cfg = {'stage': 'public10', 'model': small_config().to_dict(), 'loss': asdict(LossConfig()),
               'sampling': asdict(SamplingConfig(window_seconds=10, samples=1, steps=1)),
               'training': {'seed': 4, 'updates': 4, 'validate_every': 2, 'batch_size': 1,
                            'learning_rate': .0001, 'weight_decay': .01, 'ema_decay': .9,
                            'gradient_clip': 1., 'boundary_probability': 0.}}
        # JSON round trip matches a real identity/config loaded by the server.
        cfg = json.loads(json.dumps(cfg))
        cp = root / 'legacy.json'
        cp.write_text(json.dumps(cfg))
        args = Namespace(config=str(cp), manifest=str(manifests['public10']), output=str(source / 'P'),
                         device='cpu', resume=False, init_checkpoint=None)
        self.interrupt_at_third_update(args, legacy=True)
        last = load_checkpoint(source / 'P/last.pt')
        for key in ('last_validated_update', 'pending_validation', 'pending_train_parts'):
            last.pop(key, None)  # Legacy807b8b0 has none of these fields.
        save_checkpoint(source / 'P/last.pt', last)
        configs = {}
        for stage in reschedule.INTERVALS:
            value = copy.deepcopy(cfg)
            value['stage'] = stage
            value['sampling']['window_seconds'] = 10 if stage == 'public10' else 40
            value['training'].update(validate_every=4, checkpoint_every=1)
            configs[stage] = value
        return source, args, configs

    @staticmethod
    def mkdir(path):
        path.mkdir(parents=True)
        return path

    def interrupt_at_third_update(self, args, legacy=False):
        original = trainer.balanced_indices
        calls = 0
        def stop(*a, **kw):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('stop after two updates')
            return original(*a, **kw)
        with patch.object(trainer, 'balanced_indices', stop):
            if legacy:
                with patch.object(trainer, 'code_sha', return_value=reschedule.BASELINE_CODE_SHA):
                    with self.assertRaisesRegex(RuntimeError, 'two updates'):
                        trainer.train(args)
            else:
                with self.assertRaisesRegex(RuntimeError, 'two updates'):
                    trainer.train(args)

    def migrate(self, source, target, configs, **kwargs):
        with patch.object(reschedule, 'stage_configs', return_value=configs):
            return reschedule.migrate(source, target, **kwargs)

    def resume_args(self, args, target):
        return Namespace(**(vars(args) | {'output': str(target / 'P'),
            'config': str(target / 'P/config.json'), 'resume': True}))

    def test_production_schedules_and_baseline_configs(self):
        configs = reschedule.stage_configs()
        for stage, interval in reschedule.INTERVALS.items():
            cfg = configs[stage]
            self.assertEqual(cfg['training']['validate_every'], interval)
            self.assertEqual(cfg['training']['checkpoint_every'], 500)
            self.assertEqual(cfg['training']['batch_size'], 32)
            self.assertEqual(reschedule.without_schedule(cfg),
                             reschedule.without_schedule(reschedule.baseline_config(stage)))

    def test_code_guard_rejects_model_sampling_or_dirty_code(self):
        reschedule.verify_code_upgrade()
        with patch.object(reschedule, 'working_package', side_effect=lambda: self.current_package):
            for name in ('model.py', 'flow.py', 'evaluate.py', 'data.py'):
                self.current_package[name] += b'\n# changed\n'
                with self.assertRaisesRegex(ValueError, 'Model/data/loss/sampling/scoring'):
                    reschedule.verify_code_upgrade()
                self.current_package[name] = self.baseline_package[name]
        with patch.object(reschedule, 'package_at_revision', side_effect=lambda revision:
                          {} if revision == 'HEAD' else self.baseline_package):
            with self.assertRaisesRegex(ValueError, 'committed tested'):
                reschedule.verify_code_upgrade()

    def test_config_guard_rejects_non_schedule_changes(self):
        cfg = reschedule.stage_configs()['public10']
        for field in ('learning_rate', 'batch_size', 'updates'):
            changed = copy.deepcopy(cfg)
            changed['training'][field] *= 2
            with patch.object(reschedule, 'read_json', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'only validation/checkpoint'):
                    reschedule.stage_configs()

    def test_migration_preserves_every_state_and_original_files_then_exact_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, args, configs = self.fixture(root)
            original_files = {str(p.relative_to(source)): (file_sha256(p), p.stat().st_mtime_ns)
                              for p in source.rglob('*') if p.is_file()}
            before = load_checkpoint(source / 'P/last.pt')
            best_before = load_checkpoint(source / 'P/best_r.pt')
            target = root / 'new'
            self.migrate(source, target, configs)
            copied = load_checkpoint(target / 'P/last.pt')
            for name in ('model', 'ema', 'optimizer', 'rng', 'initialization', 'update', 'best_r'):
                reschedule.assert_same(before[name], copied[name])
            reschedule.assert_same(best_before['ema'], load_checkpoint(target / 'P/best_r.pt')['ema'])
            for name, snapshot in original_files.items():
                p = source / name
                self.assertEqual((file_sha256(p), p.stat().st_mtime_ns), snapshot, name)
            self.assertEqual(copied['last_validated_update'], 2)
            self.assertEqual(copied['identity']['code_sha256'], trainer.code_sha())
            with patch.object(reschedule, 'stage_configs', return_value=configs):
                reschedule.validate_migration(target, source)
            trainer.train(self.resume_args(args, target))
            # Compare against an uninterrupted run using the NEW schedule.
            new_cp = root / 'reference_config.json'
            new_cp.write_text(json.dumps(configs['public10']))
            reference = Namespace(**(vars(args) | {'config': str(new_cp), 'output': str(root / 'reference')}))
            trainer.train(reference)
            actual, expected = load_checkpoint(target / 'P/last.pt'), load_checkpoint(root / 'reference/last.pt')
            for name in ('model', 'ema', 'optimizer', 'rng'):
                reschedule.assert_same(actual[name], expected[name])
            history = [json.loads(line) for line in (target / 'P/history.jsonl').read_text().splitlines()]
            self.assertEqual([row['update'] for row in history], [0, 2, 4])

    def test_periodic_checkpoint_without_validation_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, args, configs = self.fixture(root)
            target = root / 'new'
            self.migrate(source, target, configs)
            resumed = self.resume_args(args, target)
            original = trainer.balanced_indices
            calls = 0
            def stop(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('between validations')
                return original(*a, **kw)
            with patch.object(trainer, 'balanced_indices', stop), patch.object(trainer, 'evaluate') as evaluator:
                with self.assertRaisesRegex(RuntimeError, 'between validations'):
                    trainer.train(resumed)
                evaluator.assert_not_called()
            last = load_checkpoint(target / 'P/last.pt')
            self.assertEqual(last['update'], 3)
            self.assertEqual(last['last_validated_update'], 2)
            self.assertFalse(last['pending_validation'])
            with patch.object(trainer, 'evaluate', wraps=trainer.evaluate) as evaluator:
                trainer.train(resumed)
                self.assertEqual(evaluator.call_count, 1)

    def test_interrupted_final_validation_retries_without_another_optimizer_update(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, args, configs = self.fixture(root)
            target = root / 'new'
            self.migrate(source, target, configs)
            resumed = self.resume_args(args, target)
            with patch.object(trainer, 'evaluate', side_effect=RuntimeError('validation interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'validation interrupted'):
                    trainer.train(resumed)
            last = load_checkpoint(target / 'P/last.pt')
            self.assertEqual(last['update'], 4)
            self.assertTrue(last['pending_validation'])
            self.assertFalse((target / 'P/completed.json').exists())
            with patch.object(trainer, 'flow_loss') as loss, \
                 patch.object(trainer, 'evaluate', wraps=trainer.evaluate) as evaluator:
                trainer.train(resumed)
                loss.assert_not_called()
                self.assertEqual(evaluator.call_count, 1)
            last = load_checkpoint(target / 'P/last.pt')
            self.assertFalse(last['pending_validation'])
            self.assertEqual(last['last_validated_update'], 4)
            self.assertTrue((target / 'P/completed.json').exists())

    def test_historical_best_kept_if_new_validation_does_not_win_and_completion_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, args, configs = self.fixture(root)
            target = root / 'new'
            self.migrate(source, target, configs)
            best_sha = file_sha256(target / 'P/best_r.pt')
            with patch.object(trainer, 'evaluate', return_value=({'r': -1.}, [])):
                trainer.train(self.resume_args(args, target))
            self.assertEqual(file_sha256(target / 'P/best_r.pt'), best_sha)
            job = Job('P', (), 'train', ['--config', str(target / 'P/config.json'), '--manifest', args.manifest],
                      target / 'P/completed.json', '0')
            result = validate_artifact(job)
            self.assertEqual(result['best_r'], load_checkpoint(source / 'P/best_r.pt')['best_r'])

    def test_interruption_after_best_write_is_recovered_from_pending_last(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, args, configs = self.fixture(root)
            target = root / 'new'
            self.migrate(source, target, configs)
            resumed = self.resume_args(args, target)
            original = trainer.save_checkpoint
            def stop(path, value):
                if Path(path).name == 'last.pt' and value['update'] == 4 and not value['pending_validation']:
                    raise RuntimeError('after best write')
                return original(path, value)
            with patch.object(trainer, 'save_checkpoint', stop):
                with self.assertRaisesRegex(RuntimeError, 'after best write'):
                    trainer.train(resumed)
            last, best = load_checkpoint(target / 'P/last.pt'), load_checkpoint(target / 'P/best_r.pt')
            self.assertTrue(last['pending_validation'])
            self.assertGreater(best['best_r'], last['best_r'])
            trainer.train(resumed)
            last = load_checkpoint(target / 'P/last.pt')
            self.assertEqual(last['best_r'], load_checkpoint(target / 'P/best_r.pt')['best_r'])
            self.assertFalse(last['pending_validation'])

    def test_completed_P_and_running_Q_are_both_migrated_without_restarting_P(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, args, configs = self.fixture(root)
            with patch.object(trainer, 'code_sha', return_value=reschedule.BASELINE_CODE_SHA):
                trainer.train(Namespace(**(vars(args) | {'resume': True})))
            # PTB long records are independently prepared; do not concatenate
            # records in production. This synthetic fixture only tests resumption.
            for p in (source / 'data/ptb40').glob('*.npz'):
                with np.load(p) as value:
                    target = np.tile(value['target_uV'], (1, 4))
                    mask = value['lead_mask'].copy()
                np.savez(p, anchor_uV=target[:1], target_uV=target, lead_mask=mask)
            q_cfg = json.loads((source / 'P/config.json').read_text())
            q_cfg['stage'] = 'public40'
            q_cfg['sampling']['window_seconds'] = 40
            q_cp = root / 'q_config.json'
            q_cp.write_text(json.dumps(q_cfg))
            q_args = Namespace(config=str(q_cp), manifest=str(source / 'data/ptb40/manifest.json'),
                output=str(source / 'Q'), device='cpu', resume=False,
                init_checkpoint=str(source / 'P/best_r.pt'))
            self.interrupt_at_third_update(q_args, legacy=True)
            before = load_checkpoint(source / 'Q/last.pt')
            target = root / 'new'
            self.migrate(source, target, configs)
            copied = load_checkpoint(target / 'Q/last.pt')
            for key in ('model', 'ema', 'optimizer', 'rng', 'initialization', 'update', 'best_r'):
                reschedule.assert_same(before[key], copied[key])
            p_job = Job('P', (), 'train', ['--config', str(target / 'P/config.json'), '--manifest', args.manifest],
                        target / 'P/completed.json', '0')
            self.assertEqual(validate_artifact(p_job)['updates'], 4)
            trainer.train(Namespace(**(vars(q_args) | {'config': str(target / 'Q/config.json'),
                'output': str(target / 'Q'), 'resume': True, 'init_checkpoint': None})))
            self.assertTrue((target / 'Q/completed.json').exists())

    def test_check_only_is_readonly_and_repeated_output_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, args, configs = self.fixture(root)
            target = root / 'new'
            self.migrate(source, target, configs, check_only=True)
            self.assertFalse(target.exists())
            self.migrate(source, target, configs)
            with self.assertRaisesRegex(ValueError, 'NEW output'):
                self.migrate(source, target, configs)

    def test_rejects_active_controller_unknown_code_changed_data_and_bad_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, args, configs = self.fixture(root)
            target = root / 'new'
            with patch.object(reschedule, 'assert_stopped', side_effect=RuntimeError('still running')):
                with self.assertRaisesRegex(RuntimeError, 'still running'):
                    self.migrate(source, target, configs)
            original = load_checkpoint(source / 'P/last.pt')
            changed = copy.deepcopy(original)
            changed['identity']['code_sha256'] = 'unknown'
            save_checkpoint(source / 'P/last.pt', changed)
            with self.assertRaisesRegex(ValueError, 'unknown source'):
                self.migrate(source, target, configs)
            save_checkpoint(source / 'P/last.pt', original)
            data = source / 'data/public10/train.npz'
            original_bytes = data.read_bytes()
            data.write_bytes(original_bytes + b'changed')
            with self.assertRaisesRegex(ValueError, 'data/manifest/scales changed'):
                self.migrate(source, target, configs)
            data.write_bytes(original_bytes)
            best = load_checkpoint(source / 'P/best_r.pt')
            best['best_r'] += .1
            save_checkpoint(source / 'P/best_r.pt', best)
            with self.assertRaisesRegex(ValueError, 'Best and last'):
                self.migrate(source, target, configs)
            self.assertFalse(target.exists())

    def test_migration_manifest_rejects_changed_cache_source_config_or_incomplete_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, args, configs = self.fixture(root)
            target = root / 'new'
            self.migrate(source, target, configs)
            with patch.object(reschedule, 'stage_configs', return_value=configs):
                for other in (None, root / 'wrong'):
                    with self.assertRaisesRegex(ValueError, 'Incomplete migration'):
                        reschedule.validate_migration(target, other)
                path = target / 'migration.json'
                record = json.loads(path.read_text())
                record['complete'] = False
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, 'Incomplete migration'):
                    reschedule.validate_migration(target, source)
                record['complete'] = True
                path.write_text(json.dumps(record))
                data = source / 'data/ptb40/train.npz'
                data.write_bytes(data.read_bytes() + b'changed')
                with self.assertRaisesRegex(ValueError, 'cache/manifest changed'):
                    reschedule.validate_migration(target, source)

    def test_reuse_plan_preserves_P_A_Q_B_dependencies_without_preparation(self):
        root = Path('/example')
        args = Namespace(output=root / 'new', data_root=root / 'raw', metadata=root / 'metadata',
                         ptb_root=root / 'ptb', reuse_data_from=root / 'old', gpu_p='0', gpu_a='0', gpu_b='0')
        jobs = {job.name: job for job in plan(args)}
        self.assertEqual(set(jobs), {'P', 'A', 'Q', 'B'})
        self.assertEqual(jobs['P'].dependencies, ())
        self.assertEqual(jobs['A'].dependencies, ('P',))
        self.assertEqual(jobs['Q'].dependencies, ('P',))
        self.assertEqual(jobs['B'].dependencies, ('Q',))
        for job in jobs.values():
            path = job.arguments[job.arguments.index('--manifest') + 1]
            self.assertIn(str((root / 'old/data').resolve()), path)

    def test_invalid_checkpoint_interval_rejected(self):
        cfg = reschedule.stage_configs()['public10']
        for value in (0, -1, True, 1.5):
            changed = copy.deepcopy(cfg)
            changed['training']['checkpoint_every'] = value
            with self.assertRaisesRegex(ValueError, 'checkpoint_every'):
                trainer.validate_settings(changed)


if __name__ == '__main__':
    unittest.main()
