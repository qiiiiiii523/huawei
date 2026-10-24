"""Independent recipes, final-only stage validation, restart state and probe behavior."""
from argparse import Namespace
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from baselines.B7 import compact, pipeline, reschedule, train as trainer
from baselines.B7.checkpoint import load_checkpoint, save_checkpoint
from baselines.B7.compact_probe import measure
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.model import B7Transformer
from baselines.B7.tests.test_b7 import make_manifest, small_config


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root, *, updates=4):
        manifest = make_manifest(root)
        cfg = {'stage': 'public10', 'model': small_config().to_dict(), 'loss': asdict(LossConfig()),
               'sampling': asdict(SamplingConfig(window_seconds=10, steps=1, samples=1)),
               'validation': {'batch_size': 2, 'initial': False},
               'training': {'seed': 7, 'updates': updates, 'validate_every': 4, 'checkpoint_every': 1,
                            'batch_size': 1, 'learning_rate': .0001, 'weight_decay': .01,
                            'ema_decay': .9, 'gradient_clip': 1., 'boundary_probability': 0.}}
        cp = root / 'config.json'
        cp.write_text(json.dumps(cfg))
        args = Namespace(config=str(cp), manifest=str(manifest), output=str(root / 'run'),
                         device='cpu', resume=False, init_checkpoint=None)
        return cfg, args

    def test_six_recipes_and_exact_parameter_counts(self):
        for name, spec in compact.PRESETS.items():
            configs = compact.load_configs(name)
            model = B7Transformer(spec['model'])
            self.assertEqual(sum(p.numel() for p in model.parameters()), spec['parameters'])
            del model
            for stage, cfg in configs.items():
                self.assertEqual(cfg['training']['checkpoint_every'], 500)
                self.assertFalse(cfg['validation']['initial'])
                self.assertEqual(cfg['training']['batch_size'], 64 if stage == 'public10' else 32)
                self.assertEqual(cfg['training']['updates'], 10000 if stage == 'public10' else 1000)
                self.assertEqual(cfg['training']['validate_every'],
                                 {'public10': 10000, 'public40': 1000, 'huawei40': 1000}[stage])
                self.assertEqual(cfg['loss'], asdict(LossConfig()))
                self.assertEqual(cfg['sampling']['samples'], 16)
                self.assertEqual(cfg['sampling']['steps'], 16)

    def test_existing_baseline_configs_are_unchanged(self):
        for stage in compact.STAGES:
            relative = f'configs/b7_{stage}.json'
            original = reschedule.git('show', f'5cffb8e:{relative}').replace(b'\r\n', b'\n')
            self.assertEqual((compact.ROOT / relative).read_bytes().replace(b'\r\n', b'\n'), original)

    def test_recipe_rejects_accidental_batch_loss_or_model_changes(self):
        cfg = compact.load_configs('g6_19m')['public10']
        for change in ('batch', 'loss', 'model'):
            altered = copy.deepcopy(cfg)
            if change == 'batch':
                altered['training']['batch_size'] = 32
            elif change == 'loss':
                altered['loss']['pcc_weight'] = .2
            else:
                altered['model']['depths'] = [2, 2, 4]
            with patch.object(compact, 'read_json', return_value=altered):
                with self.assertRaisesRegex(ValueError, 'Unexpected'):
                    compact.load_configs('g6_19m')

    def test_P_evaluates_only_final_update_and_preserves_training_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, args = self.fixture(root)
            original = trainer.evaluate
            evaluated_states = []
            def capture_ema(model, *a, **kw):
                evaluated_states.append(copy.deepcopy(model.state_dict()))
                return original(model, *a, **kw)
            with patch.object(trainer, 'evaluate', side_effect=capture_ema) as evaluator:
                trainer.train(args)
                self.assertEqual(evaluator.call_count, 1)
            history = [json.loads(line) for line in (root / 'run/history.jsonl').read_text().splitlines()]
            self.assertEqual([row['update'] for row in history], [4])
            last = load_checkpoint(root / 'run/last.pt')
            reschedule.assert_same(evaluated_states[0], last['ema'])
            self.assertTrue(any(not torch.equal(last['model'][k], last['ema'][k]) for k in last['model']))
            self.assertEqual(last['last_validated_update'], 4)
            self.assertFalse(last['pending_validation'])
            self.assertEqual(last['initialization']['kind'], 'random_B7T')
            self.assertFalse(last['initialization']['B5_weights_used'])
            self.assertFalse(last['initialization']['B7_U7_weights_used'])
            self.assertTrue((root / 'run/best_r.pt').exists())
            checkpoints = [json.loads(line) for line in (root / 'run/checkpoint_history.jsonl').read_text().splitlines()]
            self.assertEqual([row['update'] for row in checkpoints], [0, 1, 2, 3, 4])
            job = pipeline.Job('P', (), 'train', ['--config', args.config, '--manifest', args.manifest],
                               root / 'run/completed.json', '0')
            pipeline.validate_artifact(job)
            stale = load_checkpoint(root / 'run/best_r.pt')
            stale['update'] = 3
            save_checkpoint(root / 'run/best_r.pt', stale)
            with self.assertRaisesRegex(RuntimeError, 'final EMA checkpoint'):
                pipeline.validate_artifact(job)

    def test_final_only_checkpoint_can_resume_before_any_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, args = self.fixture(root)
            original = trainer.balanced_indices
            calls = 0
            def stop(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError('pause before first validation')
                return original(*a, **kw)
            with patch.object(trainer, 'balanced_indices', stop), patch.object(trainer, 'evaluate') as evaluator:
                with self.assertRaisesRegex(RuntimeError, 'pause before first'):
                    trainer.train(args)
                evaluator.assert_not_called()
            last = load_checkpoint(root / 'run/last.pt')
            self.assertEqual(last['update'], 2)
            self.assertEqual(last['last_validated_update'], -1)
            self.assertEqual(last['best_r'], -float('inf'))
            self.assertFalse(last['pending_validation'])
            self.assertFalse((root / 'run/best_r.pt').exists())
            args.resume = True
            with patch.object(trainer, 'evaluate', wraps=trainer.evaluate) as evaluator:
                trainer.train(args)
                self.assertEqual(evaluator.call_count, 1)
            actual = load_checkpoint(root / 'run/last.pt')
            args.output = str(root / 'reference')
            args.resume = False
            trainer.train(args)
            expected = load_checkpoint(root / 'reference/last.pt')
            for field in ('model', 'ema', 'optimizer', 'rng'):
                reschedule.assert_same(actual[field], expected[field])

    def test_interrupted_final_only_validation_is_retried_before_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, args = self.fixture(root)
            with patch.object(trainer, 'evaluate', side_effect=RuntimeError('final evaluation interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'final evaluation interrupted'):
                    trainer.train(args)
            last = load_checkpoint(root / 'run/last.pt')
            self.assertEqual(last['update'], 4)
            self.assertTrue(last['pending_validation'])
            self.assertEqual(last['last_validated_update'], -1)
            self.assertFalse((root / 'run/completed.json').exists())
            args.resume = True
            with patch.object(trainer, 'flow_loss') as loss, \
                 patch.object(trainer, 'evaluate', wraps=trainer.evaluate) as evaluator:
                trainer.train(args)
                loss.assert_not_called()
                self.assertEqual(evaluator.call_count, 1)
            self.assertTrue((root / 'run/best_r.pt').exists())
            self.assertTrue((root / 'run/completed.json').exists())

    def test_stage_end_is_evaluated_even_if_not_interval_multiple(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, args = self.fixture(root, updates=3)
            with patch.object(trainer, 'evaluate', wraps=trainer.evaluate) as evaluator:
                trainer.train(args)
                self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(load_checkpoint(root / 'run/best_r.pt')['update'], 3)

    def test_initial_validation_flag_is_strict_boolean_and_defaults_to_true(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _ = self.fixture(Path(temp))
            for value in (0, 1, 'false', None):
                altered = copy.deepcopy(cfg)
                altered['validation']['initial'] = value
                with self.assertRaisesRegex(ValueError, 'boolean initial'):
                    trainer.validate_settings(altered)
            cfg['validation'].pop('initial')
            trainer.validate_settings(cfg)

    def test_dedicated_entry_selects_independent_output_and_serial_default(self):
        for preset in compact.PRESETS:
            argv = ['compact', '--preset', preset, '--data-root', '/raw', '--metadata', '/metadata',
                    '--ptb-root', '/ptb', '--reuse-data-from', '/old', '--dry-run']
            with patch.object(sys, 'argv', argv), patch.object(pipeline, 'run', return_value=0) as run:
                with self.assertRaises(SystemExit) as status:
                    compact.main()
                self.assertEqual(status.exception.code, 0)
                args = run.call_args.args[0]
                self.assertEqual(args.max_train, 1)
                self.assertEqual(args.config_dir, compact.config_directory(preset))
                self.assertIn(preset, str(args.output))

    def test_plan_uses_only_selected_configs_and_keeps_both_routes(self):
        for preset in compact.PRESETS:
            args = Namespace(output=Path('/new') / preset, data_root=Path('/raw'), metadata=Path('/metadata'),
                ptb_root=Path('/ptb'), gpu_p='0', gpu_a='0', gpu_b='0', reuse_data_from=Path('/old'),
                config_dir=compact.config_directory(preset))
            jobs = {job.name: job for job in pipeline.plan(args)}
            self.assertEqual(set(jobs), {'P', 'A', 'Q', 'B'})
            self.assertEqual(jobs['A'].dependencies, ('P',))
            self.assertEqual(jobs['Q'].dependencies, ('P',))
            self.assertEqual(jobs['B'].dependencies, ('Q',))
            for job in jobs.values():
                config_path = Path(job.arguments[job.arguments.index('--config') + 1])
                self.assertEqual(config_path.parent, compact.config_directory(preset))
                cfg = json.loads(config_path.read_text())
                self.assertEqual(cfg['model']['depths'], list(compact.PRESETS[preset]['model'].depths))
            self.assertNotIn('--init-checkpoint', jobs['P'].arguments)

    def test_fingerprint_tracks_selected_config_directory_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected, other = root / 'selected', root / 'other'
            selected.mkdir()
            other.mkdir()
            (selected / 'b7_public10.json').write_text('{}')
            path = other / 'b7_public10.json'
            path.write_text('{}')
            args = Namespace(config_dir=selected, data_root=root / 'raw', metadata=root / 'metadata',
                             ptb_root=root / 'ptb', output=root / 'output')
            before = pipeline.fingerprint(args)
            path.write_text('{"other": true}')
            self.assertEqual(pipeline.fingerprint(args), before)
            (selected / 'b7_public10.json').write_text('{"selected": true}')
            self.assertNotEqual(pipeline.fingerprint(args), before)

    def test_legacy_migration_guard_rejects_compact_code(self):
        original = reschedule.package_at_revision(reschedule.BASELINE_REVISION)
        current = reschedule.working_package()
        with patch.object(reschedule, 'package_at_revision', side_effect=lambda revision:
                          current if revision == 'HEAD' else original):
            with self.assertRaisesRegex(ValueError, 'Model/data/loss/sampling/scoring'):
                reschedule.verify_code_upgrade()

    def test_probe_CPU_smoke_uses_real_training_and_does_not_claim_GPU_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _ = self.fixture(Path(temp))
            result = measure(cfg, batch_size=1, device='cpu', warmup=1, iterations=1)
            self.assertEqual(result['status'], 'cpu_only')
            self.assertIsNone(result['within_memory_budget'])
            self.assertIsNone(result['peak_reserved_gib'])
            self.assertTrue(result['anchor_unchanged'])
            self.assertTrue(result['finite_parameters'])
            self.assertTrue(result['finite_clipped_gradients'])
            self.assertGreater(result['windows_per_second'], 0)

    def test_probe_rejects_invalid_settings_before_allocating(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _ = self.fixture(Path(temp))
            for kwargs in ({'batch_size': 0}, {'batch_size': 1, 'warmup': 0},
                           {'batch_size': 1, 'iterations': True}, {'batch_size': 1, 'memory_budget_gib': -1}):
                with self.assertRaises(ValueError):
                    measure(cfg, device='cpu', **kwargs)

    def shell_fixture(self, root):
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path(shutil.which('bash') or '/missing')
        if not bash.is_file():
            self.skipTest('Bash unavailable for launcher control-flow test')
        def shell_path(p):
            value = p.resolve().as_posix()
            return '/' + value[0].lower() + value[2:] if os.name == 'nt' else value
        cache, output = root / 'cache', root / 'output'
        for name in ('public10', 'ptb40', 'huawei'):
            directory = cache / 'data' / name
            directory.mkdir(parents=True)
            (directory / 'manifest.json').write_text('{}')
        fake = root / 'fake_python.sh'
        fake.write_text('#!/usr/bin/env bash\n'
            'printf "%s\\n" "$*" >> "$B7T_TEST_CALLS"\n'
            'if [[ "$*" == *compact_probe* && "${B7T_TEST_FAIL_PROBE:-0}" == 1 ]]; then exit 2; fi\n'
            'exit 0\n')
        fake.chmod(0o755)
        calls = root / 'calls.log'
        env = dict(os.environ, B7T_PYTHON=shell_path(fake), B7T_CACHE_RUN=shell_path(cache),
                   B7T_OUTPUT=shell_path(output), B7T_TEST_CALLS=shell_path(calls), B7T_PROBE_FIRST='1')
        command = [str(bash), '--noprofile', '--norc', 'scripts/run_b7_t_compact.sh', 'g6_19m']
        return command, env, calls, output

    def test_shell_launcher_probes_before_training_and_skips_probe_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            command, env, calls, output = self.shell_fixture(Path(temp))
            process = subprocess.run(command, cwd=compact.ROOT, env=env, capture_output=True)
            self.assertEqual(process.returncode, 0, process.stderr.decode(errors='replace'))
            lines = calls.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn('compact_probe', lines[0])
            self.assertIn('--window-seconds 10', lines[0])
            self.assertIn('--window-seconds 40', lines[1])
            self.assertIn('--reuse-data-from', lines[2])
            self.assertIn('--max-train 1', lines[2])
            (output / 'P').mkdir()
            (output / 'P/last.pt').write_bytes(b'fixture')
            calls.write_text('')
            process = subprocess.run(command, cwd=compact.ROOT, env=env, capture_output=True)
            self.assertEqual(process.returncode, 0)
            lines = calls.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertNotIn('compact_probe', lines[0])

    def test_shell_launcher_does_not_train_after_failed_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            command, env, calls, output = self.shell_fixture(Path(temp))
            env['B7T_TEST_FAIL_PROBE'] = '1'
            process = subprocess.run(command, cwd=compact.ROOT, env=env, capture_output=True)
            self.assertNotEqual(process.returncode, 0)
            lines = calls.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn('compact_probe', lines[0])
            self.assertFalse((output / 'P').exists())


if __name__ == '__main__':
    unittest.main()
