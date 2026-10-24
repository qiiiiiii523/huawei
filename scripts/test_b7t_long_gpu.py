from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from scripts import b7t_long_gpu as tool


class LongGpuTests(unittest.TestCase):
    def args(self, root):
        return tool.setup(Namespace(output=root / 'run', gpu_p='1', gpu_long='0',
            data_root=root / 'raw', metadata=root / 'metadata', ptb_root=root / 'ptb',
            reuse_data_from=root / 'cache', controller_pid=12345))

    def fixture(self, root):
        args = self.args(root)
        args.output.mkdir()
        before = tool.specification(args, tool.jobs_for(args, changed=False))
        tool.atomic_json(args.output / 'pipeline_spec.json', before)
        return args, before

    def test_only_long_GPU_changes_in_all_eight_jobs(self):
        args = self.args(Path('/example'))
        before = tool.specification(args, tool.jobs_for(args, changed=False))
        after = tool.specification(args, tool.jobs_for(args, changed=True))
        tool.validate_change(before, after)
        for old, new in zip(before['experiments'], after['experiments']):
            self.assertEqual(old['gpu'], '1')
            self.assertEqual(new['gpu'], '1' if new['name'].endswith(':P') else '0')
            self.assertEqual(old['config'], new['config'])
            self.assertEqual(old['arguments'], new['arguments'])

    def test_migration_backs_up_identity_and_starts_with_one_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            args, before = self.fixture(Path(temp))
            with patch.object(tool, 'verify_completed_P') as verify:
                tool.migrate(args)
                verify.assert_called_once_with(args)
            after = tool.read(args.output / 'pipeline_spec.json')
            self.assertEqual(tool.read(args.output / tool.LIMIT), {'max_train': 1})
            tool.verify_audit(args, after)
            self.assertEqual(tool.read(tool.read(args.output / tool.AUDIT)['backup']), before)
            tool.migrate(args)
            self.assertEqual(tool.read(args.output / 'pipeline_spec.json'), after)

    def test_config_or_identity_changes_are_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            args, before = self.fixture(Path(temp))
            before['experiments'][0]['config']['training']['batch_size'] = 999
            tool.atomic_json(args.output / 'pipeline_spec.json', before)
            with self.assertRaisesRegex(ValueError, 'migration refused'):
                tool.migrate(args)
            self.assertEqual(tool.read(args.output / 'pipeline_spec.json'), before)
            self.assertFalse((args.output / tool.AUDIT).exists())

    def test_validation_failure_keeps_old_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            args, before = self.fixture(Path(temp))
            with patch.object(tool, 'verify_completed_P', side_effect=ValueError('P validation incomplete')):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    tool.migrate(args)
            self.assertEqual(tool.read(args.output / 'pipeline_spec.json'), before)

    def test_active_controller_lock_prevents_migration(self):
        with tempfile.TemporaryDirectory() as temp:
            args, before = self.fixture(Path(temp))
            lock = tool.pipeline.acquire_lock(args.output / '.compact_suite.lock')
            try:
                with self.assertRaises(OSError):
                    tool.migrate(args)
            finally:
                lock.close()
            self.assertEqual(tool.read(args.output / 'pipeline_spec.json'), before)

    def test_live_limit_changes_without_checkpoint_or_plan_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            args, before = self.fixture(Path(temp))
            live = tool.LiveArgs(args)
            for value in (1, 2, 1):
                tool.atomic_json(args.output / tool.LIMIT, {'max_train': value})
                self.assertEqual(live.max_train, value)
                self.assertEqual(tool.read(args.output / 'pipeline_spec.json'), before)
            for value in (0, 3, True, '2'):
                tool.atomic_json(args.output / tool.LIMIT, {'max_train': value})
                with self.assertRaises(ValueError):
                    _ = live.max_train

    def test_new_driver_passes_changed_plan_and_restores_function(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            original = tool.compact_suite.suite_jobs
            with patch.object(tool, 'verify_completed_P'):
                tool.migrate(args)
            def fake_run(live):
                self.assertEqual(live.max_train, 1)
                jobs = tool.compact_suite.suite_jobs(live)
                self.assertEqual([j.gpu for j in jobs[:2]], ['1', '1'])
                self.assertTrue(all(j.gpu == '0' for j in jobs[2:]))
                return 0
            with patch.object(tool.compact_suite, 'run', side_effect=fake_run):
                self.assertEqual(tool.run(args), 0)
            self.assertIs(tool.compact_suite.suite_jobs, original)

    def test_controller_check_rejects_wrong_owner_running_or_wrong_output(self):
        args = self.args(Path('/example'))
        good = {'uid': 1000, 'cwd': tool.ROOT, 'state': 'T',
                'command': ['python', '-m', 'baselines.B7.compact_suite', '--output', str(args.output)]}
        with patch.object(tool.os, 'getuid', return_value=1000, create=True):
            tool.check_controller(good, args)
            for key, value in [('uid', 1001), ('state', 'R'), ('cwd', Path('/other'))]:
                altered = dict(good, **{key: value})
                with self.assertRaises(ValueError):
                    tool.check_controller(altered, args)

    def test_switch_waits_for_P_children_before_retiring_only_idle_controller(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            for job in tool.jobs_for(args, changed=False):
                if job.name.endswith(':P'):
                    job.artifact.parent.mkdir(parents=True)
                    tool.atomic_json(job.artifact, {'completed': True})
            killed = []
            def process(pid):
                return {'uid': 1000, 'cwd': tool.ROOT, 'state': 'Z' if killed else 'T', 'start': '10',
                    'command': ['python', '-m', 'baselines.B7.compact_suite', '--output', str(args.output)]}
            def sleep(seconds):
                self.assertFalse(killed, 'active P child must not be interrupted')
            with patch.object(tool, 'require_live_switch'), \
                 patch.object(tool, 'process', side_effect=process), \
                 patch.object(tool, 'active_children', side_effect=[[777], [], []]), \
                 patch.object(tool.os, 'getuid', return_value=1000, create=True), \
                 patch.object(tool.os, 'pidfd_open', return_value=99, create=True), \
                 patch.object(tool.signal, 'SIGKILL', 9, create=True), \
                 patch.object(tool.signal, 'pidfd_send_signal', side_effect=lambda *a: killed.append(a), create=True), \
                 patch.object(tool.os, 'close') as close, \
                 patch.object(tool.time, 'sleep', side_effect=sleep) as waited, \
                 patch.object(tool, 'verify_completed_P'), \
                 patch.object(tool, 'run', return_value=0):
                self.assertEqual(tool.switch(args), 0)
                waited.assert_called_once_with(15)
                self.assertEqual(killed, [(99, signal.SIGKILL if hasattr(signal, 'SIGKILL') else 9)])
                close.assert_called_once_with(99)
            self.assertEqual(tool.read(args.output / tool.LIMIT)['max_train'], 1)


if __name__ == '__main__':
    unittest.main()
