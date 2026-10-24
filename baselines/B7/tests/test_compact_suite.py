from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baselines.B7 import compact_suite, pipeline


class CompactSuiteTests(unittest.TestCase):
    def args(self, root):
        return Namespace(output=root / 'suite', data_root=root / 'raw', metadata=root / 'metadata',
                         ptb_root=root / 'ptb', reuse_data_from=root / 'cache', gpu='0',
                         max_train=2, probe_first=False, memory_budget_gib=35., dry_run=False)

    def test_repaired_PTB_cache_is_preferred_and_legacy_cache_is_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            self.assertEqual(pipeline.reused_manifests(args.reuse_data_from)['ptb40'],
                             args.reuse_data_from / 'data/ptb40/manifest.json')
            repaired = args.reuse_data_from / 'ptb_recovery/data/manifest.json'
            repaired.parent.mkdir(parents=True)
            repaired.write_text('{}')
            jobs = compact_suite.suite_jobs(args)
            for job in jobs:
                if job.name.endswith(':Q'):
                    self.assertEqual(job.arguments[job.arguments.index('--manifest') + 1], str(repaired))
            repaired.unlink()
            self.assertEqual(pipeline.reused_manifests(args.reuse_data_from)['ptb40'],
                             args.reuse_data_from / 'data/ptb40/manifest.json')

    def test_eight_stage_graph_preserves_independent_lineage(self):
        args = self.args(Path('/example'))
        jobs = compact_suite.suite_jobs(args)
        self.assertEqual(len(jobs), 8)
        self.assertTrue(all(j.name.endswith(':P') for j in jobs[:2]))
        for job in jobs:
            preset, stage = job.name.split(':')
            self.assertTrue(all(dep.startswith(preset + ':') for dep in job.dependencies))
            self.assertEqual(job.artifact.parent, args.output.resolve() / preset / 'seed42' / stage)
            if stage == 'P':
                self.assertNotIn('--init-checkpoint', job.arguments)
            else:
                parent = 'Q' if stage == 'B' else 'P'
                self.assertEqual(Path(job.arguments[job.arguments.index('--init-checkpoint') + 1]),
                                 job.artifact.parent.parent / parent / 'best_r.pt')

    def test_suite_rejects_unbounded_or_more_than_two_workers(self):
        for count in (0, 3, 4):
            args = self.args(Path('/example'))
            args.max_train = count
            with self.assertRaisesRegex(ValueError, 'at most two'):
                compact_suite.run(args)

    def test_real_scheduler_global_two_limit_and_failure_isolation(self):
        for failed in (None, 'g6_19m:Q'):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as temp:
                args = self.args(Path(temp))
                for folder in (args.data_root, args.metadata, args.ptb_root):
                    folder.mkdir()
                jobs = compact_suite.suite_jobs(args)
                args.config_dir = Path(temp) / 'unused-config-root'
                args.experiment_suite = True
                active, peak, events = set(), [0], []
                by_output = {str(j.artifact.parent): j for j in jobs}
                class Process:
                    def __init__(self, command, **kw):
                        self.job = by_output[command[command.index('--output') + 1]]
                        self.polls = 0
                        active.add(self.job.name)
                        peak[0] = max(peak[0], len(active))
                        events.append(('start', self.job.name))
                    def poll(self):
                        self.polls += 1
                        if self.polls < (4 if self.job.name.endswith(':A') else 2):
                            return None
                        active.remove(self.job.name)
                        events.append(('end', self.job.name))
                        return int(self.job.name == failed)
                with patch.object(pipeline, 'fingerprint', return_value='test'), \
                     patch('torch.cuda.is_available', return_value=True), \
                     patch.object(pipeline.subprocess, 'Popen', Process), \
                     patch.object(pipeline, 'validate_artifact', return_value={'best_r': .7}), \
                     patch('baselines.B7.data.read_manifest', return_value={'scales_uV': [1.] * 12, 'records': []}), \
                     patch.object(pipeline.time, 'sleep'):
                    code = pipeline.run(args, jobs=jobs)
                self.assertEqual(peak[0], 2)
                self.assertFalse(active)
                status = json.loads((args.output / 'pipeline_status.json').read_text())
                self.assertEqual(code, int(failed is not None))
                self.assertEqual(status['jobs']['w144_g5_21m:B'], 'done')
                self.assertEqual(status['jobs']['g6_19m:B'], 'blocked' if failed else 'done')
                for job in jobs:
                    if ('start', job.name) in events:
                        for dep in job.dependencies:
                            self.assertLess(events.index(('end', dep)), events.index(('start', job.name)))

    def test_failed_probe_prevents_all_training(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            args.probe_first = True
            process = unittest.mock.Mock()
            process.wait.return_value = process.poll.return_value = 2
            with patch('baselines.B7.data.read_manifest', return_value={'scales_uV': [1.] * 12, 'records': []}), \
                 patch.object(compact_suite.subprocess, 'Popen', return_value=process), \
                 patch.object(pipeline, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                    compact_suite.run(args)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
