"""Known-version pre-training repair must preserve GPU work and fail closed otherwise."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from baselines.B5_best import SAMPLING, SCORE_PROTOCOL
from baselines.B5_best.bundle import export_bundle
from baselines.B5_best.common import read, write, sha
from baselines.B5_best.tests.test_hybrid import fixture
from baselines.B5_slow_pcc import PROTOCOL, CACHE_PROTOCOL
from baselines.B5_slow_pcc.common import code_digest
from baselines.B5_slow_pcc.repair_cache import repair, OLD_CODE_SHA256, OLD_CACHE_PROTOCOL


def old_run(root):
    shared, cache = root / 'shared', root / 'cache'
    shared.mkdir(); (cache / 'train').mkdir(parents=True)
    fp, sp, ip, profile = fixture(shared)
    export_bundle(fp, sp, ip, shared / 'components.pt', profile)
    study_path = shared / 'study.json'
    write(study_path, {'protocol': PROTOCOL, 'sampling': dict(SAMPLING), 'score_protocol': SCORE_PROTOCOL,
          'beta': .75, 'code_sha256': OLD_CODE_SHA256, 'bundle_sha256': sha(shared / 'components.pt'),
          'source_checks': {}, 'source_evaluation_checks': {}, 'train_records': 1})
    path = cache / 'train/generated.npz'
    np.savez(path, fm_uV=np.ones((12, 60000), np.float32))
    write(cache / 'progress.json', {'protocol': OLD_CACHE_PROTOCOL, 'study_sha256': sha(study_path),
          'records': {'train:record': {'target_record_id': 'record', 'subject_id': 'train_patient',
                      'file': 'train/generated.npz', 'sha256': sha(path), 'data_sha256': 'original'}}})
    return study_path, cache, path


class CacheRepairTests(unittest.TestCase):
    def test_known_source_rebinds_only_manifest_identity_and_keeps_cached_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            study, cache, path = old_run(Path(tmp))
            original_file, original_study = sha(path), sha(study)
            result = repair(study, cache)
            self.assertEqual(result['train_records_reused'], 1)
            self.assertEqual(sha(path), original_file)
            self.assertEqual(read(study)['code_sha256'], code_digest())
            self.assertEqual(sha(study.with_name('study.before_pair_cache_fix.json')), original_study)
            self.assertEqual(read(cache / 'progress.json')['protocol'], CACHE_PROTOCOL)
            self.assertEqual(read(cache / 'progress.json')['study_sha256'], sha(study))
            first = sha(study)
            repair(study, cache)  # Restart/idempotent repair, no new FM sampling.
            self.assertEqual(sha(study), first)
            self.assertEqual(sha(path), original_file)

    def test_rejects_started_training_and_unknown_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study, cache, _ = old_run(root)
            original = sha(study)
            (root / 'control').mkdir()
            (root / 'control/last.pt').write_bytes(b'existing training state')
            with self.assertRaisesRegex(ValueError, 'Training already started'):
                repair(study, cache)
            self.assertEqual(sha(study), original)
        with tempfile.TemporaryDirectory() as tmp:
            study, cache, _ = old_run(Path(tmp))
            value = read(study); value['code_sha256'] = 'unrecognized implementation'; write(study, value)
            with self.assertRaisesRegex(ValueError, 'exact released'):
                repair(study, cache)

    def test_finish_requests_cpu_reuse_only_never_new_FM_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            study, cache, path = old_run(Path(tmp))
            original = sha(path)
            with patch('baselines.B5_slow_pcc.cache.build') as build, \
                 patch('baselines.B5_slow_pcc.repair_cache.load_cache', return_value={'completed': True}):
                report = repair(study, cache, finish=True)
            build.assert_called_once_with(study.resolve(), cache.resolve(), 'cpu', resume=True, reuse_only=True)
            self.assertTrue(report['cache_completed'])
            self.assertEqual(sha(path), original)

    def test_rejects_corrupt_cache_without_rewriting_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            study, cache, path = old_run(Path(tmp))
            original = sha(study)
            with path.open('ab') as handle: handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'saved cache record changed'):
                repair(study, cache)
            self.assertEqual(sha(study), original)


if __name__ == '__main__':
    unittest.main()
