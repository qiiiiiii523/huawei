from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.B7 import D12_LEADS
from baselines.B7.prepare import prepare, read_wfdb
from baselines.B7.pipeline import command_for, Job, preparation_only_retry_allowed
from baselines.B7.tests.test_b7 import make_manifest


class PreparationRecoveryTests(unittest.TestCase):
    def test_joined_first_signal_line_is_checked_and_reordered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = (np.arange(5000*12).reshape(5000,12) % 300 - 150).astype('<i2')
            (root/'sample.mat').write_bytes(bytes(24) + raw.tobytes())
            rows = []
            for j, lead in enumerate(D12_LEADS):
                rows.append(f'sample.mat 16+24 1000/mV 16 0 {raw[0,j]} {int(raw[:,j].sum())%65536} 0 {lead}')
            path = root/'sample.hea'
            path.write_text('SAMPLE 12 500 5000 '+rows[0]+'\n'+'\n'.join(rows[1:]), encoding='utf-8')
            actual, fs, repair = read_wfdb(path)
            np.testing.assert_allclose(actual, raw.T, rtol=1e-7, atol=2e-5)
            self.assertEqual(fs,500)
            self.assertEqual(repair,'WFDB_joined_first_signal_line_verified')
            rows[3] = rows[3].replace(' aVR', ' I')
            path.write_text('SAMPLE 12 500 5000 '+rows[0]+'\n'+'\n'.join(rows[1:]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'standard12'):
                read_wfdb(path)

    def prepare_partial(self, root, corrupt=False):
        manifest = make_manifest(root)
        rows = json.loads(manifest.read_text())['records']
        index = root/'index.json'
        index.write_text(json.dumps(rows), encoding='utf-8')
        cache = root/'prepared'
        (cache/'records').mkdir(parents=True)
        first = cache/'records/0000000.npz'
        if corrupt:
            with np.load(root/'train.npz') as value:
                np.savez(first, anchor_uV=value['anchor_uV']+1, target_uV=value['target_uV'], lead_mask=value['lead_mask'])
        else:
            first.write_bytes((root/'train.npz').read_bytes())
        return Namespace(index=str(index), output=str(cache), public_source=None, root=None,
                         scale_manifest=None, resume=True), first

    def test_resume_verifies_prefix_without_rewriting_then_completes(self):
        with tempfile.TemporaryDirectory() as temp:
            args, first = self.prepare_partial(Path(temp))
            before = (first.read_bytes(), first.stat().st_mtime_ns)
            prepare(args)
            self.assertEqual(before, (first.read_bytes(),first.stat().st_mtime_ns))
            self.assertTrue((Path(args.output)/'records/0000001.npz').is_file())
            self.assertTrue((Path(args.output)/'manifest.json').is_file())

    def test_resume_rejects_cache_from_different_raw_data(self):
        with tempfile.TemporaryDirectory() as temp:
            args, first = self.prepare_partial(Path(temp),corrupt=True)
            before = first.read_bytes()
            with self.assertRaisesRegex(ValueError,'differs from raw record'):
                prepare(args)
            self.assertEqual(first.read_bytes(),before)
            self.assertFalse((Path(args.output)/'manifest.json').exists())

    def test_retry_is_opt_in_and_disallowed_after_training_started(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=Namespace(output=root,retry_preparation=False)
            self.assertFalse(preparation_only_retry_allowed(args))
            args.retry_preparation=True
            self.assertTrue(preparation_only_retry_allowed(args))
            job=Job('prepare',(),'prepare',['--output',str(root),'--resume'],root/'manifest.json')
            self.assertIn('--resume',command_for(job))
            (root/'P').mkdir()
            self.assertFalse(preparation_only_retry_allowed(args))


if __name__ == '__main__':
    unittest.main()
