"""Canonical PTB patient directories must define sample identity, never root copies."""
from argparse import Namespace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.B7 import D12_LEADS
from baselines.B7.data import WindowDataset, read_manifest
from baselines.B7.prepare import prepare, public_index


def write_record(directory, name, samples=5000):
    directory.mkdir(parents=True, exist_ok=True)
    raw = (np.arange(samples * 12).reshape(samples, 12) % 1000 - 500).astype('<i2')
    (directory / (name + '.dat')).write_bytes(raw.tobytes())
    lines = [f'{name} 12 500 {samples}']
    for j, lead in enumerate(D12_LEADS):
        checksum = int(raw[:, j].sum(dtype='int64')) % 65536
        lines.append(f'{name}.dat 16 1000/mV 16 0 {raw[0,j]} {checksum} 0 {lead}')
    lines += ['# age: 50', '# sex: male']
    path = directory / (name + '.hea')
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


class PTBIndexTests(unittest.TestCase):
    def test_root_duplicate_and_unowned_root_record_are_excluded_not_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_record(root / 'patient279', 's0533_re')
            copy = root / path.name
            copy.write_bytes(path.read_bytes())
            (root / 'orphan.hea').write_text('not a patient record', encoding='utf-8')
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            audit = []
            rows = public_index('ptbdb', root, audit)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['record_id'], 'ptbdb:patient279/s0533_re')
            self.assertEqual(rows[0]['target_record_id'], rows[0]['record_id'])
            self.assertEqual(rows[0]['subject_id'], 'ptbdb:patient279')
            self.assertEqual(Path(rows[0]['path']), path.resolve())
            self.assertEqual(audit[0]['headers'], 3)
            self.assertEqual(audit[0]['retained'], 1)
            self.assertEqual(set(audit[0]['excluded_root_headers']), {'s0533_re.hea', 'orphan.hea'})
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_split_remains_patient_hash_and_records_do_not_change_patient(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_record(root / 'patient279', 's0533_re')
            write_record(root / 'patient279', 's0534_re')
            rows = public_index('ptbdb', root)
            number = int(hashlib.sha256(b'B7-split|ptbdb|patient279').hexdigest()[:8], 16) % 10
            split = 'validation' if number == 8 else 'test' if number == 9 else 'train'
            self.assertEqual({r['split'] for r in rows}, {split})
            self.assertEqual({r['subject_id'] for r in rows}, {'ptbdb:patient279'})

    def test_record_identity_includes_patient_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for patient in ('patient001', 'patient002'):
                write_record(root / patient, 'sample')
            rows = public_index('ptbdb', root)
            self.assertEqual({r['record_id'] for r in rows}, {'ptbdb:patient001/sample', 'ptbdb:patient002/sample'})
            self.assertEqual(len({r['target_record_id'] for r in rows}), 2)

    def test_root_copy_does_not_reorder_or_change_canonical_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = write_record(root / 'patient002', 'b')
            write_record(root / 'patient001', 'a')
            before = public_index('ptbdb', root)
            (root / '000_first.hea').write_bytes(a.read_bytes())
            self.assertEqual(before, public_index('ptbdb', root))

    def test_unexpected_nested_headers_fail_instead_of_guessing_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_record(root / 'patient001', 'a')
            write_record(root / 'backup/patient001', 'a')
            with self.assertRaisesRegex(ValueError, 'Unexpected nested PTB header'):
                public_index('ptbdb', root)

    def test_flat_directory_without_patient_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_record(root, 'sample')
            with self.assertRaisesRegex(ValueError, 'patient-number directories'):
                public_index('ptbdb', root)

    def test_prepare_retains_audit_and_produces_one_real_40s_training_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            patient = next(f'patient{i:03d}' for i in range(1, 100)
                           if int(hashlib.sha256(f'B7-split|ptbdb|patient{i:03d}'.encode()).hexdigest()[:8], 16) % 10 < 8)
            data = root / 'raw'
            path = write_record(data / patient, 'sample', samples=20000)
            (data / 'sample.hea').write_bytes(path.read_bytes())
            output = root / 'prepared'
            prepare(Namespace(index=None, public_source='ptbdb', root=str(data),
                              output=str(output), scale_manifest=None, resume=False))
            manifest = read_manifest(output / 'manifest.json')
            self.assertEqual(len(manifest['records']), 1)
            self.assertEqual(manifest['public_index_audit'][0]['excluded_root_headers'], ['sample.hea'])
            self.assertEqual(manifest['records'][0]['subject_id'], f'ptbdb:{patient}')
            self.assertEqual(manifest['records'][0]['real_samples'], 20000)
            self.assertEqual(len(WindowDataset(manifest, 40)), 1)
            self.assertTrue((data / 'sample.hea').is_file())


if __name__ == '__main__':
    unittest.main()
