from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from baselines.B5.diagnostic_signals import slow_component
from baselines.B5.plot_anchor_slow import aligned_anchor, anchor_index, make_curves, run


class FakeDataset:
    def __init__(self, rows, target):
        # Reordered cache tests identity alignment rather than positional indexing.
        self._rows, self.target = [dict(row) for row in reversed(rows)], target[::-1].copy()
        self._indices = list(range(len(rows)))

    def __getitem__(self, index):
        return SimpleNamespace(anchor_i_ecg=self.target[index, :1].copy(), Y_12lead=self.target[index].copy())


def fixture(root):
    folder = root / 'evaluation' / 'task2'
    folder.mkdir(parents=True)
    t = np.arange(10000)/500.
    target = np.zeros((2, 12, 5000), dtype=np.float32)
    target[:, 0] = (50*np.sin(t/3)).reshape(2, 5000)
    target[:, 8] = (500*t/20-900+150*np.sin(t*8)).reshape(2, 5000)
    prediction = target.copy()
    prediction[:, 0] = 9999  # Must never be used as input I.
    prediction[:, 8] -= (500*t/20).reshape(2, 5000)
    rows = [{'pair_id': 'pair', 'target_record_id': 'record', 'subject_id': 's',
             'start_sample_500hz': str(i*5000), 'expected_window_count': '2'} for i in range(2)]
    with (folder/'window_metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.save(folder/'target_uV.npy', target)
    np.save(folder/'prediction_uV.npy', prediction)
    return rows, target, FakeDataset(rows, target)


class AnchorSlowTests(unittest.TestCase):
    def test_identity_alignment_and_stale_target_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, target, dataset = fixture(Path(tmp))
            anchor = aligned_anchor(dataset, anchor_index(dataset), rows, target, [0, 1])
            np.testing.assert_array_equal(anchor, target[:, 0].reshape(-1))
            changed = target.copy(); changed[0, 8, 0] += 1
            with self.assertRaisesRegex(ValueError, 'target disagrees'):
                aligned_anchor(dataset, anchor_index(dataset), rows, changed, [0, 1])
            rows[0]['subject_id'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'subject mismatch'):
                aligned_anchor(dataset, anchor_index(dataset), rows, target, [0, 1])

    def test_filter_complete_record_and_display_only_centering(self):
        x = np.r_[np.zeros(5000), np.full(5000, 1000.)]
        original = x.copy()
        slow, centered = make_curves(x, x+100, x-200, 501)
        np.testing.assert_allclose(slow[0], slow_component(x, 501))
        self.assertGreater(slow[0, 4999], 0)
        self.assertLess(slow[0, 5000], 1000)
        np.testing.assert_allclose(centered.mean(axis=1), 0, atol=1e-10)
        np.testing.assert_array_equal(x, original)

    def test_full_export_uses_actual_anchor_and_preserves_input_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, target, dataset = fixture(root)
            folder = root/'evaluation'/'task2'
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            output = root/'out'
            manifest = run(root/'evaluation', output, dataset, pair_ids=['pair'])
            self.assertFalse(manifest['training_started'])
            self.assertFalse(manifest['model_inference_started'])
            self.assertTrue(manifest['selected_cache_targets_match_evaluation'])
            with np.load(output/'record_00_V3_anchor_slow_curves.npz') as arrays:
                np.testing.assert_allclose(arrays['anchor_slow_uV'], slow_component(target[:, 0].reshape(-1), 501))
                self.assertEqual(len(arrays['time_seconds']), 10000)
                self.assertLess(abs(arrays['anchor_slow_uV']).max(), 100)
            self.assertTrue((output/'record_00_V3_anchor_slow_raw.png').stat().st_size > 10000)
            self.assertTrue((output/'record_00_V3_anchor_slow_centered.png').stat().st_size > 10000)
            self.assertEqual(json.loads((output/'plot_manifest.json').read_text())['selection'], 'explicit pair IDs')
            for path in folder.iterdir():
                self.assertEqual(path.read_bytes(), before[path.name])
            with self.assertRaises(FileExistsError):
                run(root/'evaluation', output, dataset, pair_ids=['pair'])

    def test_unknown_pair_and_incomplete_record_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, target, dataset = fixture(root)
            with self.assertRaisesRegex(ValueError, 'pair IDs'):
                run(root/'evaluation', root/'out', dataset, pair_ids=['absent'])
            metadata = root/'evaluation'/'task2'/'window_metadata.csv'
            metadata.write_text(metadata.read_text().replace('5000,2', '10000,2'))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                run(root/'evaluation', root/'out', dataset)


if __name__ == '__main__':
    unittest.main()
