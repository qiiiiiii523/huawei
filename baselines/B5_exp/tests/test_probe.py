"""Synthetic CPU checks. Never run an optimizer or train on ECG data."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from baselines.B5_exp import PROTOCOL
from baselines.B5_exp.core import (SlowCNN, contexts, generate, objective, slow, downsample,
                                 upsample, correlations, macro, Records, sha, json_write, load_prepared)
from baselines.B5_exp.prepare import group_windows, pack
from baselines.B5_exp.evaluate import hybrid_wave
from baselines.B5_exp.train import state_hash
from baselines.B5.metadata import encode_demographics
from ecg12gen.evaluate import evaluate_record_predictions, COMPETITION_SCORE_PROTOCOL


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_context_only_changes_input_range(self):
        x = torch.arange(6000.)[None, None]
        short, long = contexts(x, 10), contexts(x, 30)
        self.assertEqual(tuple(short.shape), (12, 1, 500))
        self.assertEqual(tuple(long.shape), (12, 1, 1500))
        self.assertTrue(torch.equal(short, long[..., 500:1000]))
        self.assertTrue(torch.equal(long[1, 0], x[0, 0, :1500]))
        self.assertTrue(torch.equal(long[0, 0, :500], torch.zeros(500)))
        self.assertTrue(torch.equal(long[-1, 0, -500:], torch.full((500,), 5999.)))

    def test_identical_initialization_and_input_interface(self):
        torch.manual_seed(42)
        a = SlowCNN()
        torch.manual_seed(42)
        b = SlowCNN()
        self.assertEqual(state_hash(a.state_dict()), state_hash(b.state_dict()))
        self.assertNotIn('target', inspect.signature(a.forward).parameters)
        meta = {'numeric': torch.zeros(1, 3), 'sex': torch.tensor([2]),
                'field_mask': torch.zeros(1, 4, dtype=torch.bool), 'age_topcoded': torch.zeros(1, 1)}
        with torch.no_grad():
            for length in (500, 1500):
                self.assertEqual(tuple(a(torch.ones(1, 1, length), **meta).shape), (1, 11, length))

    def test_center_outputs_stitch_in_order(self):
        class Echo(torch.nn.Module):
            def forward(self, anchor, **kwargs):
                return anchor.expand(-1, 11, -1)
        batch = {'anchor': torch.arange(12000.).reshape(2, 1, 6000), 'numeric': torch.zeros(2, 3),
                 'sex': torch.tensor([2, 2]), 'field_mask': torch.zeros(2, 4, dtype=torch.bool),
                 'age_topcoded': torch.zeros(2, 1)}
        a, b = generate(Echo(), batch, 10), generate(Echo(), batch, 30)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[:, :1], batch['anchor']))

    def test_full_record_filter_then_crop_and_means_retained(self):
        x = np.zeros((12, 60000))
        x[:, 5000:] = 100.
        full = slow(x)
        self.assertGreater(full[0, 4999], 0.)  # A per-window filter would produce 0.
        self.assertAlmostEqual(float(slow(np.full_like(x, 123.4))[0].mean()), 123.4, places=6)
        self.assertEqual(downsample(full).shape, (12, 6000))
        self.assertTrue(np.allclose(upsample(np.full((11, 6000), 321.)), 321.))

    def test_mask_and_absolute_and_cross_window_errors(self):
        target = torch.zeros(1, 11, 6000)
        target[..., 3000:] = 1.
        mask = torch.ones(1, 11, dtype=torch.bool)
        mask[:, -1] = False
        target[:, -1] = float('nan')
        p = torch.zeros_like(target, requires_grad=True)
        loss, parts = objective(p, target, mask)
        self.assertGreater(float(loss.detach()), 0.)
        loss.backward()  # Derivative check only; no optimizer updates.
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(torch.equal(p.grad[:, -1], torch.zeros_like(p.grad[:, -1])))
        clean = torch.nan_to_num(target)
        exact, _ = objective(clean, clean, mask)
        offset, _ = objective(clean + 2., clean, mask)
        self.assertLess(float(exact), 1e-6)
        self.assertGreater(float(offset), 1.)

    def test_hybrid_uses_prediction_not_target_and_keeps_I(self):
        p = np.random.default_rng(4).normal(size=(12, 60000))
        h = slow(p[1:])
        combined = hybrid_wave(p, h)
        self.assertTrue(np.array_equal(combined[0], p[0]))
        self.assertTrue(np.allclose(combined, p, atol=1e-12))
        self.assertNotIn('target', inspect.signature(hybrid_wave).parameters)
        with self.assertRaises(ValueError):
            hybrid_wave(p, h[:, :100])

    def test_prepared_hashes_and_record_contract(self):
        rows = [{'target_record_id': 't', 'subject_id': 'p', 'start_sample_500hz': str(i * 5000)} for i in range(12)]
        self.assertEqual(len(group_windows(rows)['t']), 12)
        with self.assertRaises(ValueError):
            group_windows(rows[:-1])
        class Demo:
            def get(self, subject):
                return encode_demographics({'age': 40, 'gender': 'male'})
        raw = np.full((12, 60000), 73., dtype=np.float32)
        arrays = pack([{'raw': raw, 'anchor': raw[:1], 'mask': np.ones(12, bool), 'subject_id': 'p'}], Demo(), np.ones(12))
        self.assertTrue(np.allclose(arrays['target'], 73.))
        self.assertEqual(arrays['sex'].shape, (1,))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            np.savez_compressed(path / 'train.npz', **arrays)
            json_write(path / 'prepared.json', {'protocol': PROTOCOL, 'completed': True,
                                                'files': {'train.npz': sha(path / 'train.npz')}})
            load_prepared(path)
            self.assertEqual(len(Records(path / 'train.npz')), 1)
            with (path / 'train.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                load_prepared(path)

    def test_latest_target_equal_score_is_used(self):
        self.assertEqual(COMPETITION_SCORE_PROTOCOL, 'correlation50_bonus10_target_record_v2')
        t = np.tile(np.linspace(-1, 1, 5000), (12, 1))
        prediction = np.stack([p for p in (t, t, -t) for _ in range(12)])
        target = np.stack([t for _ in range(36)])
        rows = [{'pair_id': pair, 'target_record_id': target_id, 'subject_id': subject,
                 'start_sample_500hz': str(w * 5000), 'expected_window_count': '12'}
                for pair, target_id, subject in (('a1', 'a', 'pa'), ('a2', 'a', 'pa'), ('b1', 'b', 'pb'))
                for w in range(12)]
        summary, _ = evaluate_record_predictions(prediction, target, 'task1', rows)
        self.assertAlmostEqual(summary['r_missing11'], 0., places=7)  # Pair weighting would be 1/3.
        self.assertEqual(summary['n_unique_target_records'], 2)


if __name__ == '__main__':
    unittest.main()
