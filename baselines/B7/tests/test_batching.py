"""Execution batching must preserve noise identity, conditions and scoring."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.data import read_manifest
from baselines.B7.evaluate import evaluate
from baselines.B7.flow import sample
from baselines.B7.inference import predict_record, predict_records
from baselines.B7.losses import flow_loss
from baselines.B7.model import B7Transformer
from baselines.B7.tests.test_b7 import make_manifest, small_config
from baselines.B7.train import validate_settings


class ConditionalVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(.1))
        self.batches = []

    def forward(self, state, time, anchor, metadata=None):
        self.batches.append(len(state))
        offset = 0 if metadata is None else metadata['numeric'][:, :1, None] * .01
        return state * self.marker + anchor * .02 + time[:, None, None] + offset


def demographics(value):
    return {'numeric': np.array([value, 0., 0.], dtype='float32'), 'sex': np.array(2),
            'field_mask': np.zeros(4, dtype=bool), 'age_topcoded': np.zeros(1, dtype='float32')}


class BatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_production_training_and_validation_batches_are_32(self):
        root = Path(__file__).resolve().parents[3]
        for name in ('b7_public10.json', 'b7_public40.json', 'b7_huawei40.json'):
            cfg = json.loads((root / 'configs' / name).read_text())
            validate_settings(cfg)
            self.assertEqual(cfg['training']['batch_size'], 32)
            self.assertEqual(cfg['validation'], {'batch_size': 32})
            self.assertEqual(cfg['sampling']['samples'], 16)
            self.assertEqual(cfg['sampling']['steps'], 16)

    def test_k16_packed_trajectories_match_serial_with_partial_batches(self):
        model = ConditionalVelocity().eval()
        q = torch.arange(3 * 40).reshape(3, 1, 40).float() / 100
        original = q.clone()
        metadata = {'numeric': torch.tensor([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])}
        sc = SamplingConfig(samples=16, steps=2)
        expected = sample(model, q, ['a', 'b', 'c'], metadata, sc)
        for size in (1, 5, 32, 64):
            model.batches.clear()
            actual = sample(model, q, ['a', 'b', 'c'], metadata, sc, batch_size=size)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertLessEqual(max(model.batches), size)
            self.assertEqual(len(model.batches), ((48 + size - 1) // size) * 4)
        self.assertTrue(torch.equal(q, original))

    def test_full120_heun16_k16_matches_serial_and_reduces_calls(self):
        model = ConditionalVelocity().eval()
        anchor = torch.sin(torch.arange(60000) / 23)[None]
        expected = predict_record(model, anchor, [100.] * 12, 'long', config=SamplingConfig())
        self.assertEqual(len(model.batches), 6144)
        model.batches.clear()
        actual = predict_record(model, anchor, [100.] * 12, 'long', config=SamplingConfig(), batch_size=32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual[:1], anchor))
        self.assertEqual(len(model.batches), 192)
        self.assertEqual(set(model.batches), {32})

    def test_records_different_lengths_metadata_and_order_are_preserved(self):
        model = ConditionalVelocity().eval()
        anchors = [torch.arange(n)[None].float() / 500 for n in (5000, 15000, 10000)]
        meta = [demographics(1), None, demographics(9)]
        sc = SamplingConfig(samples=16, steps=1)
        expected = [predict_record(model, a, [100.] * 12, str(i), m, sc)
                    for i, (a, m) in enumerate(zip(anchors, meta))]
        for order in ([0, 1, 2], [2, 0, 1]):
            actual = predict_records(model, [anchors[i] for i in order], [100.] * 12,
                [str(i) for i in order], [meta[i] for i in order], sc, batch_size=32)
            for i, output in zip(order, actual):
                torch.testing.assert_close(output, expected[i], rtol=0, atol=0)

    def test_real_transformer_batched_sampling_matches_serial(self):
        torch.manual_seed(81)
        model = B7Transformer(small_config()).eval()
        q = torch.randn(3, 1, 80)
        sc = SamplingConfig(samples=16, steps=2)
        a = sample(model, q, ['a', 'b', 'c'], config=sc)
        b = sample(model, q, ['a', 'b', 'c'], config=sc, batch_size=32)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)

    def test_evaluation_QC_target_group_score_parity_and_cross_record_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = make_manifest(root)
            manifest = read_manifest(path)
            first = manifest['records'][1]
            with np.load(first['resolved_path']) as value:
                target = value['target_uV'].copy()
                mask = value['lead_mask'].copy()
            mask[2:6] = False
            np.savez(root / 'other.npz', anchor_uV=target[:1], target_uV=target, lead_mask=mask)
            # Preserve repeated-target pair averaging as well as per-lead QC.
            manifest['records'] += [dict(first, record_id='pair2'), dict(first, record_id='other',
                subject_id='other', target_record_id='other', resolved_path=str(root / 'other.npz'))]
            sc = SamplingConfig(window_seconds=10, samples=16, steps=1)
            model = ConditionalVelocity().eval()
            expected = evaluate(model, manifest, sc, 'public10', batch_size=1)
            model.batches.clear()
            actual = evaluate(model, manifest, sc, 'public10', batch_size=32)
            self.assertEqual(actual, expected)
            self.assertEqual(model.batches, [32, 32, 16, 16])

    def test_batch32_real10s_transformer_optimizer_step(self):
        torch.manual_seed(42)
        model = B7Transformer(small_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        anchor = torch.randn(32, 1, 5000)
        original = anchor.clone()
        batch = {'anchor': anchor, 'target': torch.randn(32, 7, 5000),
                 'lead_mask': torch.ones(32, 7, dtype=torch.bool)}
        loss, _ = flow_loss(model, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        before = model.patch_embed.projection.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.patch_embed.projection.weight))
        self.assertTrue(torch.equal(original, anchor))

    def test_invalid_execution_batches_are_rejected(self):
        model = ConditionalVelocity().eval()
        for size in (0, -1, True, 2.5):
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                sample(model, torch.zeros(1, 1, 40), ['a'], batch_size=size)
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                evaluate(model, {}, SamplingConfig(), 'public10', batch_size=size)
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                predict_record(model, torch.zeros(1, 5000), [100.] * 12, 'a', batch_size=size)
        cfg = {'stage': 'public10', 'model': small_config().to_dict(), 'loss': asdict(LossConfig()),
               'sampling': asdict(SamplingConfig(window_seconds=10)), 'training': {}}
        for size in (0, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'validation'):
                validate_settings(dict(cfg, validation={'batch_size': size}))


if __name__ == '__main__':
    unittest.main()
