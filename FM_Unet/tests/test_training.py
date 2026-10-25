from argparse import Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from FM_Unet.config import LossConfig
from FM_Unet.fm_experiment import build
from FM_Unet.train import train
from FM_Unet.train_data import load_record, native_target, read_manifest
from FM_Unet.train_eval import aggregate, sample_window
from FM_Unet.train_loss import flow_loss


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, directory):
        root = Path(directory)
        rows = []
        for i, split in enumerate(('train', 'train', 'validation')):
            x = np.arange(5000)/500
            target = np.stack([100*np.sin(2*np.pi*(.2+j*.01)*x)+j for j in range(12)]).astype('float32')
            path = root/f'{i}.npz'
            np.savez(path, anchor_uV=target[:1], target_uV=target, lead_mask=np.ones(12, bool))
            rows.append(dict(record_id=str(i), subject_id=str(i), target_record_id=str(i),
                             source='ptbxl' if i != 1 else 'chapman', group='public', path=path.name,
                             split=split, input_reliable=True, real_samples=5000))
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps(dict(protocol='B7-raw-uV-records-v1', sample_rate=500,
            public_qc_version='B7-public-QC-v2', records=rows)))
        config = json.loads((Path(__file__).parents[2]/'configs/fm_unet_m7/public10.json').read_text())
        config['training'].update(batch_size=2, updates=2, checkpoint_every=1)
        path = root/'config.json'; path.write_text(json.dumps(config))
        return manifest, path

    def test_masked_nonfinite_targets_do_not_change_loss(self):
        model = build(rate=10, input_seconds=10, output_seconds=10)
        mask = torch.ones(1, 7, dtype=torch.bool); mask[:, 1] = False
        batch = dict(anchor=torch.zeros(1, 1, 5000), target=torch.zeros(1, 7, 100),
                     lead_mask=mask, metadata=None)
        noise, time = torch.randn(1, 7, 100), torch.tensor([.5])
        first = flow_loss(model, batch, LossConfig(), [1.]*7, noise=noise, time=time)[0]
        batch['target'][:, 1] = float('nan')
        second = flow_loss(model, batch, LossConfig(), [1.]*7, noise=noise, time=time)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        second.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_native_targets_mask_before_filtering(self):
        target = np.ones((12, 5000), np.float32)
        mask = np.ones(12, bool); mask[6] = False; target[6] = np.nan
        actual = native_target(target, mask, [1.]*12, 10)
        self.assertEqual(actual.shape, (7, 100))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual[1], 0)
        np.testing.assert_allclose(actual[0], 1, atol=1e-6)

    def test_no_midrun_validation_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, config = self.fixture(directory)
            common = dict(manifest=str(manifest), config=str(config), cache_dir=str(Path(directory)/'cache'),
                          statistics=str(Path(__file__).parents[1]/'best.pt'), device='cpu', cpu_threads=2,
                          init_checkpoint=None, fast_code=None, fast_checkpoint=None, demographics_csv=None)
            first = Namespace(**common, output=str(Path(directory)/'interrupted'), resume=False, stop_after=1)
            train(first)
            self.assertFalse((Path(first.output)/'validation_final.json').exists())
            first.resume, first.stop_after = True, 2
            # Stop after final update for the test using a larger budget, preserving resume identity.
            # The real final-only evaluator is exercised by the standalone sampling test below.
            from unittest.mock import patch
            def evaluated(*args, **kw):
                return {'r': .1, 'score_scope': 'test_only'}, []
            with patch('FM_Unet.train.evaluate', side_effect=evaluated) as validation:
                train(first)
                self.assertEqual(validation.call_count, 1)
            reference = Namespace(**common, output=str(Path(directory)/'reference'), resume=False, stop_after=None)
            with patch('FM_Unet.train.evaluate', side_effect=evaluated) as validation:
                train(reference)
                self.assertEqual(validation.call_count, 1)
            a = torch.load(Path(first.output)/'last.pt', weights_only=True)
            b = torch.load(Path(reference.output)/'last.pt', weights_only=True)
            self.assertTrue(a['final_validated'])
            for field in ('model', 'ema'):
                for key in a[field]:
                    torch.testing.assert_close(a[field][key], b[field][key], rtol=0, atol=0)
            self.assertTrue((Path(first.output)/'inference.pt').is_file())

    def test_real_window_sampling_and_group_weights(self):
        model = build(rate=10, input_seconds=10, output_seconds=10).eval()
        from FM_Unet.train_data import empty_metadata
        settings = dict(samples=2, steps=1, sample_batch_size=2)
        result = sample_window(model, np.zeros((1, 5000)), empty_metadata(), [1.]*12, [1.]*7, 'test', settings)
        self.assertEqual(result.shape, (11, 5000))
        self.assertTrue(np.isfinite(result).all())
        rows = [dict(group=group, target_record_id=group,
                     leads={str(j): dict(r=value, rmse_uV=1.) for j in range(1, 12)})
                for group, value in [('task1', .8), ('machine', .4), ('body', .6)]]
        self.assertAlmostEqual(aggregate(rows, 'huawei120')['r'], .65)


if __name__ == '__main__':
    unittest.main()
