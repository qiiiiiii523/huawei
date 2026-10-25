from argparse import Namespace
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from baselines.B7.data import read_manifest, load_record, WindowDataset
from baselines.B7.prepare import prepare, public_index
from baselines.B7.public_qc import PUBLIC_QC_VERSION, inspect_public, deduplicate, flat_samples
from scripts.b7_public_qc_v2 import rebase_long_manifest
from baselines.B7.tests.test_b7 import make_manifest
from baselines.B7.losses import flow_loss
from baselines.B7 import train as trainer


def waveform(seed=1):
    y = np.random.default_rng(seed).normal(0, 100, (12, 5000)).astype('float32')
    y[2] = y[1] - y[0]
    y[3] = -(y[0] + y[1]) / 2
    y[4] = y[0] - y[1] / 2
    y[5] = y[1] - y[0] / 2
    return y


class PublicQCTests(unittest.TestCase):
    def test_new_public_run_refuses_legacy_cache_before_allocating_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_manifest(root)
            value = json.loads(manifest.read_text())
            for row in value['records']:
                row['source'] = 'chapman'
            manifest.write_text(json.dumps(value))
            args = Namespace(config=str(Path(__file__).resolve().parents[3]/'configs/b7_public10.json'),
                             manifest=str(manifest), resume=False, init_checkpoint=None,
                             output=str(root/'run'), device='cpu')
            with self.assertRaisesRegex(ValueError, 'requires QC-v2'):
                trainer.train(args)
            self.assertFalse((root/'run').exists())

    def test_masked_targets_do_not_enter_flow_or_waveform_loss(self):
        class Velocity(torch.nn.Module):
            def forward(self, state, time, anchor, metadata=None):
                return state * .2
        target = torch.from_numpy(waveform()[[1,6,7,8,9,10,11]][None])
        batch = {'anchor': torch.from_numpy(waveform()[:1][None]), 'target': target,
                 'lead_mask': torch.tensor([[True,False,True,True,True,True,True]])}
        noise, time = torch.zeros_like(target), torch.tensor([.5])
        a, parts_a = flow_loss(Velocity(), batch, noise=noise, time=time)
        batch['target'] = target.clone()
        batch['target'][:,1] = 1e9
        b, parts_b = flow_loss(Velocity(), batch, noise=noise, time=time)
        self.assertEqual(a.item(), b.item())
        for name in ('fm','mae','pcc'):
            self.assertEqual(parts_a[name].item(), parts_b[name].item())

    def test_rebase_reuses_waveforms_but_changes_only_scale_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = make_manifest(root)
            value = json.loads(old.read_text())
            for row in value['records']:
                row['source'] = 'ptbdb'
            old.write_text(json.dumps(value))
            before = {row['path']: hashlib.sha256((root/row['path']).read_bytes()).hexdigest()
                      for row in value['records']}
            short = root/'short.json'
            clean = json.loads(old.read_text())
            clean['public_qc_version'] = PUBLIC_QC_VERSION
            clean['scales_uV'] = [999.] * 12
            short.write_text(json.dumps(clean))
            dest = root/'rebased/manifest.json'
            rebase_long_manifest(old, short, dest)
            rebase_long_manifest(old, short, dest)
            actual = read_manifest(dest)
            self.assertEqual(actual['scales_uV'], [999.] * 12)
            self.assertFalse(actual['long_waveform_qc_changed'])
            for row in actual['records']:
                self.assertEqual(hashlib.sha256(Path(row['resolved_path']).read_bytes()).hexdigest(),
                                 before[Path(row['resolved_path']).name])
            clean.pop('public_qc_version')
            short.write_text(json.dumps(clean))
            with self.assertRaisesRegex(ValueError, 'corrected public10'):
                rebase_long_manifest(old, short, root/'bad.json')

    def test_constants_partial_flats_discontinuities_and_input(self):
        y = waveform()
        y[6] = 0
        y[7, 100:600] = 31
        y[8, 0:2] = [-32768, 32767]
        mask, ok, audit = inspect_public(y[:1], y, np.ones(12, bool), 500)
        self.assertTrue(ok)
        self.assertFalse(mask[6:9].any())
        self.assertTrue(mask[9:].all())
        self.assertEqual(flat_samples(np.array([0, 1, 1, 1, 2])), 3)
        self.assertEqual(audit['flags']['exact_flat_ge_1s'], [6, 7])
        y[0] = 0
        self.assertFalse(inspect_public(y[:1], y, mask, 500)[1])

    def test_review_flag_alone_does_not_remove_amplitude(self):
        y = waveform()
        y[9, 4] = 12000
        mask, _, audit = inspect_public(y[:1], y, np.ones(12, bool), 500)
        self.assertTrue(mask[9])
        self.assertIn(9, audit['flags']['abs_gt_10mV_review'])
        self.assertIn(9, audit['flags']['jump_gt_5mV_review'])

    def test_derived_limb_mismatch_mask_only(self):
        y = waveform()
        y[4] += 100
        mask, _, _ = inspect_public(y[:1], y, np.ones(12, bool), 500)
        self.assertFalse(mask[4])
        self.assertTrue(mask[1])
        self.assertTrue(mask[6:].all())

    def test_heldout_priority_deterministic_and_idempotent(self):
        records = [dict(record_id=str(i), split=split, source='chapman',
                        public_qc={'waveform_sha256': 'same'})
                   for i, split in enumerate(['train', 'validation', 'test'])]
        result, audit = deduplicate(records)
        self.assertEqual(result, [records[2]])
        self.assertEqual(deduplicate(list(reversed(records)))[0], result)
        self.assertEqual(len(audit), 1)
        self.assertEqual(deduplicate(result)[1], [])

    def test_ptb_sex_encoding_and_unknown_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / 'ptbxl_database.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename_hr','patient_id','strat_fold','sex'])
                writer.writeheader()
                for i, sex in enumerate(['0','1','']):
                    writer.writerow(dict(filename_hr=str(i), patient_id=i, strat_fold=1, sex=sex))
            rows = public_index('ptbxl', root)
            self.assertEqual([r['metadata']['sex'] for r in rows], [0,1,2])
            self.assertEqual([r['metadata']['field_mask'][1] for r in rows], [True,True,False])

    def test_end_to_end_masks_dedup_train_scales_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i, split in enumerate(['train','validation','test','train','train']):
                y = waveform(1 if i < 2 else i)
                if i == 3:
                    y[6] = 0
                path = root / f'{i}.npz'
                np.savez(path, target_uV=y)
                rows.append(dict(record_id=str(i), target_record_id=str(i), subject_id=str(i),
                                 source='chapman', group='chapman', split=split, path=str(path),
                                 input_reliable=True))
            index = root / 'index.json'
            index.write_text(json.dumps(rows))
            output = root / 'cache'
            args = Namespace(index=str(index), output=str(output), public_source=None,
                             root=None, scale_manifest=None, resume=False)
            prepare(args)
            manifest = read_manifest(output / 'manifest.json')
            self.assertEqual(manifest['public_qc_version'], PUBLIC_QC_VERSION)
            self.assertEqual([r['record_id'] for r in manifest['records']], ['1','2','3','4'])
            train = manifest['records'][-2]
            self.assertFalse(load_record(train)[2][6])
            self.assertFalse(WindowDataset(manifest, 10)[0]['lead_mask'][1])
            # Recompute from the retained training record only.
            y = load_record(train)[1]
            widths = [np.diff(np.percentile(load_record(r)[1][1], [5,95]))[0]
                      for r in manifest['records'] if r['split'] == 'train']
            self.assertAlmostEqual(manifest['scales_uV'][1], max(float(np.median(widths)),100))
            before = hashlib.sha256((output/'records/0000003.npz').read_bytes()).hexdigest()
            (output/'manifest.json').unlink()
            (output/'records/0000000.tmp').write_bytes(b'interrupted np.savez')
            (output/'manifest.pending.json').write_text('incomplete')
            args.resume = True
            prepare(args)
            self.assertEqual(before, hashlib.sha256((output/'records/0000003.npz').read_bytes()).hexdigest())
            (output/'manifest.json').unlink()
            (output/'prepare_identity.json').unlink()
            with self.assertRaisesRegex(ValueError, 'identity/QC changed'):
                prepare(args)


if __name__ == '__main__':
    unittest.main()
