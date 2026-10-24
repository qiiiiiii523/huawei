from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts import pack_b7t_diagnosis as pack


class DiagnosticPackTests(unittest.TestCase):
    def manifest(self, root):
        root.mkdir(parents=True)
        t = np.linspace(0, 5, 128)
        i, ii = np.sin(t), np.cos(t)
        target = np.stack([i, ii, ii-i, -(i+ii)/2, i-ii/2, ii-i/2] + [i*(j+1) for j in range(6)])
        rows = []
        for split, count in [('train', 3), ('validation', 5), ('test', 1)]:
            for j in range(count):
                name = f'{split}{j}'
                np.savez(root / (name+'.npz'), anchor_uV=target[:1], target_uV=target,
                         lead_mask=np.ones(12, dtype=bool))
                rows.append(dict(record_id=name, subject_id=name, target_record_id=name,
                    group='task1', source='synthetic', split=split, input_reliable=True, path=name+'.npz'))
        value = {'scales_uV': [100.]*12, 'records': rows, 'sample_rate': 500}
        pack.write(root/'manifest.json', value)
        return root/'manifest.json', value

    def test_selection_includes_train_and_low_middle_high_validation_without_test(self):
        with tempfile.TemporaryDirectory() as temp:
            _, manifest = self.manifest(Path(temp)/'cache')
            selected = pack.select_samples(manifest, {f'validation{j}': j/5 for j in range(5)})
            self.assertEqual([r['record_id'] for r in selected],
                             ['train0', 'train1', 'validation0', 'validation2', 'validation4'])
            self.assertTrue(all(r['split'] != 'test' for r in selected))

    def test_sample_manifest_is_portable_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _ = self.manifest(root/'cache')
            source_hash = hashlib.sha256((path.parent/'train0.npz').read_bytes()).hexdigest()
            pack.collect_samples(path, root/'bundle')
            subset = pack.read(root/'bundle/manifest_subset.json')
            for row in subset['records']:
                self.assertTrue((root/'bundle'/row['path']).is_file())
            stats = pack.read(root/'bundle/sample_stats.json')[0]['stats']
            self.assertEqual(stats['anchor_I_max_abs_difference_uV'], 0.)
            self.assertTrue(all(v < 1e-10 for v in stats['derived_limb_residual_rms_uV'].values()))
            self.assertEqual(hashlib.sha256((path.parent/'train0.npz').read_bytes()).hexdigest(), source_hash)

    def test_thin_EMA_export_preserves_inference_fields_but_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'best.pt'
            value = {k: {} for k in ('model_config', 'sampling', 'loss', 'identity')}
            value.update(architecture_id='test', format_version=1, generated_leads=('II',),
                ema={'weight': torch.ones(2)}, model={'weight': torch.zeros(2)}, optimizer={}, rng={},
                scales_uV=[100.]*12, stage='huawei40', update=1000, best_r=.48)
            torch.save(value, source)
            before = source.read_bytes()
            pack.export_ema(source, root/'bundle/B.pt')
            thin = torch.load(root/'bundle/B.pt', weights_only=True)
            self.assertTrue(torch.equal(thin['ema']['weight'], value['ema']['weight']))
            self.assertFalse(thin['resume_supported'])
            for key in ('model', 'optimizer', 'rng', 'identity'):
                self.assertNotIn(key, thin)
            self.assertEqual(source.read_bytes(), before)

    def test_end_to_end_collection_contains_four_weights_samples_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'run'
            for stage in ('P', 'Q', 'A', 'B'):
                directory = run/stage
                directory.mkdir(parents=True)
                torch.save(dict(architecture_id='test', format_version=1, generated_leads=('II',),
                    model_config={}, ema={'weight': torch.ones(2)}, sampling={}, loss={},
                    scales_uV=[100.]*12, stage=stage, update=1000, best_r=.48), directory/'best_r.pt')
                pack.write(directory/'completed.json', {'completed': True})
            for dataset in ('public10', 'ptb40', 'huawei'):
                self.manifest(root/'cache/data'/dataset)
            args = Namespace(output=root/'bundle', run=run, new_run=root/'new_run',
                repo=root/'absent_repo', new_repo=root/'absent_new_repo', metadata=root/'metadata',
                cache_run=root/'cache')
            pack.collect(args)
            report = pack.read(args.output/'bundle_report.json')
            self.assertEqual(report['weights'], ['P', 'Q', 'A', 'B'])
            self.assertEqual(len(list((args.output/'weights').glob('*.pt'))), 4)
            self.assertTrue((args.output/'SHA256SUMS').is_file())
            self.assertTrue(all(report['data'][name]['selected'] == 5 for name in report['data']))
            with self.assertRaises(FileExistsError):
                pack.collect(args)


if __name__ == '__main__':
    unittest.main()
