"""Git newline portability must not permit changed pair identities or quality."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from ecg12gen.cache_metadata import metadata_sha256_matches


class MetadataHashTests(unittest.TestCase):
    def test_B_store_accepts_only_newline_change_but_keeps_real_pair_and_array_guards(self):
        import csv
        from io import StringIO
        from baselines.B5_best.common import write, sha
        from baselines.B5_best_context.body_history import B_POLICY
        module = ModuleType('ecg12gen.body_scale_detrend')
        module.FILTER_SPEC = B_POLICY['filter']
        with patch.dict('sys.modules', {'ecg12gen.body_scale_detrend': module}):
            from ecg12gen.body_scale_ablation import BodyScaleBContextStore, OVERLAY_PROTOCOL, OVERLAY_METADATA
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, overlay = root/'canonical', root/'B'
            base.mkdir(); overlay.mkdir()
            row = dict(array_index='0', subject_id='HNU1', split='train', input_type='body_scale_d6',
                       input_record_id='history1', context_window_index='0', start_sample_500hz='0',
                       valid_length='5000', valid_sample_count='30000', context_window_valid='true',
                       expected_context_window_count='1', input_record_baseline_uV='0|0|0|0|0|0')
            def csv_bytes(record):
                buf = StringIO(newline='')
                w = csv.DictWriter(buf, fieldnames=list(record)); w.writeheader(); w.writerow(record)
                return buf.getvalue().encode('utf-8-sig')
            context = csv_bytes(row)
            files = dict(target_metadata=base/'task2_window_metadata.csv', context_metadata=base/'task2_context_window_metadata.csv',
                         pair_manifest=root/'pairs.csv', subject_split=root/'split.csv', device_qc=root/'qc.csv')
            expected = {}
            for key, path in files.items():
                data = context if key == 'context_metadata' else b'key,value\r\nA,B\r\n'
                expected[key] = hashlib.sha256(data).hexdigest()
                path.write_bytes(data.replace(b'\r\n', b'\n'))
            overlay_meta = csv_bytes({**row, 'canonical_array_index': '0'})
            (overlay/OVERLAY_METADATA).write_bytes(overlay_meta.replace(b'\r\n', b'\n'))
            np.save(overlay/'body_scale_train_context_B.npy', np.zeros((1,6,5000), np.float32))
            np.save(overlay/'body_scale_train_context_valid_mask_B.npy', np.ones((1,6,5000), bool))
            hashes = {p.name: sha(p) for p in overlay.iterdir()}
            hashes[OVERLAY_METADATA] = hashlib.sha256(overlay_meta).hexdigest()
            write(overlay/'body_scale_ablation_manifest.json', dict(protocol=OVERLAY_PROTOCOL, filter=B_POLICY['filter'],
                  overlay_files_sha256=hashes, canonical_metadata_sha256=expected))
            paths = dict(task2_output=base, task2_body_scale_ablation=overlay, task2_pair_manifest_csv=files['pair_manifest'],
                         subject_split_csv=files['subject_split'], device_interpretation_qc_csv=files['device_qc'])
            cfg = SimpleNamespace(path=lambda key: paths[key])
            store = BodyScaleBContextStore(cfg, 'train')
            self.assertEqual(store.context('history1', 'HNU1').raw_uV.shape, (1,6,5000))
            store.close()
            files['pair_manifest'].write_bytes(b'key,value\nA,changed\n')
            with self.assertRaisesRegex(ValueError, 'canonical pair_manifest'):
                BodyScaleBContextStore(cfg, 'train')
            files['pair_manifest'].write_bytes(b'key,value\nA,B\n')
            np.save(overlay/'body_scale_train_context_B.npy', np.ones((1,6,5000), np.float32))
            with self.assertRaisesRegex(ValueError, 'array changed'):
                BodyScaleBContextStore(cfg, 'train')

    def test_CRLF_and_LF_match_in_both_directions_without_file_mutation(self):
        crlf = b'\xef\xbb\xbfpair_id,subject_id,split\r\nA,HNU1,train\r\nB,HNU2,validation\r\n'
        lf = crlf.replace(b'\r\n', b'\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pairs.csv'
            for actual, original in ((crlf, crlf), (lf, crlf), (crlf, lf), (lf, lf)):
                path.write_bytes(actual)
                self.assertTrue(metadata_sha256_matches(path, hashlib.sha256(original).hexdigest()))
                self.assertEqual(path.read_bytes(), actual)

    def test_changed_patient_split_row_order_and_BOM_do_not_pass(self):
        original = b'\xef\xbb\xbfpair_id,subject_id,split\r\nA,HNU1,train\r\nB,HNU2,validation\r\n'
        expected = hashlib.sha256(original).hexdigest()
        changes = [original.replace(b'HNU1', b'HNU9'), original.replace(b'A,HNU1,train', b'A,HNU1,validation'),
                   original.replace(b'A,HNU1,train\r\nB,HNU2,validation', b'B,HNU2,validation\r\nA,HNU1,train'),
                   original[3:], original.replace(b'HNU1', b' HNU1'), original + b'A,HNU1,train\r\n']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pairs.csv'
            for actual in changes:
                path.write_bytes(actual.replace(b'\r\n', b'\n'))
                self.assertFalse(metadata_sha256_matches(path, expected))
            self.assertFalse(metadata_sha256_matches(path, None))


if __name__ == '__main__':
    unittest.main()
