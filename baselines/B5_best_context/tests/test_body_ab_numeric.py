"""Small float roundoff is allowed; preprocessing/identity/mask changes still fail."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.B5_best.common import read
from baselines.B5_best_context.adapters import HistoryRecord
from baselines.B5_best_context.body_ab_prepare import numerical_comparison, verify_overlay_history, verify_overlay_features


class OverlayNumericTests(unittest.TestCase):
    def test_small_float32_roundoff_allowed_without_modifying_values(self):
        a = np.full((6, 100), 250., np.float32)
        b = a.copy(); b[0, 0] = np.nextafter(b[0, 0], np.float32(1000))
        before = b.copy()
        mask = np.ones(a.shape, bool)
        h = lambda x: HistoryRecord('body_scale_d6', 'same_record', x, mask)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'check.json'
            r = verify_overlay_history(h(a), h(a), h(b), p, dict(pair_id='pair', subject_id='subject'))
            self.assertFalse(r['waveform_uV']['exact'])
            self.assertTrue(r['waveform_passed'])
            fa = (np.ones((2,300), np.float32), np.ones(27, np.float32), True)
            fb = (fa[0]+np.float32(1e-6), fa[1]+np.float32(1e-6), True)
            verify_overlay_features(fa, fb, r, p)
            self.assertTrue(read(p)['features_passed'])
            np.testing.assert_array_equal(b, before)

    def test_real_voltage_mask_shape_or_identity_changes_fail_with_saved_diagnostics(self):
        a = np.ones((6,100), np.float32)
        mask = np.ones(a.shape, bool)
        original = HistoryRecord('body_scale_d6', 'same', a, mask)
        b = a.copy(); b[1, 0] += 1.
        badmask = mask.copy(); badmask[0, 0] = False
        cases = [HistoryRecord('body_scale_d6', 'same', b, mask),
                 HistoryRecord('body_scale_d6', 'same', a, badmask),
                 HistoryRecord('body_scale_d6', 'other', a, mask),
                 HistoryRecord('body_scale_d6', 'same', a[:,:50], mask[:,:50])]
        with tempfile.TemporaryDirectory() as tmp:
            for i, other in enumerate(cases):
                p=Path(tmp)/f'{i}.json'
                with self.assertRaisesRegex(ValueError, 'pair=pair, subject=subject'):
                    verify_overlay_history(original, original, other, p, dict(pair_id='pair', subject_id='subject'))
                self.assertFalse(read(p)['waveform_passed'])

    def test_nonfinite_features_and_changed_peak_availability_fail(self):
        a = (np.ones((2,300), np.float32), np.ones(27, np.float32), True)
        for b in ((a[0]*2, a[1], True), (a[0], a[1], False), (a[0]*np.nan, a[1], True)):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'features.json'
                with self.assertRaisesRegex(ValueError, 'beyond roundoff'):
                    verify_overlay_features(a, b, dict(pair_id='pair', subject_id='subject'), p)
                self.assertFalse(read(p)['features_passed'])
        self.assertFalse(numerical_comparison([1], [1,2], atol=.001, rtol=1e-6)['within_tolerance'])


if __name__ == '__main__':
    unittest.main()
