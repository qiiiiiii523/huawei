import unittest
import numpy as np
from baselines.B5_best_context.body_multitemplate import cluster_bank, mix_templates, K
from baselines.B5_best_context.body_features import relation_features, BEAT16_ENCODING
from baselines.B5_best_context.adapters import HistoryRecord


class MultiTemplateTests(unittest.TestCase):
    def make_beats(self):
        t = np.linspace(-1, 1, 300)
        a = np.exp(-((t + .15) / .1) ** 2) - .4 * np.exp(-((t - .1) / .08) ** 2)
        b = -np.exp(-((t - .05) / .13) ** 2) + .3 * np.exp(-((t + .2) / .09) ** 2)
        beats = np.array([[a, 2 * a]] * 12 + [[b, -b]] * 12, np.float32)
        return beats, a, b

    def test_grouping_uses_I_only_and_is_deterministic(self):
        beats, _, _ = self.make_beats()
        templates, counts, dispersion = cluster_bank(beats)
        changed = beats.copy(); changed[:, 1] *= -100
        second, counts2, dispersion2 = cluster_bank(changed)
        np.testing.assert_array_equal(templates[:, 0], second[:, 0])
        np.testing.assert_array_equal(counts, counts2)
        np.testing.assert_array_equal(dispersion, dispersion2)
        self.assertEqual(int(counts.sum()), len(beats))
        np.testing.assert_array_equal(cluster_bank(beats)[0], templates)

    def test_current_I_selects_different_history_groups(self):
        beats, a, b = self.make_beats()
        templates, counts, dispersion = cluster_bank(beats)
        q = np.array([a] * 6 + [b] * 6, np.float32)
        mixed, weights = mix_templates(q, np.ones(12, bool), templates, counts, dispersion)
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-7)
        self.assertNotEqual(int(weights[0].argmax()), int(weights[-1].argmax()))
        self.assertGreater(np.linalg.norm(mixed[0] - mixed[-1]), 1)

    def test_empty_and_missing_query_fallbacks(self):
        templates, counts, dispersion = cluster_bank(np.empty((0, 2, 300), np.float32))
        mixed, weights = mix_templates(np.zeros((12, 300)), np.zeros(12, bool), templates, counts, dispersion)
        self.assertFalse(mixed.any()); self.assertFalse(weights.any())
        beats, _, _ = self.make_beats()
        templates, counts, dispersion = cluster_bank(beats)
        _, weights = mix_templates(np.zeros((12, 300)), np.zeros(12, bool), templates, counts, dispersion)
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_array_equal(weights, np.repeat(weights[:1], 12, axis=0))

    def test_optional_bank_preserves_original_single_features(self):
        t = np.arange(10000)
        signal = sum(np.exp(-((t - peak) / 8) ** 2) for peak in range(250, 10000, 500))
        raw = np.array([signal, 2 * signal, signal, -.5 * signal, .4 * signal, .8 * signal], np.float32)
        history = HistoryRecord("body_scale_d6", "synthetic", raw, np.ones_like(raw, bool))
        original = relation_features(history, encoding=BEAT16_ENCODING)
        bank = {}
        exported = relation_features(history, encoding=BEAT16_ENCODING, beat_bank=bank)
        self.assertTrue(original[2])
        np.testing.assert_array_equal(original[0], exported[0])
        np.testing.assert_array_equal(original[1], exported[1])
        self.assertEqual(original[2], exported[2])
        self.assertGreaterEqual(len(bank["beats"]), 4)
        self.assertEqual(bank["beats"].shape[1:], (2, 300))


if __name__ == "__main__":
    unittest.main()
