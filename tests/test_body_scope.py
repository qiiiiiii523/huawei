"""Synthetic numerical/route checks; no weights, CUDA or patient data required."""
import unittest
import numpy as np
from baselines.B5_final.body_scope import transform
from baselines.B5_best_context.fir import B, C, HistoricalFIR, apply_postprocess


class BodyScopeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.base = rng.normal(size=(12, 401)).astype(np.float32)
        self.anchor = rng.normal(size=(1, 401)).astype(np.float32)
        self.corrected = (self.base[1] + rng.normal(size=401) * .1).astype(np.float32)

    def test_projection_matches_existing_implementation(self):
        expected, _ = apply_postprocess(self.base, self.anchor, HistoricalFIR(None, 0., False, "test"))
        result = transform(self.base, self.anchor, self.corrected, projection=True, scope="none")
        np.testing.assert_array_equal(result, expected.astype(np.float32))

    def test_c0_exact_and_inputs_immutable(self):
        before = self.base.copy()
        result = transform(self.base, self.anchor, self.corrected, projection=False, scope="II")
        np.testing.assert_array_equal(result[1], self.corrected)
        np.testing.assert_array_equal(result[[0, *range(2, 12)]], before[[0, *range(2, 12)]])
        np.testing.assert_array_equal(self.base, before)

    def test_propagation_uses_original_delta_with_and_without_projection(self):
        delta = self.corrected.astype(float) - self.base[1]
        for projection in (False, True):
            start = transform(self.base, self.anchor, self.corrected, projection=projection, scope="none")
            result = transform(self.base, self.anchor, self.corrected, projection=projection, scope="five")
            np.testing.assert_allclose(result[1:6], start[1:6].astype(float) + B * delta, atol=3e-7)
            np.testing.assert_array_equal(result[[0, *range(6, 12)]], self.base[[0, *range(6, 12)]])

    def test_projected_five_preserves_relations_but_ii_only_breaks_them(self):
        five = transform(self.base, self.anchor, self.corrected, projection=True, scope="five")
        ii = transform(self.base, self.anchor, self.corrected, projection=True, scope="II")
        np.testing.assert_allclose(five[1:6], B * five[1] + C * self.anchor, atol=4e-7)
        self.assertGreater(np.max(np.abs(ii[2] - (ii[1] - self.anchor[0]))), .01)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            transform(self.base, self.anchor, self.corrected, projection=False, scope="chest")
        bad = self.corrected.copy(); bad[0] = np.nan
        with self.assertRaises(ValueError):
            transform(self.base, self.anchor, bad, projection=True, scope="five")


if __name__ == "__main__":
    unittest.main()
