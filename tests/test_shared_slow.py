import unittest
import numpy as np
from baselines.B5_final.shared_slow import pca_basis, masked_oracle, local_features


class SharedSlowTests(unittest.TestCase):
    def test_train_basis_recovers_shared_structure_with_missing_leads(self):
        rng = np.random.default_rng(42)
        q, _ = np.linalg.qr(rng.normal(size=(6, 2)))
        train = [q @ rng.normal(size=(2, 100)) for _ in range(3)]
        basis, values = pca_basis(train, np.ones(3))
        self.assertGreater(values[:2].sum() / values.sum(), .999999)
        heldout = q @ rng.normal(size=(2, 100))
        mask = np.array([True, True, False, True, True, False])
        observed, reconstruction = masked_oracle(heldout, basis[:, :2], mask)
        np.testing.assert_allclose(observed, reconstruction, atol=1e-10)
        self.assertIsNone(masked_oracle(heldout, basis[:, :3], np.array([True, False, False, False, False, False])))

    def test_invalid_pca_data_not_zero_filled(self):
        x = np.ones((6, 20)); x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            pca_basis([x], [1.])

    def test_probe_feature_locality_and_visible_only_interface(self):
        rng = np.random.default_rng(1)
        anchor = rng.normal(size=60000)
        meta = {"numeric": np.zeros(3), "sex": np.array(0),
                "field_mask": np.ones(4), "age_topcoded": np.zeros(1)}
        first = local_features(anchor, meta, 1.)
        changed = anchor.copy(); changed[50000:] += 100
        second = local_features(changed, meta, 1.)
        self.assertEqual(first.shape[0], 600)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertFalse(np.array_equal(first[-1], second[-1]))
        np.testing.assert_array_equal(first[:, -1], np.ones(600))


if __name__ == "__main__":
    unittest.main()
