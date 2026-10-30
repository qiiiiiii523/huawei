import unittest

import numpy as np
import torch

from baselines.B5_final.trend_corrector.model import TrendCorrector, compose_d12, make_features


class TrendCorrectorTests(unittest.TestCase):
    def test_shape_and_zero_mean_invariant(self):
        prediction = np.zeros((12, 60000), dtype=np.float32)
        prediction[1:, :500] = 3.0
        model = TrendCorrector(np.ones(12, dtype=np.float32))
        with torch.no_grad():
            correction, controls = model(make_features(prediction, np.ones(12, dtype=np.float32))[None])
        self.assertEqual(tuple(correction.shape), (1, 7, 60000))
        self.assertEqual(tuple(controls.shape), (1, 7, 480))
        self.assertLess(float(correction.abs().max()), 1e-8)  # zero-initialized heads
        self.assertLess(float(controls.abs().max()), 1e-8)

    def test_limb_leads_are_rebuilt_from_final_i_and_ii(self):
        i = torch.full((1, 1, 3), 10.)
        generated = torch.zeros((1, 7, 3)); generated[:, 0] = 4.
        full = compose_d12(i, generated)
        self.assertTrue(torch.equal(full[:, 0], i[:, 0]))
        self.assertTrue(torch.equal(full[:, 1], generated[:, 0]))
        self.assertTrue(torch.equal(full[:, 2], generated[:, 0] - i[:, 0]))
        self.assertTrue(torch.equal(full[:, 3], -(i[:, 0] + generated[:, 0]) / 2))
        self.assertTrue(torch.equal(full[:, 4], i[:, 0] - generated[:, 0] / 2))
        self.assertTrue(torch.equal(full[:, 5], generated[:, 0] - i[:, 0] / 2))


if __name__ == "__main__":
    unittest.main()
