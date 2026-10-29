import unittest
import torch
from baselines.B8.tcn120 import CachedPredictionTCN120, SAMPLES

class TCN120Tests(unittest.TestCase):
    def test_zero_head_is_exact_identity_and_demeaned(self):
        model = CachedPredictionTCN120(width=8, dilations=(1, 2)).eval()
        source = torch.randn(1, 7, SAMPLES)
        with torch.no_grad(): delta = model(source)
        self.assertEqual(delta.shape, source.shape)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(delta.mean(-1), torch.zeros(1, 7)))

    def test_nonzero_curve_has_no_per_lead_record_mean(self):
        model = CachedPredictionTCN120(width=8, dilations=(1, 2)).eval()
        torch.nn.init.normal_(model.head.weight)
        with torch.no_grad(): delta = model(torch.randn(1, 7, SAMPLES))
        self.assertLess(float(delta.mean(-1).abs().max()), 2e-6)

if __name__ == "__main__": unittest.main()
