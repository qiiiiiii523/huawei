import unittest
import torch
from baselines.B8.losses import regression_loss
from baselines.B8.model import B8M1TimeRegressor

class B8Test(unittest.TestCase):
    def test_time_only_direct_regression_shape_and_backward(self):
        model = B8M1TimeRegressor(d_model=32, blocks=1, heads=4, ffn=64, dropout=0.)
        self.assertTrue(all(block.attention.batch_first for block in model.blocks))
        output = model(torch.randn(1, 1, 5000))
        self.assertEqual(output.shape, (1, 7, 5000))
        output.square().mean().backward()

    def test_qc_masked_missing11_loss(self):
        pred = torch.randn(2, 7, 5000, requires_grad=True); target = torch.randn_like(pred)
        lead_mask = torch.ones(2, 7, dtype=torch.bool); lead_mask[:, 3] = False
        loss, parts = regression_loss(pred, target, lead_mask, torch.ones(2, 5000, dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(parts["pcc_valid_leads"].item(), 12)
        loss.backward()

    def test_model_rejects_wrong_window(self):
        with self.assertRaises(ValueError): B8M1TimeRegressor()(torch.randn(1, 1, 4999))

if __name__ == "__main__": unittest.main()
