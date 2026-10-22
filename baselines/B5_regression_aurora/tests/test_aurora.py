import unittest

import torch

from baselines.B5.config import ModelConfig
from baselines.B5_regression_aurora.losses import aurora_loss
from baselines.B5_regression_aurora.model import AuroraRegression, derive_limb_leads


class AuroraRegressionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.scale = torch.arange(1, 13, dtype=torch.float32) * 100
        self.config = ModelConfig(channels=(8, 16, 32), time_dim=8, metadata_dim=8,
                                  metadata_dropout=0., metadata_field_dropout=0., dilations=(1, 2))
        self.batch = {"anchor": torch.randn(2, 1, 64), "target": torch.randn(2, 12, 64),
                      "quality_mask": torch.ones(2, 12, dtype=torch.bool),
                      "numeric": torch.zeros(2, 3), "sex": torch.zeros(2, dtype=torch.long),
                      "field_mask": torch.ones(2, 4, dtype=torch.bool),
                      "age_topcoded": torch.zeros(2, 1, dtype=torch.bool)}

    def test_vector_identities_are_applied_in_raw_uv(self):
        lead_i, lead_ii = torch.randn(2, 1, 32), torch.randn(2, 1, 32)
        limb = derive_limb_leads(lead_i, lead_ii, self.scale)
        self.assertTrue(torch.equal(limb[:, :1], lead_i))
        self.assertTrue(torch.equal(limb[:, 1:2], lead_ii))
        raw = limb * self.scale[:6].reshape(1, 6, 1)
        self.assertTrue(torch.allclose(raw[:, 2], raw[:, 1] - raw[:, 0]))
        self.assertTrue(torch.allclose(raw[:, 3], -(raw[:, 0] + raw[:, 1]) / 2))
        self.assertTrue(torch.allclose(raw[:, 4], (raw[:, 0] - raw[:, 2]) / 2))
        self.assertTrue(torch.allclose(raw[:, 5], (raw[:, 1] + raw[:, 2]) / 2))

    def test_forward_and_full_objective(self):
        model = AuroraRegression(self.config)
        output = model(self.batch["anchor"], self.batch["numeric"], self.batch["sex"],
                       self.batch["field_mask"], self.batch["age_topcoded"], self.scale)
        self.assertEqual(tuple(output.full12.shape), (2, 12, 64))
        self.assertTrue(torch.equal(output.full12[:, :1], self.batch["anchor"]))
        loss, parts = aurora_loss(model, self.batch,
            {"generation": 1., "pcc": .1, "source_reconstruction": .5,
             "representation_consistency": .1, "huber_delta": 1.}, self.scale)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"generation", "pcc", "source_reconstruction",
                                      "representation_consistency"})
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
