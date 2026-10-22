import unittest
import numpy as np
import torch

from baselines.B5_best.slow import SlowCNN
from baselines.B5_cnn120_res.model import ResidualSlowCNN, correct_base, visible_inputs, inputs


def batch(size=1):
    return {"anchor50": torch.randn(size, 1, 6000),
            "numeric": torch.zeros(size, 3), "sex": torch.zeros(size, dtype=torch.long),
            "field_mask": torch.zeros(size, 4, dtype=torch.bool),
            "age_topcoded": torch.zeros(size, 1)}


class CNN120ResidualTests(unittest.TestCase):
    def test_matched_controls_are_exact_zero_at_initialization(self):
        torch.manual_seed(43)
        source = SlowCNN()
        torch.manual_seed(43)
        local = ResidualSlowCNN("L120")
        local.initialize_from_s50(source.state_dict())
        torch.manual_seed(43)
        long = ResidualSlowCNN("C120")
        long.initialize_from_s50(source.state_dict())
        values = batch(2)
        local_delta, long_delta = local(**inputs(values)), long(**inputs(values))
        self.assertTrue(torch.equal(local_delta, long_delta))
        self.assertEqual(torch.count_nonzero(local_delta).item(), 0)

    def test_zero_residual_preserves_base_bit_for_bit(self):
        source = SlowCNN()
        model = ResidualSlowCNN("C120")
        model.initialize_from_s50(source.state_dict())
        base = torch.randn(1, 11, 6000)
        delta = model(**inputs(batch()))
        self.assertTrue(torch.equal(base, base + delta))
        base_uV = np.random.default_rng(7).normal(size=(12, 60000)).astype(np.float32)
        corrected = correct_base(base_uV, delta[0].detach().numpy(), np.ones(12))
        self.assertTrue(np.array_equal(base_uV, corrected))

    def test_long_arm_can_reach_beyond_original_s50_radius(self):
        torch.manual_seed(3)
        source = SlowCNN()
        long = ResidualSlowCNN("C120")
        long.initialize_from_s50(source.state_dict())
        with torch.no_grad():
            long.extra[0].two.conv.weight.fill_(.01)
            long.correction.weight.fill_(.01)
        values = batch()
        values["anchor50"].requires_grad_(True)
        long(**inputs(values))[..., 3000].sum().backward()
        self.assertGreater(float(values["anchor50"].grad[..., 3000 - 1024].abs().sum()), 0.)

    def test_visible_input_has_no_target_leads(self):
        values = visible_inputs(np.ones(60000, np.float32), 2.,
                                {"numeric": np.zeros(3, np.float32), "sex": np.array(0),
                                 "field_mask": np.zeros(4, bool), "age_topcoded": np.zeros(1, np.float32)})
        self.assertEqual(values["anchor50"].shape, (1, 6000))
        self.assertNotIn("target", values)


if __name__ == "__main__":
    unittest.main()
