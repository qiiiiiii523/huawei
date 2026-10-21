import unittest
import numpy as np
import torch

from baselines.B5_best.slow import SlowCNN
from baselines.B5_cnn120.model import FullSlowCNN, visible_inputs, inputs


class CNN120Tests(unittest.TestCase):
    def test_matched_controls_are_identical_at_identity_initialization(self):
        torch.manual_seed(43)
        source = SlowCNN()
        torch.manual_seed(43)
        local = FullSlowCNN("L120")
        local.initialize_from_s50(source.state_dict())
        torch.manual_seed(43)
        long = FullSlowCNN("C120")
        long.initialize_from_s50(source.state_dict())
        batch = {"anchor50": torch.randn(2, 1, 6000), "base_slow": torch.randn(2, 11, 6000),
                 "numeric": torch.zeros(2, 3), "sex": torch.zeros(2, dtype=torch.long),
                 "field_mask": torch.zeros(2, 4, dtype=torch.bool), "age_topcoded": torch.zeros(2, 1)}
        self.assertTrue(torch.equal(local(**inputs(batch)), long(**inputs(batch))))

    def test_long_arm_reaches_distant_input_after_one_nonzero_long_block(self):
        torch.manual_seed(3)
        source = SlowCNN()
        long = FullSlowCNN("C120")
        long.initialize_from_s50(source.state_dict())
        with torch.no_grad():
            long.extra[0].two.conv.weight.fill_(.01)
        batch = {"anchor50": torch.randn(1, 1, 6000, requires_grad=True), "base_slow": torch.zeros(1, 11, 6000),
                 "numeric": torch.zeros(1, 3), "sex": torch.zeros(1, dtype=torch.long),
                 "field_mask": torch.zeros(1, 4, dtype=torch.bool), "age_topcoded": torch.zeros(1, 1)}
        long(**inputs(batch))[..., 3000].sum().backward()
        self.assertGreater(float(batch["anchor50"].grad[..., 3000 - 1024].abs().sum()), 0.)

    def test_visible_input_has_no_target_leads(self):
        values = visible_inputs(np.ones(60000, np.float32), 2.,
                                {"numeric": np.zeros(3, np.float32), "sex": np.array(0),
                                 "field_mask": np.zeros(4, bool), "age_topcoded": np.zeros(1, np.float32)})
        self.assertEqual(values["anchor50"].shape, (1, 6000))
        self.assertNotIn("target", values)


if __name__ == "__main__":
    unittest.main()
