"""Model-only tests: no competition data, training, or random masks required."""
from __future__ import annotations

import torch
import unittest

from ecg12gen.contracts import ContractError
from ecg12gen.b3_model import B3Model


class B3ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.model = B3Model(fusion_mode="film_gated_residual").eval()

    @torch.no_grad()
    def test_task1_shape_parameter_budget_and_complete_generation(self) -> None:
        output = B3Model(fusion_mode="none")(torch.randn(1, 1, 5000))
        self.assertEqual(output.shape, (1, 12, 5000))
        self.assertGreater(self.model.parameter_count, 600_000)

    @torch.no_grad()
    def test_task2_and_canonicalized_input_are_supported(self) -> None:
        anchor = torch.randn(1, 1, 5000)
        mask = torch.tensor([[True] * 6])
        machine_output = self.model(anchor, context_ecg=torch.randn(1, 6, 5000),
                                    context_source_type="ecg_machine_d6", context_lead_mask=mask)
        body_output = self.model(anchor, context_ecg=torch.randn(1, 6, 5000),
                                 context_source_type="body_scale_d6", context_lead_mask=mask)
        self.assertEqual(machine_output.shape, (1, 12, 5000))
        self.assertEqual(body_output.shape, (1, 12, 5000))

    def test_mask_must_be_complementary(self) -> None:
        anchor = torch.randn(1, 1, 5000)
        with self.assertRaisesRegex(ContractError, "lead mask"):
            self.model(anchor, context_ecg=torch.randn(1, 6, 5000),
                       context_source_type="ecg_machine_d6",
                       context_lead_mask=torch.tensor([[True, False, False, False, False, False]]))
