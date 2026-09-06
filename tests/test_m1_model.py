"""Model-only tests: no competition data, training, or random masks required."""
from __future__ import annotations

import torch
import unittest

from ecg12gen.contracts import ContractError
from ecg12gen.m1_model import M1MaskedCNNLeadTimeTransformer


class M1ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.model = M1MaskedCNNLeadTimeTransformer().eval()

    @torch.no_grad()
    def test_task1_shape_parameter_budget_and_complete_generation(self) -> None:
        lead_mask = torch.tensor([[True] + [False] * 11])
        output = self.model(torch.randn(1, 1, 5000), lead_mask, ~lead_mask)
        self.assertEqual(output.shape, (1, 12, 5000))
        self.assertTrue(600_000 <= self.model.parameter_count <= 850_000)

    @torch.no_grad()
    def test_task2_and_canonicalized_input_are_supported(self) -> None:
        lead_mask = torch.tensor([[True] * 6 + [False] * 6])
        task2_output = self.model(torch.randn(1, 6, 5000), lead_mask, ~lead_mask)
        canonical_output = self.model(torch.randn(1, 12, 5000), lead_mask, ~lead_mask)
        self.assertEqual(task2_output.shape, canonical_output.shape)
        self.assertEqual(task2_output.shape, (1, 12, 5000))

    def test_mask_must_be_complementary(self) -> None:
        lead_mask = torch.tensor([[True] + [False] * 11])
        with self.assertRaisesRegex(ContractError, "complement"):
            self.model(torch.randn(1, 1, 5000), lead_mask, lead_mask)
