'''Model-only tests for the formal Axial M1 contract.'''
from __future__ import annotations

import unittest
import torch

from ecg12gen.contracts import ContractError
from ecg12gen.m1_axial import M1AxialLeadTimeModel


class M1ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.anchor = torch.randn(1, 1, 5000)
        cls.watch = torch.randn(1, 1, 5000)
        cls.d6 = torch.randn(1, 6, 5000)

    def test_p0_shape_grid_and_no_anchor_copy(self) -> None:
        model = M1AxialLeadTimeModel(fusion_mode='none', task_id='task1').eval()
        with torch.no_grad():
            output, trace = model(self.anchor, return_trace=True)
        self.assertEqual(output.shape, (1, 12, 5000))
        self.assertEqual(trace.z_shape, (1, 12, 250, 128))
        self.assertEqual(trace.time_attention_calls, 4)
        self.assertEqual(trace.lead_attention_calls, 4)
        self.assertFalse(trace.time_attention_is_causal)
        self.assertFalse(torch.equal(output[:, :1], self.anchor))

    def test_all_fusions_backward(self) -> None:
        for mode in ('film', 'gated_residual', 'film_gated_residual'):
            model = M1AxialLeadTimeModel(fusion_mode=mode, task_id='task1')
            output = model(self.anchor, context=self.watch, context_source_type='watch_ecg')
            self.assertEqual(output.shape, (1, 12, 5000))
            output.square().mean().backward()

    def test_task2_requires_one_complete_d6_source(self) -> None:
        model = M1AxialLeadTimeModel(fusion_mode='film', task_id='task2')
        mask = torch.ones(1, 6, dtype=torch.bool)
        output = model(self.anchor, context=self.d6, context_source_type='ecg_machine_d6', context_lead_mask=mask)
        self.assertEqual(output.shape, (1, 12, 5000))
        with self.assertRaises(ContractError):
            model(self.anchor, context=self.d6, context_source_type='body_scale_d6+ecg_machine_d6', context_lead_mask=mask)

    def test_masks_and_none_path_are_strict(self) -> None:
        model = M1AxialLeadTimeModel(fusion_mode='none', task_id='task1')
        lead_mask = torch.tensor([[True] + [False] * 11])
        bad_missing = lead_mask.clone()
        with self.assertRaises(ContractError):
            model(self.anchor, lead_mask=lead_mask, context_lead_mask=bad_missing)
        with self.assertRaises(ContractError):
            model(self.anchor, context=self.watch, context_source_type='watch_ecg')


if __name__ == '__main__':
    unittest.main()
