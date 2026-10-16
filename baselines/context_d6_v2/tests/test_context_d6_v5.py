from __future__ import annotations

import unittest
from pathlib import Path

import torch

from baselines.B5.checkpoint import load_checkpoint
from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet
from baselines.context_d6_v5 import ContextD6V5UNet, load_b5_weights


ROOT = Path(__file__).resolve().parents[3]


class ContextD6V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = load_checkpoint(ROOT / "results/B5/context_start/best.pt")
        cls.model_config = ModelConfig.from_dict(cls.checkpoint["model_config"])

    def _inputs(self):
        generator = torch.Generator().manual_seed(31)
        anchor = torch.randn(1, 1, 5000, generator=generator)
        common = dict(
            anchor=anchor,
            numeric=torch.zeros(1, 3),
            sex=torch.full((1,), 2, dtype=torch.long),
            field_mask=torch.zeros(1, 4, dtype=torch.bool),
            age_topcoded=torch.zeros(1, 1),
        )
        context = torch.randn(1, 3, 6, 5000, generator=generator)
        context_args = dict(
            context=context,
            context_time_mask=torch.ones_like(context, dtype=torch.bool),
            context_window_mask=torch.ones(1, 3, dtype=torch.bool),
            context_window_weights=torch.full((1, 3), 1 / 3),
        )
        return common, context_args, generator

    def test_zero_initialized_v5_is_exact_b5(self):
        base = B5UNet(self.model_config).eval()
        model = ContextD6V5UNet(self.model_config).eval()
        base.load_state_dict(self.checkpoint["ema"], strict=True)
        load_b5_weights(model, self.checkpoint["ema"])
        common, context_args, generator = self._inputs()
        base_condition = base.encode_conditions(**common)
        condition = model.encode_conditions(**common, **context_args)
        state = torch.randn(1, 11, 5000, generator=generator)
        time = torch.tensor([0.4])
        torch.testing.assert_close(
            model.velocity(state, time, condition),
            base.velocity(state, time, base_condition), rtol=0, atol=0)

    def test_missing_context_is_finite_and_exact_fallback(self):
        model = ContextD6V5UNet(self.model_config).eval()
        load_b5_weights(model, self.checkpoint["ema"])
        common, _, generator = self._inputs()
        condition = model.encode_conditions(**common)
        state = torch.randn(1, 11, 5000, generator=generator)
        output = model.velocity(state, torch.tensor([0.5]), condition)
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(condition.context_available.any())
        self.assertFalse(condition.context_token_mask.any())

    def test_cross_attention_gate_receives_gradient(self):
        model = ContextD6V5UNet(self.model_config)
        load_b5_weights(model, self.checkpoint["ema"])
        common, context_args, generator = self._inputs()
        condition = model.encode_conditions(**common, **context_args)
        state = torch.randn(1, 11, 5000, generator=generator)
        model.velocity(state, torch.tensor([0.5]), condition).square().mean().backward()
        gradient = model.context_cross_gate.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient))
        self.assertGreater(float(gradient.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
