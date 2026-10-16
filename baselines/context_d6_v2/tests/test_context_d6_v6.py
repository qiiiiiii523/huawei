from __future__ import annotations

import unittest
from pathlib import Path

import torch

from baselines.B5.checkpoint import load_checkpoint
from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet
from baselines.context_d6_v6 import (
    ContextD6V6UNet,
    effective_window_log_weights,
    load_b5_weights,
)
from baselines.context_d6_v5 import ContextD6V5UNet, load_b5_weights as load_b5_weights_v5


ROOT = Path(__file__).resolve().parents[3]


class ContextD6V6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = load_checkpoint(ROOT / "results/B5/context_start/best.pt")
        cls.model_config = ModelConfig.from_dict(cls.checkpoint["model_config"])

    def _inputs(self):
        generator = torch.Generator().manual_seed(52)
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
            context_window_weights=torch.tensor([[0.2, 0.3, 0.5]]),
        )
        return common, context_args, generator

    def test_zero_reliability_preserves_v5_window_log_weights(self):
        weights = torch.tensor([[0.2, 0.3, 0.5]])
        zero_logits = torch.zeros_like(weights)
        torch.testing.assert_close(
            effective_window_log_weights(weights, zero_logits),
            weights.clamp_min(1e-8).log(), rtol=0, atol=0)

        model = ContextD6V6UNet(self.model_config).eval()
        load_b5_weights(model, self.checkpoint["ema"])
        common, context_args, _ = self._inputs()
        condition = model.encode_conditions(**common, **context_args)
        self.assertTrue(torch.equal(condition.context_reliability_logits,
                                    torch.zeros_like(condition.context_reliability_logits)))
        expected = context_args["context_window_weights"].clamp_min(1e-8).log()
        torch.testing.assert_close(condition.context_effective_log_weights,
                                   expected, rtol=0, atol=0)

        v5 = ContextD6V5UNet(self.model_config).eval()
        load_b5_weights_v5(v5, self.checkpoint["ema"])
        v5_encoder_state = {
            key: value for key, value in model.context_encoder.state_dict().items()
            if not key.startswith("reliability_head.")
        }
        v5.context_encoder.load_state_dict(v5_encoder_state, strict=True)
        v5.query_projection.load_state_dict(model.query_projection.state_dict(), strict=True)
        v5_condition = v5.encode_conditions(**common, **context_args)
        torch.testing.assert_close(condition.context_attention, v5_condition.context_attention,
                                   rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(condition.context_embedding, v5_condition.context_embedding,
                                   rtol=1e-5, atol=5e-7)

    def test_missing_context_and_all_masked_context_are_finite(self):
        model = ContextD6V6UNet(self.model_config).eval()
        load_b5_weights(model, self.checkpoint["ema"])
        common, context_args, generator = self._inputs()
        for kwargs in ({}, {
                **context_args,
                "context_time_mask": torch.zeros_like(context_args["context_time_mask"]),
                "context_window_mask": torch.zeros_like(context_args["context_window_mask"]),
                "context_window_weights": torch.zeros_like(context_args["context_window_weights"]),
        }):
            condition = model.encode_conditions(**common, **kwargs)
            state = torch.randn(1, 11, 5000, generator=generator)
            output = model.velocity(state, torch.tensor([0.5]), condition)
            self.assertTrue(torch.isfinite(output).all())
            self.assertFalse(condition.context_available.any())
            self.assertFalse(condition.context_token_mask.any())

    def test_reliability_head_receives_gradient(self):
        model = ContextD6V6UNet(self.model_config)
        load_b5_weights(model, self.checkpoint["ema"])
        common, context_args, _ = self._inputs()
        condition = model.encode_conditions(**common, **context_args)
        objective = (condition.context_attention * torch.tensor([[0.0, 1.0, 2.0]])).sum()
        objective.backward()
        gradient = model.context_encoder.reliability_head[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_zero_cross_attention_gate_falls_back_exactly_to_b5(self):
        base = B5UNet(self.model_config).eval()
        model = ContextD6V6UNet(self.model_config).eval()
        base.load_state_dict(self.checkpoint["ema"], strict=True)
        load_b5_weights(model, self.checkpoint["ema"])
        common, context_args, generator = self._inputs()
        state = torch.randn(1, 11, 5000, generator=generator)
        time = torch.tensor([0.4])
        base_condition = base.encode_conditions(**common)
        for kwargs in ({}, {
                **context_args,
                "context_time_mask": torch.zeros_like(context_args["context_time_mask"]),
                "context_window_mask": torch.zeros_like(context_args["context_window_mask"]),
                "context_window_weights": torch.zeros_like(context_args["context_window_weights"]),
        }):
            condition = model.encode_conditions(**common, **kwargs)
            torch.testing.assert_close(
                model.velocity(state, time, condition),
                base.velocity(state, time, base_condition), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
