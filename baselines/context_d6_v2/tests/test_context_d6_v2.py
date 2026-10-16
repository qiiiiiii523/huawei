from __future__ import annotations

import unittest
from pathlib import Path

import torch

from baselines.B5.checkpoint import load_checkpoint
from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet
from baselines.context_d6_v2.model import AttentiveContextEncoder, ContextD6V2UNet, load_b5_weights

ROOT = Path(__file__).resolve().parents[3]


class ContextD6V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = load_checkpoint(ROOT / "results/B5/context_start/best.pt")
        cls.model_config = ModelConfig.from_dict(cls.checkpoint["model_config"])

    def test_all_masked_context_is_finite_zero(self):
        encoder = AttentiveContextEncoder(32).eval()
        context = torch.randn(2, 3, 6, 5000)
        embedding, available, attention = encoder(
            context, torch.zeros_like(context, dtype=torch.bool),
            torch.zeros(2, 3, dtype=torch.bool), torch.zeros(2, 3))
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertTrue(torch.equal(embedding, torch.zeros_like(embedding)))
        self.assertFalse(available.any())
        self.assertTrue(torch.equal(attention, torch.zeros_like(attention)))

    def test_masked_window_gets_zero_attention(self):
        encoder = AttentiveContextEncoder(32).eval()
        context = torch.randn(1, 3, 6, 5000)
        time_mask = torch.ones_like(context, dtype=torch.bool)
        window_mask = torch.tensor([[True, False, True]])
        weights = torch.tensor([[0.5, 0.0, 0.5]])
        _, available, attention = encoder(context, time_mask, window_mask, weights)
        self.assertTrue(bool(available[0]))
        self.assertEqual(float(attention[0, 1].detach()), 0.0)
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(1))

    def test_permutation_and_padding_do_not_change_embedding(self):
        encoder = AttentiveContextEncoder(32).eval()
        generator = torch.Generator().manual_seed(23)
        context = torch.randn(1, 3, 6, 5000, generator=generator)
        time_mask = torch.ones_like(context, dtype=torch.bool)
        window_mask = torch.ones(1, 3, dtype=torch.bool)
        weights = torch.tensor([[0.2, 0.3, 0.5]])
        original = encoder(context, time_mask, window_mask, weights)[0]
        order = torch.tensor([2, 0, 1])
        permuted = encoder(context[:, order], time_mask[:, order],
                           window_mask[:, order], weights[:, order])[0]
        torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)

        padded_context = torch.cat((context, torch.full((1, 1, 6, 5000), float("nan"))), dim=1)
        padded_time = torch.cat((time_mask, torch.zeros(1, 1, 6, 5000, dtype=torch.bool)), dim=1)
        padded_windows = torch.tensor([[True, True, True, False]])
        padded_weights = torch.tensor([[0.2, 0.3, 0.5, 0.0]])
        padded = encoder(padded_context, padded_time, padded_windows, padded_weights)[0]
        torch.testing.assert_close(original, padded, rtol=1e-5, atol=1e-6)

    def test_zero_initialized_v2_is_exact_b5(self):
        base = B5UNet(self.model_config).eval()
        model = ContextD6V2UNet(self.model_config).eval()
        base.load_state_dict(self.checkpoint["ema"], strict=True)
        load_b5_weights(model, self.checkpoint["ema"])
        generator = torch.Generator().manual_seed(17)
        anchor = torch.randn(1, 1, 5000, generator=generator)
        numeric = torch.zeros(1, 3)
        sex = torch.full((1,), 2, dtype=torch.long)
        fields = torch.zeros(1, 4, dtype=torch.bool)
        topcoded = torch.zeros(1, 1)
        context = torch.randn(1, 2, 6, 5000, generator=generator)
        condition = model.encode_conditions(
            anchor, numeric, sex, fields, topcoded, context=context,
            context_time_mask=torch.ones_like(context, dtype=torch.bool),
            context_window_mask=torch.ones(1, 2, dtype=torch.bool),
            context_window_weights=torch.full((1, 2), 0.5))
        base_condition = base.encode_conditions(anchor, numeric, sex, fields, topcoded)
        state = torch.randn(1, 11, 5000, generator=generator)
        time = torch.tensor([0.4])
        torch.testing.assert_close(model.velocity(state, time, condition),
                                   base.velocity(state, time, base_condition), rtol=0, atol=0)

    def test_projection_gradient_is_finite(self):
        model = ContextD6V2UNet(self.model_config)
        load_b5_weights(model, self.checkpoint["ema"])
        context = torch.randn(1, 2, 6, 5000)
        condition = model.encode_conditions(
            torch.randn(1, 1, 5000), torch.zeros(1, 3),
            torch.full((1,), 2, dtype=torch.long), torch.zeros(1, 4, dtype=torch.bool),
            torch.zeros(1, 1), context=context,
            context_time_mask=torch.ones_like(context, dtype=torch.bool),
            context_window_mask=torch.ones(1, 2, dtype=torch.bool),
            context_window_weights=torch.full((1, 2), 0.5))
        model.velocity(torch.randn(1, 11, 5000), torch.tensor([0.5]), condition).square().mean().backward()
        gradients = [layer.weight.grad for layer in model.context_projection]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)


if __name__ == "__main__":
    unittest.main()

