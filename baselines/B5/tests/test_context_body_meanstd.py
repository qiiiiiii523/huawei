from __future__ import annotations

import unittest
from dataclasses import asdict
from unittest.mock import patch

import torch
from torch import nn

from baselines.B5.config import ModelConfig
from baselines.B5.context_body import B5BodyContext
from baselines.B5.context_body_meanstd import B5BodyContextMeanStd, BodyContextEncoderMeanStd
from baselines.B5.model import B5UNet
from scripts.evaluate_b5_body_diagnostics import load_model


class MeanStdEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_dispersion_changes_embedding_at_fixed_mean(self) -> None:
        encoder = BodyContextEncoderMeanStd(4)
        encoder.cnn = nn.Identity()
        encoder.project = nn.Linear(12, 4, bias=False)
        encoder.fuse = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            encoder.project.weight.zero_()
            encoder.project.weight[0, 0] = 1
            encoder.fuse.weight.zero_()
            encoder.fuse.weight[0, 4] = 1
        varied = torch.zeros(1, 2, 6, 16)
        varied[0, 0, 0] = 1
        varied[0, 1, 0] = -1
        constant = torch.zeros_like(varied)
        mask = torch.ones_like(varied, dtype=torch.bool)
        weights = torch.full((1, 2), 0.5)
        available = torch.tensor([True])
        varied_result = encoder(varied, mask, weights, available)
        constant_result = encoder(constant, mask, weights, available)
        self.assertGreater(float((varied_result[0, 0] - constant_result[0, 0]).detach()), 0.9)

    def test_mask_and_window_order(self) -> None:
        encoder = BodyContextEncoderMeanStd(8)
        context = torch.randn(2, 3, 6, 64)
        mask = torch.ones_like(context, dtype=torch.bool)
        mask[:, 1, :, 40:] = False
        counts = mask.sum(dim=(2, 3)).float()
        weights = counts / counts.sum(dim=1, keepdim=True)
        available = torch.tensor([True, True])
        original = encoder(context, mask, weights, available)
        changed_padding = context.masked_fill(~mask, 9999)
        torch.testing.assert_close(encoder(changed_padding, mask, weights, available), original, rtol=0, atol=0)
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(encoder(context[:, order], mask[:, order], weights[:, order], available),
                                   original, rtol=1e-5, atol=1e-6)
        original.square().sum().backward()
        self.assertGreater(float(encoder.fuse[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(encoder.cnn[0].weight.grad.abs().sum()), 0)

    def test_zero_film_preserves_original_b5_velocity(self) -> None:
        config = ModelConfig(channels=(8, 16, 32), time_dim=16, metadata_dim=16,
                             metadata_dropout=0, metadata_field_dropout=0)
        original = B5UNet(config).eval()
        modified = B5BodyContextMeanStd(config).eval()
        missing, unexpected = modified.load_state_dict(original.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing and all(name.startswith(("body_encoder.", "body_film.")) for name in missing))
        anchor = torch.randn(1, 1, 64)
        state = torch.randn(1, 11, 64)
        time = torch.tensor([0.5])
        numeric = torch.zeros(1, 3)
        sex = torch.zeros(1, dtype=torch.long)
        field_mask = torch.ones(1, 4, dtype=torch.bool)
        age_topcoded = torch.zeros(1, 1)
        context = torch.randn(1, 2, 6, 64)
        mask = torch.ones_like(context, dtype=torch.bool)
        weights = torch.full((1, 2), 0.5)
        available = torch.tensor([True])
        with torch.no_grad():
            base_velocity, base_anchor = original(state, time, anchor, numeric, sex,
                                                   field_mask, age_topcoded)
            velocity, predicted_anchor = modified.forward_body(
                state, time, anchor, numeric, sex, field_mask, age_topcoded,
                context, mask, weights, available)
        torch.testing.assert_close(velocity, base_velocity, rtol=0, atol=0)
        torch.testing.assert_close(predicted_anchor, base_anchor, rtol=0, atol=0)

    def test_shared_encoder_initialization_matches_previous_variant(self) -> None:
        config = ModelConfig(channels=(8, 16, 32), time_dim=16, metadata_dim=16)
        torch.manual_seed(42)
        previous = B5BodyContext(config)
        torch.manual_seed(42)
        improved = B5BodyContextMeanStd(config)
        for part in ("cnn", "project"):
            old_weights = getattr(previous.body_encoder, part).state_dict()
            new_weights = getattr(improved.body_encoder, part).state_dict()
            for name, tensor in old_weights.items():
                torch.testing.assert_close(tensor, new_weights[name], rtol=0, atol=0)

    def test_diagnostic_loader_selects_new_encoder(self) -> None:
        config = ModelConfig(channels=(8, 16, 32), time_dim=16, metadata_dim=16)
        base = B5UNet(config)
        modified = B5BodyContextMeanStd(config)
        modified.load_state_dict(base.state_dict(), strict=False)
        source = {"model_config": asdict(config), "ema": base.state_dict()}
        payload = {"format": "b5-body-frozen-adapter-v1", "epoch": 5,
                   "run_config": {"experiment": "A_frozen_meanstd_v1",
                                  "source_checkpoint_sha256": "fixed", "steps": 16,
                                  "samples": 16, "frozen_backbone": True,
                                  "encoder_variant": "mean_std"},
                   "ema": modified.state_dict()}
        with patch("scripts.evaluate_b5_body_diagnostics.torch.load", return_value=payload), \
             patch("scripts.evaluate_b5_body_diagnostics.file_sha256", return_value="fixed"):
            loaded = load_model("A_true", source, torch.device("cpu"), checkpoint_kind="frozen_meanstd")
        self.assertIsInstance(loaded, B5BodyContextMeanStd)


if __name__ == "__main__":
    unittest.main()
