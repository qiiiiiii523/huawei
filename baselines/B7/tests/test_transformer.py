"""Transformer-specific checks in addition to the unchanged B7 behavior contract."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from baselines.B7 import ARCHITECTURE_ID
from baselines.B7.checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint
from baselines.B7.config import ModelConfig
from baselines.B7.flow import integrate
from baselines.B7.model import B7Transformer, PatchEmbedding, SelfAttention, TemporalBlock, temporal_positions
from baselines.B7.tests.test_b7 import payload, small_config


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_patchification_keeps_all_raw_samples_in_order(self):
        embedding = PatchEmbedding(10, 80)
        with torch.no_grad():
            embedding.projection.weight.copy_(torch.eye(80))
            embedding.projection.bias.zero_()
        raw = torch.arange(8 * 20).reshape(1, 8, 20).float()
        actual = embedding(raw).reshape(1, 20, 8).transpose(1, 2)
        self.assertTrue(torch.equal(actual, raw))

    def test_architecture_has_no_convolution_and_has_global_bottleneck(self):
        model = B7Transformer()
        self.assertFalse(any(isinstance(m, nn.Conv1d) for m in model.modules()))
        self.assertEqual(model.patch_embed.projection.in_features, 80)
        self.assertEqual(len(model.bottleneck), 4)
        self.assertTrue(all(b.window_size is None for b in model.bottleneck))
        self.assertEqual([b.shift for b in model.encode0], [0, 32])
        self.assertEqual([b.shift for b in model.encode1], [0, 32])

    def test_40s_attention_lengths_are_2000_1000_500(self):
        model = B7Transformer().eval()
        seen = []
        hooks = [module.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[1]))
                 for module in (model.encode0[0], model.encode1[0], model.bottleneck[0])]
        with torch.no_grad():
            output = model(torch.zeros(1, 7, 20000), torch.tensor([.5]), torch.zeros(1, 1, 20000))
        for h in hooks:
            h.remove()
        self.assertEqual(seen, [2000, 1000, 500])
        self.assertEqual(output.shape, (1, 7, 20000))
        self.assertTrue(torch.isfinite(output).all())

    def test_shift_crosses_previous_boundary_without_wrapping_end_to_start(self):
        torch.manual_seed(11)
        regular = TemporalBlock(8, 2, 8, 2, window_size=4, shift=0).eval()
        shifted = TemporalBlock(8, 2, 8, 2, window_size=4, shift=2).eval()
        shifted.load_state_dict(regular.state_dict())
        tokens = torch.randn(1, 8, 8)
        valid = torch.ones(1, 8, dtype=torch.bool)
        changed = tokens.clone()
        changed[:, 4] += 3
        # Positions3/4 share the shifted window, but not the regular window.
        torch.testing.assert_close(regular.attend(tokens, valid)[:, 3], regular.attend(changed, valid)[:, 3], rtol=0, atol=0)
        self.assertGreater(float((shifted.attend(tokens, valid)[:, 3] - shifted.attend(changed, valid)[:, 3]).detach().abs().max()), 1e-5)
        changed = tokens.clone()
        changed[:, -1] += 100
        torch.testing.assert_close(shifted.attend(tokens, valid)[:, 0], shifted.attend(changed, valid)[:, 0], rtol=0, atol=0)

    def test_padded_keys_are_excluded_and_fully_masked_windows_stay_finite(self):
        torch.manual_seed(12)
        attention = SelfAttention(8, 2, window_size=4).eval()
        tokens = torch.randn(2, 4, 8)
        valid = torch.tensor([[True, True, False, False], [False, False, False, False]])
        changed = tokens.clone()
        changed[~valid] = 1e5
        a, b = attention(tokens, valid), attention(changed, valid)
        torch.testing.assert_close(a[valid], b[valid], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(b).all())
        self.assertEqual(float(b[~valid].detach().abs().sum()), 0.)

    def test_short_or_odd_patch_and_merge_lengths_restore_exact_sample_count(self):
        model = B7Transformer(small_config()).eval()
        with torch.no_grad():
            for length in (8, 11, 39, 41, 5001):
                output = model(torch.randn(2, 7, length), torch.tensor([0., 1.]), torch.randn(2, 1, length))
                self.assertEqual(output.shape, (2, 7, length))
                self.assertTrue(torch.isfinite(output).all())

    def test_output_head_unpatchifies_samples_and_generated_leads_in_order(self):
        model = B7Transformer(small_config()).eval()
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.copy_(torch.arange(70))
            output = model(torch.zeros(1, 7, 41), torch.tensor([.5]), torch.zeros(1, 1, 41))
        expected = torch.arange(70).reshape(10, 7).T.repeat(1, 5)[:, :41]
        self.assertTrue(torch.equal(output[0], expected))

    def test_far_context_reaches_central_waveform_through_global_attention(self):
        torch.manual_seed(13)
        model = B7Transformer(small_config())
        anchor = torch.randn(1, 1, 20000, requires_grad=True)
        output = model(torch.randn(1, 7, 20000), torch.tensor([.5]), anchor)
        gradient = torch.autograd.grad(output[0, 0, 10000], anchor)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[:, :, :100].abs().sum()), 1e-8)

    def test_flow_time_and_demographics_condition_the_velocity(self):
        torch.manual_seed(14)
        model = B7Transformer(small_config()).eval()
        anchor, state = torch.randn(1, 1, 5000), torch.randn(1, 7, 5000)
        metadata = {'numeric': torch.tensor([[.5, .8, .4]]), 'sex': torch.tensor([0]),
                    'field_mask': torch.ones(1, 4, dtype=torch.bool), 'age_topcoded': torch.zeros(1, 1)}
        with torch.no_grad():
            a = model(state, torch.tensor([.2]), anchor)
            b = model(state, torch.tensor([.8]), anchor)
            c = model(state, torch.tensor([.2]), anchor, metadata)
        self.assertGreater(float((a - b).abs().max()), 1e-5)
        self.assertGreater(float((a - c).abs().max()), 1e-5)

    def test_position_prefix_is_identical_between_short_and_long_windows(self):
        ref = torch.zeros(1)
        self.assertTrue(torch.equal(temporal_positions(500, 128, ref), temporal_positions(2000, 128, ref)[:, :500]))

    def test_U7_checkpoints_rejected_at_load_and_model_construction(self):
        self.assertEqual(ARCHITECTURE_ID, 'B7-T-temporal-transformer-v1')
        model = B7Transformer(small_config())
        old = payload(model)
        old['architecture_id'] = 'B7-U7-raw-I-v1'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'u7.pt'
            save_checkpoint(p, old)
            with self.assertRaisesRegex(ValueError, 'B7-U7'):
                load_checkpoint(p)
            with self.assertRaisesRegex(ValueError, 'B7-T'):
                model_from_checkpoint(old)

    def test_model_config_roundtrip_and_invalid_dimensions(self):
        self.assertEqual(ModelConfig.from_dict(asdict(ModelConfig())), ModelConfig())
        for values in ({'heads': (3, 8, 12)}, {'depths': (1, 2, 4)}, {'patch_size': 0}, {'window_size': 1}):
            with self.assertRaises(ValueError):
                ModelConfig(**values)

    def test_actual_transformer_Heun_keeps_original_I_at_every_step(self):
        torch.manual_seed(15)
        model = B7Transformer(small_config()).eval()
        anchor, state = torch.randn(1, 1, 5000), torch.randn(1, 7, 5000)
        observed = []
        h = model.patch_embed.register_forward_pre_hook(lambda m, args: observed.append(args[0][:, :1].clone()))
        output = integrate(model, anchor, state, steps=2)
        h.remove()
        self.assertEqual(len(observed), 4)
        self.assertTrue(all(torch.equal(a, anchor) for a in observed))
        self.assertEqual(output.shape, state.shape)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
