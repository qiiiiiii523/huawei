import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from FM_Unet.config import ModelConfig
from FM_Unet.fm_experiment import build, predict_slow
from FM_Unet.inference import SlowPredictor
from FM_Unet.model import SlowUNet7
from FM_Unet.native50 import pack_i


class DepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_seven_levels_forward_backward(self):
        model = build()
        sizes = []
        hooks = [getattr(model.core, f'down{i}').register_forward_hook(
            lambda module, args, output: sizes.append(output.shape[-1])) for i in range(1, 7)]
        state = torch.randn(1, 7, 600)
        anchor = torch.randn(1, 1, 60000)
        output = model(state, torch.tensor([0.5]), anchor, None)
        for hook in hooks:
            hook.remove()
        self.assertEqual(sizes, [300, 150, 75, 37, 18, 9])
        self.assertEqual(output.shape, state.shape)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_odd_lengths_and_serialization(self):
        config = ModelConfig.from_dict(ModelConfig().to_dict())
        model = SlowUNet7(config)
        output = model(torch.randn(1, 7, 601), torch.tensor([0.5]), torch.randn(1, 1, 601))
        self.assertEqual(output.shape, (1, 7, 601))
        clone = SlowUNet7(config)
        clone.load_state_dict(model.state_dict(), strict=True)

    def test_supported_rates_and_short_input_guard(self):
        with torch.no_grad():
            for rate in (5, 10, 50):
                model = build(rate=rate)
                self.assertEqual(model(torch.zeros(1, 7, 120*rate), torch.tensor([0.0]),
                                       torch.zeros(1, 1, 60000), None).shape, (1, 7, 120*rate))
            model = build(rate=50, input_seconds=10, output_seconds=10)
            self.assertEqual(model(torch.zeros(1, 7, 500), torch.tensor([0.0]),
                                   torch.zeros(1, 1, 5000), None).shape, (1, 7, 500))
        with self.assertRaisesRegex(ValueError, 'too short'):
            build(rate=5, input_seconds=10, output_seconds=10)

    def test_legacy_checkpoint_matches_original_forward(self):
        checkpoint = torch.load(Path(__file__).parents[1] / 'best.pt', weights_only=True)
        model = build(ModelConfig.from_dict(checkpoint['model_config']), checkpoint['native_hz'],
                      checkpoint['input_seconds'], checkpoint['output_seconds']).eval()
        model.load_state_dict(checkpoint['ema'], strict=True)
        self.assertEqual(len(model.config.channels), 3)
        state, anchor, time = torch.randn(1, 7, 600), torch.randn(1, 1, 60000), torch.tensor([0.5])
        from FM_Unet.fm_experiment import linear_expand, restrict50
        with torch.no_grad():
            c = model.core
            te, me = c.time_embedding(time), c.metadata_encoder(anchor)
            h = restrict50(c.stem(torch.cat((pack_i(anchor), linear_expand(state, 10)), 1)), 5)
            s0 = c._blocks(c.encode0, h, te, me)
            s1 = c._blocks(c.encode1, c.down1(s0), te, me)
            h = c._blocks(c.bottleneck, c.down2(s1), te, me)
            h = c.up1(F.interpolate(h, size=s1.shape[-1], mode='nearest'))
            h = c._blocks(c.decode1, c.merge1(torch.cat((h, s1), 1)), te, me)
            h = c.up0(F.interpolate(h, size=s0.shape[-1], mode='nearest'))
            expected = c.output(c._blocks(c.decode0, c.merge0(torch.cat((h, s0), 1)), te, me))
            torch.testing.assert_close(model(state, time, anchor, None), expected, rtol=0, atol=0)
        self.assertEqual(len(SlowPredictor(device='cpu').model.config.channels), 3)

    def test_seven_level_sampling_returns_slow11(self):
        import numpy as np
        model = build()
        meta = {'numeric': np.zeros(3), 'sex': np.array(2),
                'field_mask': np.zeros(4, dtype=bool), 'age_topcoded': np.zeros(1)}
        result = predict_slow(model, np.zeros((1, 60000)), meta, np.ones(12), np.ones(7),
                              'FM', 'depth-test', steps=1, samples=2, trajectory_batch=2)
        self.assertEqual(result.shape, (11, 60000))
        self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
