import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5_best.check_beta_seed import ComponentPredictor, mix
from baselines.B5_best.combine import combine


class BetaSeedTests(unittest.TestCase):
    def test_beta_one_reproduces_original_combination(self):
        rng = np.random.default_rng(4)
        fm = rng.normal(size=(12, 60000)).astype(np.float32)
        slow = rng.normal(size=(11, 60000))
        np.testing.assert_array_equal(mix(fm, slow, 1), combine(fm, slow))

    def test_beta_only_changes_missing_slow_contribution(self):
        fm = np.ones((12, 60000), np.float32)
        slow = np.full((11, 60000), 8.0)
        a, b = mix(fm, slow, 1), mix(fm, slow, .75)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1:] - b[1:], 2)

    def test_candidate_seed_reaches_sampler_without_changing_stable_profile(self):
        engine = ComponentPredictor.__new__(ComponentPredictor)
        engine.device = torch.device('cpu')
        engine.scale = np.ones(12, np.float32)
        engine.fm, engine.slow = object(), object()
        engine.bundle = {'sampling': {'seed': 42, 'steps': 16, 'samples': 16, 'solver': 'heun'}}
        meta = {'numeric': np.zeros(4, np.float32), 'sex': np.array(0, np.int64),
                'field_mask': np.zeros(4, bool), 'age_topcoded': np.zeros(1, np.float32)}
        def generated(model, inputs, keys, **settings):
            self.assertEqual(settings['seed'], 43)
            self.assertEqual(settings['samples'], 16)
            self.assertTrue(all(k.startswith('record:') for k in keys))
            return torch.ones(len(keys), 12, 5000)
        with patch('baselines.B5_best.check_beta_seed.predict_native',
                   return_value=torch.zeros(1, 11, 6000)), patch(
                   'baselines.B5_best.check_beta_seed.sample', side_effect=generated) as sampler:
            fm, slow = engine.components(np.zeros((1, 60000), np.float32), meta, 'record', 43, 4)
        self.assertEqual(sampler.call_count, 3)
        self.assertEqual(engine.bundle['sampling']['seed'], 42)
        self.assertEqual(fm.shape, (12, 60000))
        self.assertEqual(slow.shape, (11, 60000))


if __name__ == '__main__':
    unittest.main()
