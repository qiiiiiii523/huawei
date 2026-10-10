from __future__ import annotations

import copy
from pathlib import Path
import unittest

from baselines.B5.config import load_config
from baselines.B5.objective import validate_mainline_loss, validate_resume_loss


ROOT = Path(__file__).resolve().parents[3]


class CleanMainlineTests(unittest.TestCase):
    def test_legacy_checkpoint_zero_slow_loss_can_resume(self):
        loss = load_config(ROOT/'configs/experiments/b5_finetune_meta.yaml')['loss']
        historical = {**loss, 'slow_trend': 0., 'slow_trend_seconds': 1., 'slow_trend_delta': 1.}
        validate_resume_loss({'loss_config': historical}, loss)
        validate_resume_loss({'config': {'loss': copy.deepcopy(loss)}}, loss)
        with self.assertRaisesRegex(ValueError, 'changed loss'):
            validate_resume_loss({'loss_config': historical}, {**loss, 'pcc': .2})

    def test_experimental_loss_is_rejected_instead_of_silently_ignored(self):
        validate_mainline_loss({'slow_trend': 0.})
        for value in (.1, -1., float('nan')):
            with self.assertRaises(ValueError):
                validate_mainline_loss({'slow_trend': value})

    def test_training_cannot_start_from_scratch(self):
        from baselines.B5.runtime import run_training
        with self.assertRaisesRegex(ValueError, 'requires'):
            run_training({'stage': 'finetune'}, 'cpu')
        with self.assertRaisesRegex(ValueError, 'only runs'):
            run_training({'stage': 'local'}, 'cpu')

    def test_only_mainline_experiment_configs_remain(self):
        self.assertEqual({p.name for p in (ROOT/'configs/experiments').glob('b5_*.yaml')},
                         {'b5_base.yaml', 'b5_finetune_meta.yaml'})


if __name__ == '__main__':
    unittest.main()
