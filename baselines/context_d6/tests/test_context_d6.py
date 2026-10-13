from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from baselines.B5.checkpoint import checkpoint_preprocessor, load_checkpoint
from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet
from baselines.context_d6.config import load_config
from baselines.context_d6.data import MachineD6Collator, MachineD6Dataset
from baselines.context_d6.flow import flow_loss
from baselines.context_d6.model import ContextD6UNet, load_b5_weights
from scripts.prepare_task2_context_scales import with_context_scales

ROOT = Path(__file__).resolve().parents[3]


class ContextD6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs/experiments/context_d6_context.yaml")
        cls.checkpoint = load_checkpoint(ROOT / "results/B5/context_start/best.pt")
        from ecg12gen.preprocessing import PreprocessingConfig
        pc = PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml")
        base = checkpoint_preprocessor(cls.checkpoint, pc)
        cls.preprocessor = with_context_scales(
            base, ROOT / "results/B5/context_params/machine_d6_scales.npz",
            "ecg_machine_d6", "A_raw_window")

    def test_machine_only_dataset_and_masks(self):
        train = MachineD6Dataset(self.config, "train", self.preprocessor)
        validation = MachineD6Dataset(self.config, "validation", self.preprocessor)
        self.assertTrue(train and validation)
        self.assertTrue(all(train[i].context_source_type == "ecg_machine_d6"
                            for i in range(min(16, len(train)))))
        batch = MachineD6Collator(self.preprocessor, train.demographics, context_enabled=True)([train[0], train[1]])
        self.assertEqual(batch["context"].shape[0], 2)
        self.assertEqual(batch["context"].shape[2:], (6, 5000))
        self.assertEqual(batch["anchor"].shape, (2, 1, 5000))
        self.assertEqual(batch["target"].shape, (2, 12, 5000))
        self.assertFalse(np.any(batch["context_time_mask"] & ~np.isfinite(batch["context"])))

    def test_control_collator_removes_context_only(self):
        train = MachineD6Dataset(self.config, "train", self.preprocessor)
        sample = train[0]
        batch = MachineD6Collator(self.preprocessor, train.demographics, context_enabled=False)([sample])
        self.assertFalse(batch["context_available"].any())
        self.assertFalse(batch["context_time_mask"].any())
        self.assertFalse(batch["context_window_mask"].any())
        self.assertTrue(np.array_equal(batch["anchor"][0],
                                       self.preprocessor.transform_window(sample.anchor_i_ecg, "ecg_machine_i").model_signal))

    def test_zero_initialized_context_is_exact_b5(self):
        cfg = ModelConfig.from_dict(self.checkpoint["model_config"])
        base, context = B5UNet(cfg).eval(), ContextD6UNet(cfg).eval()
        base.load_state_dict(self.checkpoint["ema"], strict=True)
        load_b5_weights(context, self.checkpoint["ema"])
        generator = torch.Generator().manual_seed(7)
        anchor = torch.randn(1, 1, 5000, generator=generator)
        numeric = torch.zeros(1, 3)
        sex = torch.full((1,), 2, dtype=torch.long)
        fields = torch.zeros(1, 4, dtype=torch.bool)
        topcoded = torch.zeros(1, 1)
        state = torch.randn(1, 11, 5000, generator=generator)
        time = torch.tensor([0.4])
        base_condition = base.encode_conditions(anchor, numeric, sex, fields, topcoded)
        context_condition = context.encode_conditions(
            anchor, numeric, sex, fields, topcoded,
            context=torch.randn(1, 2, 6, 5000, generator=generator),
            context_time_mask=torch.ones(1, 2, 6, 5000, dtype=torch.bool),
            context_window_mask=torch.ones(1, 2, dtype=torch.bool),
            context_window_weights=torch.full((1, 2), 0.5))
        torch.testing.assert_close(context.velocity(state, time, context_condition),
                                   base.velocity(state, time, base_condition), rtol=0, atol=0)
        torch.testing.assert_close(context_condition.anchor_prediction,
                                   base_condition.anchor_prediction, rtol=0, atol=0)

    def test_real_batch_backward_reaches_context_injection(self):
        train = MachineD6Dataset(self.config, "train", self.preprocessor)
        batch = MachineD6Collator(
            self.preprocessor, train.demographics, context_enabled=True)([train[0]])
        tensors = {key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
                   for key, value in batch.items()}
        cfg = ModelConfig.from_dict(self.checkpoint["model_config"])
        model = ContextD6UNet(cfg)
        load_b5_weights(model, self.checkpoint["ema"])
        scale = torch.as_tensor(self.preprocessor.scale_uV_by_source["d12"])
        loss, _ = flow_loss(model, tensors, self.config["loss"], scale,
                            noise=torch.zeros(1, 11, 5000), time=torch.tensor([0.5]))
        loss.backward()
        gradients = [layer.weight.grad for layer in model.context_projection]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                            for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)


if __name__ == "__main__":
    unittest.main()

