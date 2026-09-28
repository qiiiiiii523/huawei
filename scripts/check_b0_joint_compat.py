"""Synthetic checks for B0 quality masks and missing-11 checkpoint semantics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ecg12gen.evaluate import evaluate_joint_anchor_predictions
from ecg12gen.losses import strict_anchor_pretrain_loss
from ecg12gen.models.b0_joint_anchor import B0JointAnchor
from scripts.train_b0_joint_p0 import ridge_weights


class _SyntheticRidgeDataset:
    def __init__(self) -> None:
        anchor = torch.ones((1, 2), dtype=torch.float32)
        first = torch.ones((12, 2), dtype=torch.float32)
        second = torch.full((12, 2), 3.0, dtype=torch.float32)
        first_mask = torch.ones(12, dtype=torch.bool)
        second_mask = torch.ones(12, dtype=torch.bool)
        second_mask[0] = False
        self.rows = ((anchor, first, first_mask), (anchor, second, second_mask))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def main() -> None:
    weights = ridge_weights(_SyntheticRidgeDataset(), alpha=0.0)
    assert np.isclose(weights[0], 1.0)
    assert np.allclose(weights[1:], 2.0)

    generator = torch.Generator().manual_seed(42)
    target = torch.randn((2, 12, 5000), generator=generator)
    anchor = target[:, :1].clone()
    prediction = target.clone()
    target_mask = torch.ones((2, 12), dtype=torch.bool)
    target_mask[:, 5] = False
    baseline_loss = strict_anchor_pretrain_loss(
        prediction, target, anchor, target_lead_mask=target_mask,
        physiology_weight=0.0, observed_weight=0.0,
    )
    prediction[:, 5] += 1000.0
    masked_loss = strict_anchor_pretrain_loss(
        prediction, target, anchor, target_lead_mask=target_mask,
        physiology_weight=0.0, observed_weight=0.0,
    )
    assert torch.allclose(baseline_loss, masked_loss)

    raw_target = target.numpy()
    raw_prediction = raw_target.copy()
    raw_prediction[:, 0] *= -1
    summary, _, _, _ = evaluate_joint_anchor_predictions(
        raw_prediction, raw_target, raw_target[:, :1], "task1", target_mask.numpy()
    )
    assert np.isclose(summary["r_missing11"], 1.0)
    assert summary["checkpoint_selection_metric"] == "r_missing11"

    model = B0JointAnchor(context_channels=6)
    model.set_stage("P1-C3")
    context = torch.randn((2, 6, 64), generator=generator)
    context_mask = torch.ones((2, 6), dtype=torch.bool)
    context_mask[:, 3] = False
    anchor_mask = torch.zeros((2, 12), dtype=torch.bool)
    anchor_mask[:, 0] = True
    altered = context.clone()
    altered[:, 3] += 1_000_000.0
    first = model(context, anchor, context_mask, anchor_mask)
    second = model(altered, anchor, context_mask, anchor_mask)
    assert torch.allclose(first, second)

    print("PASS: B0 quality masks, masked Ridge fit, and r_missing11 checkpoint contract")


if __name__ == "__main__":
    main()
