"""Non-training B2-v1 contract and smoke checks."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b2_data import _weak_samples, build_strict_dataset, fit_b2_preprocessor
from ecg12gen.b2_model import B2MaskedPatchTransformer
from ecg12gen.b2_train import strict_missing_loss, weak_route_loss
from ecg12gen.dataset import ECGDataConfig
from ecg12gen.d12_pretrain import StrictD12PretrainDataset


def main() -> None:
    torch.manual_seed(42)
    model = B2MaskedPatchTransformer()
    assert 250_000 <= model.parameter_count <= 400_000
    task1_input = torch.randn(2, 1, 5000)
    task1_mask = torch.tensor([[True] + [False] * 11] * 2)
    task1_missing = ~task1_mask
    task1_output = model(task1_input, task1_mask, task1_missing)
    assert task1_output.shape == (2, 12, 5000)
    task2_input = torch.randn(2, 6, 5000)
    task2_mask = torch.tensor([[True] * 6 + [False] * 6] * 2)
    task2_output = model(task2_input, task2_mask, ~task2_mask)
    assert task2_output.shape == (2, 12, 5000)
    assert torch.equal(task1_missing, ~task1_mask) and torch.equal(~task2_mask, ~task2_mask)
    assert not any(parameter.requires_grad for name, parameter in model.named_buffers() if name == "positional_encoding")

    prediction = torch.randn(2, 12, 5000)
    target = torch.randn(2, 12, 5000)
    missing = torch.tensor([[False] + [True] * 11, [False] + [True] * 11])
    baseline_loss = strict_missing_loss(prediction, target, missing)
    observed_changed = prediction.clone(); observed_changed[:, 0] += 100.0
    target_observed_changed = target.clone(); target_observed_changed[:, 0] -= 100.0
    assert torch.allclose(strict_missing_loss(observed_changed, target_observed_changed, missing), baseline_loss)

    weak_source = inspect.getsource(weak_route_loss)
    assert "masked_huber_loss" not in weak_source and "masked_pcc_loss" not in weak_source
    assert "observed_consistency_loss" in weak_source and "spectral_stat_loss" in weak_source
    assert "pair_invariant_stat_loss" in weak_source
    predict_source = (ROOT / "scripts" / "predict_b2.py").read_text(encoding="utf-8")
    assert "baseline_uV" not in predict_source and "target_baseline" not in predict_source

    config = ECGDataConfig.from_yaml(ROOT / "configs" / "common.yaml")
    strict_rows = StrictD12PretrainDataset(config, "d12_i_pretrain")
    assert len(strict_rows) > 0 and all(sample.split == "train" for sample in strict_rows)
    preprocessor = fit_b2_preprocessor(ROOT / "configs" / "common.yaml", "task2", "B_detrend_0p2Hz_then_window")
    strict = build_strict_dataset(ROOT / "configs" / "common.yaml", "task2", preprocessor)
    assert all(sample.split == "train" for sample in strict.samples)
    task2_b_samples = _weak_samples(config, "task2", "train", "B_detrend_0p2Hz_then_window")
    assert task2_b_samples and all(sample.meta.get("input_processing_variant", "") != "A_raw_window" for sample in task2_b_samples if "input_processing_variant" in sample.meta)
    assert all(sample.meta.get("device_type") != "body_scale_d6" or sample.meta.get("input_processing_variant") == "B_detrend_0p2Hz_then_window" for sample in task2_b_samples)
    print(f"PASS: B2 forward/masks/loss guards; parameters={model.parameter_count}; strict_train={len(strict_rows)}; task2-B={len(task2_b_samples)}")


if __name__ == "__main__":
    main()
