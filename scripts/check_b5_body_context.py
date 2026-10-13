"""CPU-only pre-best checks for the Task-2 body-scale context adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from baselines.B5.config import ModelConfig
from baselines.B5.context_body import B5BodyContext, sample_body
from baselines.B5.model import B5UNet
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from ecg12gen.record_context import collate_record_context
from ecg12gen.record_dataset import JointAnchorDataset
from scripts.prepare_task2_context_scales import with_context_scales


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "A": ("A_raw_window", "body_A_scales.npz"),
    "B": ("B_detrend_0p2Hz_then_window", "body_B_scales.npz"),
}


def check_data() -> None:
    config = PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml")
    base = ECGPreprocessor.load(config, ROOT / "results/B5/context_start/preprocessing_scales.npz")
    for label, (variant, scale_file) in VARIANTS.items():
        preprocessor = with_context_scales(base, ROOT / "results/B5/context_params" / scale_file,
                                           "body_scale_d6", variant)
        for split in ("train", "validation"):
            dataset = JointAnchorDataset(ROOT / "configs/common.yaml", "task2", split,
                                         body_scale_variant=variant)
            try:
                body_indices = [i for i, row_index in enumerate(dataset._indices)
                                if dataset._rows[row_index]["input_type"] == "body_scale_d6"]
                assert body_indices, (label, split)
                batch = collate_record_context([dataset[body_indices[0]]], preprocessor)
                assert batch["context"].shape[2:] == (6, 5000)
                assert batch["anchor_i"].shape == (1, 1, 5000)
                assert batch["target"].shape == (1, 12, 5000)
                mask, context = batch["context_time_mask"], batch["context"]
                assert np.all(context[~mask] == 0)
                counts = mask.sum(axis=(2, 3))
                expected = counts / max(int(counts.sum()), 1)
                np.testing.assert_allclose(batch["context_window_weights"], expected)
                assert bool(batch["context_available"][0]) == bool(mask.any())
                assert batch["context_source_type"] == ["body_scale_d6"]
                dropped = collate_record_context([dataset[body_indices[0]]], preprocessor,
                                                  context_dropout=1.0, rng=np.random.default_rng(42)) if split == "train" else None
                if dropped is not None:
                    assert not dropped["context_available"].any()
                    assert not dropped["context_time_mask"].any()
                    assert not dropped["context_window_weights"].any()
                print(f"{label} {split}: {len(body_indices)} body target windows, "
                      f"{context.shape[1]} context windows, {int(mask.sum())} valid points")
            finally:
                dataset.close()


def check_model() -> None:
    torch.manual_seed(42)
    config = ModelConfig(channels=(8, 16, 32), time_dim=16, metadata_dim=16, dilations=(1, 2))
    base = B5UNet(config).eval()
    model = B5BodyContext(config).eval()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(key.startswith(("body_encoder.", "body_film.")) for key in missing)
    assert not unexpected
    batch, windows, length = 2, 3, 64
    inputs = {
        "anchor": torch.randn(batch, 1, length),
        "numeric": torch.zeros(batch, 3), "sex": torch.full((batch,), 2, dtype=torch.long),
        "field_mask": torch.zeros(batch, 4, dtype=torch.bool), "age_topcoded": torch.zeros(batch, 1),
        "context": torch.randn(batch, windows, 6, length),
        "context_time_mask": torch.ones(batch, windows, 6, length, dtype=torch.bool),
        "context_window_weights": torch.full((batch, windows), 1 / windows),
        "context_available": torch.ones(batch, dtype=torch.bool),
    }
    state, time = torch.randn(batch, 11, length), torch.full((batch,), 0.4)
    with torch.no_grad():
        original, original_anchor = base(state, time, *[inputs[key] for key in
            ("anchor", "numeric", "sex", "field_mask", "age_topcoded")])
        adapted, adapted_anchor = model.forward_body(state, time, *[inputs[key] for key in
            ("anchor", "numeric", "sex", "field_mask", "age_topcoded", "context",
             "context_time_mask", "context_window_weights", "context_available")])
        torch.testing.assert_close(adapted, original, rtol=0, atol=0)
        torch.testing.assert_close(adapted_anchor, original_anchor, rtol=0, atol=0)
        # Window order must be irrelevant to record-level Deep Sets pooling.
        permuted = dict(inputs)
        for key in ("context", "context_time_mask", "context_window_weights"):
            permuted[key] = inputs[key][:, [2, 0, 1]]
        emb = model.encode_body_conditions(*[inputs[key] for key in
            ("anchor", "numeric", "sex", "field_mask", "age_topcoded", "context",
             "context_time_mask", "context_window_weights", "context_available")]).body_embedding
        perm_emb = model.encode_body_conditions(*[permuted[key] for key in
            ("anchor", "numeric", "sex", "field_mask", "age_topcoded", "context",
             "context_time_mask", "context_window_weights", "context_available")]).body_embedding
        torch.testing.assert_close(emb, perm_emb, rtol=1e-5, atol=1e-6)
        masked = dict(inputs)
        masked["context_time_mask"] = inputs["context_time_mask"].clone()
        masked["context_time_mask"][:, -1, :, length // 2:] = False
        counts = masked["context_time_mask"].sum(dim=(2, 3)).float()
        masked["context_window_weights"] = counts / counts.sum(dim=1, keepdim=True)
        poisoned = dict(masked)
        poisoned["context"] = inputs["context"].clone()
        poisoned["context"][~masked["context_time_mask"]] = float("nan")
        clean_emb = model.body_encoder(masked["context"], masked["context_time_mask"],
                                       masked["context_window_weights"], masked["context_available"])
        poison_emb = model.body_encoder(poisoned["context"], poisoned["context_time_mask"],
                                        poisoned["context_window_weights"], poisoned["context_available"])
        torch.testing.assert_close(clean_emb, poison_emb, rtol=0, atol=0)
        output = sample_body(model, inputs, ["a", "b"], steps=1, samples=1)
        assert output.shape == (batch, 12, length) and torch.isfinite(output).all()
    # Initial zero FiLM learns first; after a nonzero FiLM update the encoder gets gradients.
    model.train()
    loss = model.forward_body(state, time, *[inputs[key] for key in
        ("anchor", "numeric", "sex", "field_mask", "age_topcoded", "context",
         "context_time_mask", "context_window_weights", "context_available")])[0].square().mean()
    loss.backward()
    assert model.body_film.projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.body_film.projection.weight.add_(0.01 * torch.randn_like(model.body_film.projection.weight))
    model.zero_grad(set_to_none=True)
    loss = model.forward_body(state, time, *[inputs[key] for key in
        ("anchor", "numeric", "sex", "field_mask", "age_topcoded", "context",
         "context_time_mask", "context_window_weights", "context_available")])[0].square().mean()
    loss.backward()
    assert model.body_encoder.cnn[0].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        changed = model.forward_body(state, time, *[inputs[key] for key in
            ("anchor", "numeric", "sex", "field_mask", "age_topcoded", "context",
             "context_time_mask", "context_window_weights", "context_available")])[0]
        assert not torch.equal(changed, original)
    absent = dict(inputs)
    absent["context_time_mask"] = torch.zeros_like(inputs["context_time_mask"])
    absent["context_window_weights"] = torch.zeros_like(inputs["context_window_weights"])
    absent["context_available"] = torch.zeros_like(inputs["context_available"])
    with torch.no_grad():
        fallback = model.forward_body(state, time, *[absent[key] for key in
            ("anchor", "numeric", "sex", "field_mask", "age_topcoded", "context",
             "context_time_mask", "context_window_weights", "context_available")])[0]
        own_base = model(state, time, *[inputs[key] for key in
            ("anchor", "numeric", "sex", "field_mask", "age_topcoded")])[0]
        torch.testing.assert_close(fallback, own_base, rtol=0, atol=0)
    print("CPU model: initial B5 identity, permutation invariance, sampling, gradients, fallback OK")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_data()
    check_model()
