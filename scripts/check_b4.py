"""Fast synthetic functionality check for B4."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b4_diffusion import B4Diffusion
from ecg12gen.b4_model import B4ConditionalUNet1D


def main() -> None:
    torch.manual_seed(42)
    model = B4ConditionalUNet1D({
        "base_channels": 8,
        "channel_multipliers": [1, 2, 4],
        "time_embedding_dim": 16,
        "dropout": 0.0,
    })
    diffusion = B4Diffusion(model, training_steps=8)
    anchor = torch.randn(2, 1, 64)
    missing_target = torch.randn(2, 11, 64)
    quality = torch.ones(2, 11, dtype=torch.bool)
    loss = diffusion.training_loss(missing_target, anchor, quality, timesteps=torch.tensor([0, 7]))
    loss.backward()
    assert torch.isfinite(loss)
    sampled = diffusion.sample(anchor, sampling_steps=4, generator=torch.Generator().manual_seed(42))
    assert sampled.shape == (2, 12, 64)
    assert torch.equal(sampled[:, :1], anchor)
    assert model.architecture_metadata["diffused_lead_indices"] == list(range(1, 12))
    assert "--target" not in (ROOT / "scripts" / "predict_b4.py").read_text(encoding="utf-8")
    print("PASS: missing11-only loss; reverse sampling; exact I reinsertion; [B,12,T] output")


if __name__ == "__main__":
    main()
