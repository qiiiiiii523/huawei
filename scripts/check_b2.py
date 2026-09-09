"""Non-training B2-v1 checks for the latest main joint-anchor contract."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b2_model import B2MaskedPatchTransformer
from ecg12gen.b2_train import P0_STAGE
from ecg12gen.losses import joint_anchor_sync_loss, strict_anchor_pretrain_loss


def main() -> None:
    torch.manual_seed(42)
    model = B2MaskedPatchTransformer()
    assert 250_000 <= model.parameter_count <= 400_000
    anchor = torch.randn(2, 1, 5000)
    context = torch.randn(2, 1, 5000)
    anchor_mask = torch.tensor([[True] + [False] * 11] * 2)
    context_mask = torch.ones(2, 1, dtype=torch.bool)
    p0 = model.forward_anchor(anchor, anchor_mask)
    p1 = model(context, anchor, context_mask, anchor_mask)
    assert p0.shape == p1.shape == (2, 12, 5000)
    assert not any(parameter.requires_grad for name, parameter in model.named_buffers() if name == "positional_encoding")
    target = torch.randn(2, 12, 5000)
    scale = torch.linspace(200.0, 900.0, 12)
    assert torch.isfinite(strict_anchor_pretrain_loss(p0, target, anchor, d12_scale_uV=scale))
    assert torch.isfinite(joint_anchor_sync_loss(p1, target, anchor, d12_scale_uV=scale))
    source = inspect.getsource(model.forward_joint)
    assert "torch.cat" not in source and "context_tokens" in source
    predict_source = (ROOT / "scripts" / "predict_b2.py").read_text(encoding="utf-8")
    assert "target input is unsupported" in predict_source and "target_baseline" not in predict_source
    print(f"PASS: B2-v1 joint-anchor interface; stage={P0_STAGE}; parameters={model.parameter_count}")


if __name__ == "__main__":
    main()
