"""Synthetic B3 contract, shape and forward/backward check."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.contracts import ContractError
from ecg12gen.evaluate import evaluate_joint_anchor_predictions
from ecg12gen.losses import joint_anchor_sync_loss, strict_anchor_pretrain_loss
from ecg12gen.b3_model import B3Model
from ecg12gen.b3_train import _p1_lr_multiplier, train_b3


def _assert_grad(model: B3Model, prediction: torch.Tensor, target: torch.Tensor, anchor: torch.Tensor) -> None:
    loss = joint_anchor_sync_loss(prediction, target, anchor)
    assert torch.isfinite(loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    optimizer.step()


def main() -> None:
    torch.manual_seed(42)
    anchor = torch.randn(1, 1, 5000)
    target = torch.randn(1, 12, 5000)
    f0, f1, f2 = B3Model(fusion_mode="none").cnn_encoder(anchor)
    assert f0.shape == (1, 64, 5000) and f1.shape == (1, 128, 1250) and f2.shape == (1, 256, 250)

    p0 = B3Model(fusion_mode="none")
    output = p0(anchor)
    assert output.shape == (1, 12, 5000)
    strict = strict_anchor_pretrain_loss(output, target, anchor)
    assert torch.isfinite(strict)
    strict.backward()
    assert any(parameter.grad is not None for parameter in p0.parameters())

    c3_init = B3Model(fusion_mode="film_gated_residual")
    assert abs(float(torch.sigmoid(c3_init.gate.bias[0]).detach()) - 0.05) < 1e-5
    assert torch.count_nonzero(c3_init.residual_adapter[-1].weight) == 0
    assert torch.count_nonzero(c3_init.residual_adapter[-1].bias) == 0

    # P1 is initialized from the exact P0 state; the strict loader is tested
    # with the same state dictionary used by the training checkpoint.
    for mode in ("none", "film_gated_residual"):
        model = B3Model(fusion_mode=mode)
        if mode != "none":
            model.load_state_dict(p0.state_dict(), strict=True)
        context = torch.randn(1, 1, 5000)
        if mode == "none":
            # Context is not read at all on this path, even if malformed.
            prediction = model(anchor, context_ecg=torch.randn(1, 6, 3), context_source_type="invalid")
        else:
            prediction = model(anchor, context_ecg=context, context_source_type="watch_ecg")
        assert prediction.shape == (1, 12, 5000)
        _assert_grad(model, prediction, target, anchor)

    # The explicit zero-context diagnostic must exactly reproduce the P0
    # anchor path before P1 optimization, without changing checkpoint shape.
    p0.eval()
    c3_zero = B3Model(fusion_mode="film_gated_residual")
    c3_zero.load_state_dict(p0.state_dict(), strict=True)
    c3_zero.eval()
    assert torch.allclose(c3_zero.forward_anchor_only(anchor), p0(anchor), atol=1e-6, rtol=1e-5)
    assert _p1_lr_multiplier(0, 100, 5, 0.05) == 0.2
    assert abs(_p1_lr_multiplier(99, 100, 5, 0.05) - 0.05) < 1e-8

    task2_mask = torch.tensor([[True, True, True, True, True, True]])
    for source in ("ecg_machine_d6", "body_scale_d6"):
        model = B3Model(fusion_mode="film_gated_residual")
        context = torch.randn(1, 6, 5000)
        prediction = model(anchor, context_ecg=context, context_source_type=source, context_lead_mask=task2_mask)
        assert prediction.shape == (1, 12, 5000)
    try:
        B3Model(fusion_mode="film_gated_residual")(anchor, context_ecg=torch.randn(1, 6, 5000),
                                     context_source_type="ecg_machine_d6",
                                     context_lead_mask=torch.tensor([[True, False, False, False, False, False]]))
        raise AssertionError("invalid d6 mask accepted")
    except ContractError:
        pass

    # Main V0 submit construction must make lead I exactly equal to raw anchor.
    summary, _, _, submit = evaluate_joint_anchor_predictions(
        torch.randn(1, 12, 5000).numpy(), target.numpy(), anchor.numpy(), "task1")
    assert torch.equal(torch.from_numpy(submit[:, :1]), anchor)
    assert summary["prediction_view"] == "submit_anchor_i_replaced"
    assert "--target" not in (ROOT / "scripts" / "predict_b3.py").read_text(encoding="utf-8")

    try:
        train_b3(SimpleNamespace(stage="P1-C3", fusion_mode="film_gated_residual", p0_checkpoint=None))
        raise AssertionError("P1-C3 without P0 checkpoint did not fail")
    except ContractError:
        pass
    print("PASS: B3-v2 shapes, strict/joint backward, zero-context path, LR schedule, source/mask selection, submit identity, and forced P0 checkpoint")


if __name__ == "__main__":
    main()
