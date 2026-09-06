from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ecg12gen.models.b0_linear import B0Linear
from ecg12gen.b0_route_loss import B0RouteLoss


def main():
    torch.manual_seed(42)
    loss_fn = B0RouteLoss(ROOT / "configs/losses.yaml")

    for channels in (1, 6):
        for profile in ("strict_sync", "L0", "L1"):
            model = B0Linear(channels)
            inputs = torch.randn(2, channels, 5000)
            prediction = model(inputs)

            observed = torch.zeros_like(prediction)
            observed[:, :channels] = inputs

            mask = torch.zeros(2, 12, dtype=torch.bool)
            mask[:, :channels] = True

            is_strict = profile == "strict_sync"
            target = torch.randn_like(prediction)

            # 同步情况下，输入与目标的可见导联保持一致。
            if is_strict:
                target[:, :channels] = inputs

            loss, logs = loss_fn(
                prediction,
                profile=profile,
                lead_mask=mask,
                observed_input=observed,
                d12_scale_uV=torch.ones(12) * 1000,
                alignment_mode=(
                    "same_window"
                    if is_strict
                    else "weak_subject_pair_record_start"
                ),
                pointwise_loss_allowed=is_strict,
                target=target,
                strict_reference=torch.randn(3, 12, 5000),
            )

            loss.backward()
            grad = model.mapping.weight.grad

            assert grad is not None
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0

            if profile == "L0":
                assert "pair_invariant_stat" not in logs
                assert "huber_missing" not in logs
                assert "pcc_missing" not in logs
            elif profile == "L1":
                assert "pair_invariant_stat" in logs
                assert "huber_missing" not in logs
                assert "pcc_missing" not in logs

            print(f"PASS: channels={channels}, profile={profile}")

    # 尝试将弱配对数据作为同步数据监督，必须被拒绝。
    try:
        loss_fn(
            prediction,
            profile="strict_sync",
            lead_mask=mask,
            observed_input=observed,
            d12_scale_uV=torch.ones(12) * 1000,
            alignment_mode="weak_subject_pair_record_start",
            pointwise_loss_allowed=False,
            target=target,
        )
    except ValueError:
        print("PASS: weak-pair pointwise supervision rejected")
    else:
        raise AssertionError("Unsafe supervision was not rejected")

    print("All B0 loss checks passed.")


if __name__ == "__main__":
    main()