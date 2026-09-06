from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ecg12gen.losses import (
    masked_huber_loss,
    masked_pcc_loss,
    observed_consistency_loss,
    pair_invariant_stat_loss,
    physiology_constraint_loss,
    spectral_stat_loss,
)


class B0RouteLoss:
    """
    B0损失调度：严格同步、弱配对L0、弱配对L1。

    prediction:
        [B, 12, T]，按d12训练尺度归一化的预测。

    observed_input:
        [B, 12, T]，可见输入转换到d12尺度后的结果。
        缺失导联填0，只通过lead_mask选择可见导联。

    d12_scale_uV:
        [12]，冻结的d12训练尺度。

    注意：
        弱配对目标仅允许用于L1统计损失；
        不允许用于逐点Huber/PCC。
    """

    PROFILE_NAMES = {
        "strict_sync": "strict_sync",
        "L0": "raw_weak_a0",
        "L1": "raw_weak_a1",
    }

    def __init__(self, config_path: str | Path) -> None:
        with Path(config_path).open(
            "r", encoding="utf-8"
        ) as handle:
            self.weights = yaml.safe_load(handle)

        for name in self.PROFILE_NAMES.values():
            if name not in self.weights:
                raise ValueError(f"Missing loss profile: {name}")

    def __call__(
        self,
        prediction: torch.Tensor,
        *,
        profile: str,
        lead_mask: torch.Tensor,
        observed_input: torch.Tensor,
        d12_scale_uV: torch.Tensor,
        alignment_mode: str,
        pointwise_loss_allowed: bool,
        target: torch.Tensor | None = None,
        strict_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if profile not in self.PROFILE_NAMES:
            raise ValueError(f"Unknown loss profile: {profile}")

        if prediction.ndim != 3 or prediction.shape[1] != 12:
            raise ValueError("prediction must be [B, 12, T]")

        if lead_mask.shape != prediction.shape[:2]:
            raise ValueError("lead_mask must be [B, 12]")

        if lead_mask.dtype != torch.bool:
            raise ValueError("lead_mask must be boolean")

        if observed_input.shape != prediction.shape:
            raise ValueError(
                "observed_input must match prediction shape"
            )

        scale = torch.as_tensor(
            d12_scale_uV,
            device=prediction.device,
            dtype=prediction.dtype,
        ).detach()

        if scale.shape != (12,):
            raise ValueError("d12_scale_uV must have shape [12]")

        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("d12 scales must be finite and positive")

        weights = self.weights[self.PROFILE_NAMES[profile]]
        missing_mask = ~lead_mask

        # 每个导联的归一化尺度不同。
        # 生理代数约束应先恢复共同物理尺度，再统一除以1 mV，
        # 不能直接约束各导联分别归一化后的数值。
        prediction_mV = prediction * scale[None, :, None] / 1000.0

        components = {
            "observed_consistency": observed_consistency_loss(
                prediction,
                observed_input.detach(),
                lead_mask,
            ),
            "physiology": physiology_constraint_loss(
                prediction_mV
            ),
        }

        if profile == "strict_sync":
            if (
                alignment_mode != "same_window"
                or not pointwise_loss_allowed
            ):
                raise ValueError(
                    "strict_sync requires same-window supervision"
                )

            if target is None or target.shape != prediction.shape:
                raise ValueError(
                    "strict_sync requires a matching D12 target"
                )

            target = target.detach()

            components["huber_missing"] = masked_huber_loss(
                prediction, target, missing_mask
            )
            components["pcc_missing"] = masked_pcc_loss(
                prediction, target, missing_mask
            )

        else:
            if (
                alignment_mode != "weak_subject_pair_record_start"
                or pointwise_loss_allowed
            ):
                raise ValueError(
                    "Weak profiles require weak alignment "
                    "and forbid pointwise target loss"
                )

            if strict_reference is None:
                raise ValueError(
                    "L0/L1 require an independent strict-train "
                    "D12 reference batch"
                )

            components["spectral_stat"] = spectral_stat_loss(
                prediction,
                strict_reference.detach(),
            )

            if profile == "L1":
                if target is None or target.shape != prediction.shape:
                    raise ValueError(
                        "L1 requires a matching paired target "
                        "for statistics only"
                    )

                components["pair_invariant_stat"] = (
                    pair_invariant_stat_loss(
                        prediction,
                        target.detach(),
                    )
                )

        total = prediction.new_zeros(())

        for name, value in components.items():
            total = total + float(weights[name]) * value

        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite B0 loss")

        logs = {
            name: float(value.detach().cpu())
            for name, value in components.items()
        }
        logs["total"] = float(total.detach().cpu())

        return total, logs