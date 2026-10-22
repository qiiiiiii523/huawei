"""Target, reverse-reconstruction and representation-consistency objectives."""
from __future__ import annotations

import torch

from ecg12gen.losses import masked_huber_loss, masked_pcc_loss
from .model import AuroraRegression


def _masked_representation_l1(left: torch.Tensor, right: torch.Tensor,
                              mask: torch.Tensor | None = None) -> torch.Tensor:
    difference = (left - right).abs()
    if mask is None:
        return difference.mean()
    weight = mask.to(difference.dtype).unsqueeze(-1)
    return (difference * weight).sum() / (weight.sum().clamp_min(1) * difference.shape[-1])


def aurora_loss(model: AuroraRegression, batch: dict[str, torch.Tensor], weights: dict[str, float],
                d12_scale: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target, mask = batch["target"], batch["quality_mask"].bool()
    if target.ndim != 3 or target.shape[1] != 12 or mask.shape != target.shape[:2]:
        raise ValueError("Expected normalized target [B,12,T] and quality mask [B,12]")
    output = model(batch["anchor"], batch["numeric"], batch["sex"], batch["field_mask"],
                   batch["age_topcoded"], d12_scale, compute_cycles=True)
    generated_mask = mask.clone()
    generated_mask[:, 0] = False
    safe_prediction = torch.where(generated_mask.unsqueeze(-1), output.full12, torch.zeros_like(output.full12))
    safe_target = torch.where(generated_mask.unsqueeze(-1), target, torch.zeros_like(target))
    generation = masked_huber_loss(safe_prediction, safe_target, generated_mask, float(weights["huber_delta"]))
    pcc = masked_pcc_loss(safe_prediction, safe_target, generated_mask)
    source_i = masked_huber_loss(output.cycles["source_i"], batch["anchor"],
                                 torch.ones_like(mask[:, :1]), float(weights["huber_delta"]))
    limb_mask = mask[:, :6].clone()
    limb_mask[:, 0] = False
    limb_cycle = masked_huber_loss(
        torch.where(limb_mask[:, :, None], output.cycles["limb6"], torch.zeros_like(output.cycles["limb6"])),
        torch.where(limb_mask[:, :, None], target[:, :6], torch.zeros_like(target[:, :6])),
        limb_mask, float(weights["huber_delta"]))
    representation = (
        _masked_representation_l1(output.representations["limb_forward"],
                                  output.representations["limb_generated"]) +
        _masked_representation_l1(output.representations["limb_reverse"],
                                  output.representations["limb_source"]) +
        _masked_representation_l1(output.representations["chest_forward"],
                                  output.representations["chest_generated"], mask[:, 6:]) +
        _masked_representation_l1(output.representations["chest_reverse"],
                                  output.representations["chest_source"], mask[:, :6])
    ) / 4
    parts = {"generation": generation, "pcc": pcc, "source_reconstruction": source_i + limb_cycle,
             "representation_consistency": representation}
    total = (float(weights["generation"]) * generation + float(weights["pcc"]) * pcc +
             float(weights["source_reconstruction"]) * parts["source_reconstruction"] +
             float(weights["representation_consistency"]) * representation)
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite AURORA regression loss")
    return total, parts
