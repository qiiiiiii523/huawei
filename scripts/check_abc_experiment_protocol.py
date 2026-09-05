"""Non-training checks for A/B/C shared experiment infrastructure."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.canonical_adapter import canonicalize_input_ecg
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.losses import masked_huber_loss, masked_pcc_loss, pair_invariant_stat_loss, physiology_constraint_loss, spectral_stat_loss
from ecg12gen.rpeak_pseudopair import beats, monotonic_match, target_time_mapping, warp_d12_to_input_grid


def main() -> None:
    with (ROOT / "metadata" / "d12_strict_pretrain_index.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["source_split"] == "train" for row in rows)
    assert len({row["dedup_key"] for row in rows}) == len(rows)
    canonical, mask = canonicalize_input_ecg(np.zeros((2, 6, 5000), dtype=np.float32))
    assert canonical.shape == (2, 12, 5000) and mask.shape == (2, 12) and mask[:, :6].all() and not mask[:, 6:].any()
    strict = StrictD12PretrainDataset(ROOT / "configs" / "common.yaml", "d12_i_pretrain")
    assert len(strict) == len(rows) and strict[0].Y_12lead.shape == (12, 5000)

    for name in ("task1_arm_a_weak.yaml", "task1_arm_a_weak_a1.yaml", "task1_arm_b_sync_weak.yaml", "task1_arm_c_sync_rpeak.yaml", "task2_arm_a_weak.yaml", "task2_arm_a_weak_a1.yaml", "task2_arm_b_sync_weak.yaml"):
        assert (ROOT / "configs" / "experiments" / name).exists()

    assert (ROOT / "configs" / "experiments" / "task1_arm_c_sync_rpeak_c2.yaml").exists()
    c1 = yaml.safe_load((ROOT / "configs" / "experiments" / "task1_arm_c_sync_rpeak.yaml").read_text(encoding="utf-8"))
    c2 = yaml.safe_load((ROOT / "configs" / "experiments" / "task1_arm_c_sync_rpeak_c2.yaml").read_text(encoding="utf-8"))
    assert c1["experiment_variant"] == "C1" and c1["pseudo_data"]["expected_windows"] == 18
    assert c2["experiment_variant"] == "C2" and c2["pseudo_data"]["expected_windows"] == 33
    assert c2["pseudo_data"]["c1_windows_preserved"] == 18
    assert c2["pseudo_data"]["added_medium_quality_windows"] == 15
    assert c2["pseudo_data"]["min_morphology_similarity"] == 0.90 and c2["pseudo_data"]["min_alignment_quality"] == 0.85

    losses = yaml.safe_load((ROOT / "configs" / "losses.yaml").read_text(encoding="utf-8"))
    assert losses["raw_weak_a0"]["observed_consistency"] == 0.10
    assert losses["raw_weak_a0"]["spectral_stat"] == 0.20 and losses["raw_weak_a0"]["physiology"] == 0.05
    assert losses["raw_weak_a1"]["pair_invariant_stat"] == 0.20

    prediction, target = torch.randn(2, 12, 500), torch.randn(2, 12, 500)
    missing = torch.tensor([[False] + [True] * 11, [False] + [True] * 11])
    assert torch.isfinite(masked_huber_loss(prediction, target, missing))
    assert torch.isfinite(masked_pcc_loss(prediction, target, missing))
    assert torch.isfinite(spectral_stat_loss(prediction, target))
    assert torch.isfinite(pair_invariant_stat_loss(prediction, target))
    assert torch.allclose(pair_invariant_stat_loss(prediction, target), pair_invariant_stat_loss(prediction, torch.roll(target, shifts=17, dims=-1)), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(physiology_constraint_loss(prediction))

    time = np.arange(5000, dtype=np.float32)
    signal = np.sin(time / 25)
    source_beats = beats(signal, [500, 1200, 1900, 2600, 3300, 4000], 200, 300)
    target_beats = beats(signal, [500, 1200, 1900, 2600, 3300, 4000], 200, 300)
    matched = monotonic_match(source_beats, target_beats)
    mapping = target_time_mapping(matched, source_beats, target_beats, 5000)
    assert len(matched) == 6 and mapping is not None and np.all(np.diff(mapping) > 0)
    assert warp_d12_to_input_grid(np.broadcast_to(signal, (12, 5000)), mapping).shape == (12, 5000)
    print(f"PASS: A/B/C configs; {len(rows)} strict train d12 windows; loss contract; monotonic pseudo-pair primitives")


if __name__ == "__main__":
    main()
