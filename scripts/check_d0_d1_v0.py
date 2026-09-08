"""Read-only joint-anchor integration check."""
from __future__ import annotations
import csv
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ecg12gen.contracts import ContractError, SupervisionMode, prepare_joint_anchor_inference
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.evaluate import evaluate_joint_anchor_predictions, evaluate_predictions
from ecg12gen.losses import joint_anchor_sync_loss, replace_output_i_with_anchor, strict_anchor_pretrain_loss

def main() -> None:
    cfg = ECGDataConfig.from_yaml(ROOT / "configs" / "common.yaml")
    # Public metadata must remain exactly as versioned; data readers use mmap="r".
    assert subprocess.run(["git", "diff", "--quiet", "--", "metadata/subject_split.csv", "metadata/d12_strict_pretrain_index.csv"], cwd=ROOT).returncode == 0
    strict = StrictD12PretrainDataset(cfg, SupervisionMode.D12_I_PRETRAIN.value)
    with (ROOT / "metadata" / "d12_strict_pretrain_index.csv").open(encoding="utf-8-sig", newline="") as h: rows = list(csv.DictReader(h))
    assert rows and len(rows) == len(strict) and all(r["source_split"] == "train" for r in rows) and len({r["dedup_key"] for r in rows}) == len(rows)
    datasets = [JointAnchorDataset(cfg, task, split) for task in ("task1", "task2") for split in ("train", "validation")]
    five = JointAnchorDataset(cfg, "task2", "train", context_channel_indices=(1,2,3,4,5))
    for data in datasets + [five]:
        sample = data[0]; assert len(data) > 0 and np.array_equal(sample.anchor_i_ecg, sample.Y_12lead[:1])
        assert sample.anchor_target_sync and not sample.context_target_sync and sample.pointwise_loss_allowed
        assert sample.anchor_lead_mask.tolist() == [True] + [False] * 11
        assert sample.meta["anchor_construction"] == "simulated_from_target_i_for_test_available_input"
    assert five[0].context_ecg.shape == (5, 5000)
    assert torch.equal(replace_output_i_with_anchor(torch.zeros((1,12,5000)), torch.ones((1,1,5000)))[:, :1], torch.ones((1,1,5000)))
    try: prepare_joint_anchor_inference(np.zeros((1,5000), np.float32), task_id="task1", context_source_type="watch_ecg", anchor_i_ecg=None); raise AssertionError("missing anchor accepted")
    except ContractError: pass
    try: prepare_joint_anchor_inference(np.zeros((1,5000), np.float32), task_id="task1", context_source_type="watch_ecg", anchor_i_ecg=np.zeros((1,5000)), target=np.zeros((12,5000))); raise AssertionError("target accepted")
    except TypeError: pass
    target = np.linspace(-1, 1, 2 * 12 * 5000, dtype=np.float32).reshape(2,12,5000); overall, details = evaluate_predictions(target, target, "task2")
    assert overall["evaluation_input_contract"] == "joint_anchor_test_like" and all(not row["input_present"] for row in details[1:])
    raw = target.copy(); raw[:, :1] *= -1
    summary, raw_details, submit_details, submit = evaluate_joint_anchor_predictions(raw, target, target[:, :1], "task2")
    assert summary["r_submit_12"] > summary["r_raw_12"] and summary["r_missing11"] == np.mean([row["pearson_r"] for row in raw_details[1:]])
    prediction_t, target_t, anchor_t = torch.randn(2,12,500), torch.randn(2,12,500), torch.randn(2,1,500)
    assert torch.isfinite(strict_anchor_pretrain_loss(prediction_t, target_t, anchor_t))
    assert torch.isfinite(joint_anchor_sync_loss(prediction_t, target_t, anchor_t))
    legacy = ["configs/rpeak_pseudopair.yaml", "ecg12gen/rpeak_pseudopair.py", "scripts/build_rpeak_pseudopairs.py", "configs/experiments/task1_arm_a_weak.yaml"]
    assert not any((ROOT / p).exists() for p in legacy)
    print(f"PASS: strict train-only index and joint-anchor contract ({len(strict)} strict rows)")
if __name__ == "__main__": main()
