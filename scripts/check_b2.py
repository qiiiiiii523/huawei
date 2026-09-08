"""Non-training B2 P0/C1/C2, leakage, and task2 dual-context smoke test."""
from __future__ import annotations
import inspect
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ecg12gen.b2_data import (DualContextIntersectionError, build_joint_dataset, build_strict_dataset,
                               fit_b2_preprocessor, task2_dual_intersection)
from ecg12gen.b2_model import B2JointAnchorPatchTransformer, B2ModelConfig
from ecg12gen.b2_train import P0_STAGE, P1_STAGE, _forward, fit_b2
from ecg12gen.evaluate import evaluate_joint_anchor_predictions
from ecg12gen.losses import joint_anchor_sync_loss, strict_anchor_pretrain_loss
from scripts.predict_b2 import _explicit

def _batch(n: int = 2) -> dict[str, torch.Tensor]:
    return {"anchor_model": torch.randn(n,1,5000), "target_model": torch.randn(n,12,5000), "anchor_lead_mask": torch.tensor([[True]+[False]*11]*n),
            "watch_context_model": torch.randn(n,1,5000), "watch_available": torch.ones(n,dtype=torch.bool),
            "machine_d6_model": torch.randn(n,6,5000), "machine_d6_mask": torch.ones(n,6,dtype=torch.bool), "machine_available": torch.ones(n,dtype=torch.bool),
            "body_d6_model": torch.randn(n,6,5000), "body_d6_mask": torch.ones(n,6,dtype=torch.bool), "body_available": torch.ones(n,dtype=torch.bool)}

def main() -> None:
    torch.manual_seed(42); p0 = B2JointAnchorPatchTransformer(B2ModelConfig(fusion_mode="none")); c2 = B2JointAnchorPatchTransformer(B2ModelConfig(fusion_mode="film_gated_residual")); c2.load_state_dict(p0.state_dict()); p0.eval(); c2.eval()
    batch = _batch(); raw_p0, raw_c2 = _forward(p0,batch,"task1"), _forward(c2,batch,"task1")
    decoded = []
    hook = p0.lead_decoder.register_forward_hook(lambda _module, _inputs, output: decoded.append(output.detach()))
    ordered = _forward(p0, batch, "task1")
    hook.remove()
    assert len(decoded) == 1 and torch.allclose(ordered, decoded[0].reshape(2, 12, 5000), atol=1e-6)
    assert raw_p0.shape == (2,12,5000) and torch.allclose(raw_p0, raw_c2, atol=1e-6)
    assert abs(float(torch.sigmoid(c2.gate.bias).mean().detach()) - .03) < .002
    assert _forward(c2,batch,"task2").shape == (2,12,5000)
    assert torch.isfinite(strict_anchor_pretrain_loss(raw_p0,batch["target_model"],batch["anchor_model"]))
    assert torch.isfinite(joint_anchor_sync_loss(raw_c2,batch["target_model"],batch["anchor_model"]))
    changed=batch["target_model"].clone(); changed[:,:1]+=100; assert not torch.allclose(joint_anchor_sync_loss(raw_c2,changed,batch["anchor_model"]), joint_anchor_sync_loss(raw_c2,batch["target_model"],batch["anchor_model"]))
    pre1=fit_b2_preprocessor(ROOT/"configs"/"common.yaml","task1"); strict=build_strict_dataset(ROOT/"configs"/"common.yaml",pre1); t1=build_joint_dataset(ROOT/"configs"/"common.yaml","task1","validation",pre1,context_view="shuffle_watch")
    assert strict[0].anchor_model.shape==(1,5000) and not strict[0].watch_available and t1[0].meta["context_shuffled"] and t1[0].meta["context_subject_id"] != t1[0].meta["subject_id"]
    pre2=fit_b2_preprocessor(ROOT/"configs"/"common.yaml","task2"); machine=build_joint_dataset(ROOT/"configs"/"common.yaml","task2","validation",pre2,context_view="machine"); body=build_joint_dataset(ROOT/"configs"/"common.yaml","task2","validation",pre2,context_view="body")
    assert machine[0].machine_available and not machine[0].body_available and body[0].body_available and not body[0].machine_available
    pairs_train, counts_train=task2_dual_intersection(ROOT/"configs"/"common.yaml","train"); pairs_val, counts_val=task2_dual_intersection(ROOT/"configs"/"common.yaml","validation")
    assert counts_train["both_rows"] == len(pairs_train) and counts_val["both_rows"] == len(pairs_val)
    try: build_joint_dataset(ROOT/"configs"/"common.yaml","task2","train",pre2,context_view="both"); assert pairs_train
    except DualContextIntersectionError: assert not pairs_train
    try: fit_b2(c2,machine,machine,pre2.scale_uV_by_source["d12"],"task2",ROOT/"tmp_should_not_exist",stage=P1_STAGE); raise AssertionError("P1 without P0 accepted")
    except ValueError: pass
    missing_anchor=subprocess.run([sys.executable,str(ROOT/"scripts"/"predict_b2.py"),"--checkpoint","unused.pt","--task-id","task1","--output-dir","unused"],capture_output=True,text=True)
    assert missing_anchor.returncode != 0 and "--anchor-npy" in (missing_anchor.stdout+missing_anchor.stderr)
    target=np.random.default_rng(42).normal(size=(2,12,5000)).astype(np.float32); anchor=np.ones((2,1,5000),np.float32); summary,_,_,submit=evaluate_joint_anchor_predictions(target,target,anchor,"task1"); assert np.array_equal(submit[:,:1],anchor) and summary["prediction_view"]=="submit_anchor_i_replaced"
    assert "target" not in inspect.signature(_explicit).parameters
    source=(ROOT/"ecg12gen"/"b2_train.py").read_text(encoding="utf-8"); assert "replace_output_i_with_anchor" not in source and "rpeak" not in source and "raw_weak" not in source
    print(f"PASS: B2 P0/C1/C2; strict={len(strict)}; T2 machine/body={len(machine)}/{len(body)}; dual train/val={counts_train['both_rows']}/{counts_val['both_rows']}")
if __name__ == "__main__": main()
