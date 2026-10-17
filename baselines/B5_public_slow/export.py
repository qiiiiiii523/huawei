"""Deployable bundle: unchanged control100 plus selected CMP slow weights."""
import argparse
import copy
from pathlib import Path

import torch

from baselines.B5.checkpoint import atomic_save
from baselines.B5_best.bundle import validate_bundle
from baselines.B5_best.common import write, sha, state_sha, read
from .common import load_study, model_from_bundle
from . import PROTOCOL


def export(study_path, checkpoint, output):
    study, original = load_study(study_path)
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Use a new bundle output")
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (cp.get("protocol") != PROTOCOL or cp.get("study_sha256") != sha(study_path) or
            cp.get("FM_state_sha256") != original["fm_state_sha256"]):
        raise ValueError("Checkpoint/study/FM identity mismatch")
    model = model_from_bundle(original, torch.device("cpu"))
    model.load_state_dict(cp["model"], strict=True)
    score = cp.get("validation_score")
    if score is None or score["score_protocol"] != study["score_protocol"]:
        raise ValueError("Export a scored best.pt, not an unscored latest state")
    bundle = copy.deepcopy(original)
    bundle["slow"]["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bundle["slow_state_sha256"] = state_sha(bundle["slow"]["model"])
    bundle["reference_qc_r"] = {k: score[k] for k in ("task1_r_missing11", "task2_r_missing11")}
    bundle["reference_score_protocol"] = study["score_protocol"]
    bundle["provenance"].update(experiment=PROTOCOL, experiment_group=cp["group"],
        source_updated_slow_stage=cp["stage"], source_updated_slow_step=cp["step"],
        slow_sha256=sha(checkpoint), original_slow_sha256=original["provenance"]["slow_sha256"],
        study_sha256=sha(study_path), FM_frozen=True, stable_reference=False)
    validate_bundle(bundle)
    atomic_save(output, bundle)
    write(output.with_suffix(".json"), {"bundle_sha256": sha(output), "checkpoint_sha256": sha(checkpoint),
          "group": cp["group"], "stage": cp["stage"], "step": cp["step"], "score": score,
          "FM_unchanged": True, "target_required_at_inference": False, "cache_required_at_inference": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    export(a.study, a.checkpoint, a.output)
    print({"portable_bundle": a.output}, flush=True)


if __name__ == "__main__":
    main()
