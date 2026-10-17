"""Validate the scratch-control EMA and create self-contained hybrid candidates."""
import copy
import hashlib

import torch

from baselines.B5.checkpoint import atomic_save, scales_digest
from baselines.B5.config import ModelConfig
from baselines.B5_best import SAMPLING
from baselines.B5_best.bundle import validate_bundle
from baselines.B5_best.common import read, sha, state_sha, write


def legacy_state_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def control_payload(path, evaluation, original):
    """Scratch control stores EMA in `model`, unlike an ordinary B5 checkpoint."""
    cp = torch.load(path, map_location="cpu", weights_only=True)
    report = read(evaluation / "evaluation.json")
    if (cp.get("protocol") != "B5-scratch-early-public-v1" or cp.get("group") != "control"
            or cp.get("slow_variant") != "none" or cp.get("epoch") != 100
            or cp.get("sampling") != SAMPLING):
        raise ValueError("Expected the completed scratch CONTROL epoch100, Heun16/K16/seed42 checkpoint")
    if (not report.get("completed") or report.get("epoch") != 100
            or report.get("target_used_at_inference") is not False
            or report.get("checkpoint_sha256") != sha(path)):
        raise ValueError("Control100 evaluation does not identify this checkpoint")
    fingerprint = report.get("fingerprint", {})
    if (fingerprint.get("group") != "control" or fingerprint.get("sampling") != SAMPLING
            or fingerprint.get("model_ema_sha256") != cp["model_ema_sha256"]
            or legacy_state_sha(cp["model"]) != cp["model_ema_sha256"]):
        raise ValueError("Control100 EMA/checkpoint/sampling fingerprint mismatch")
    base = original["fm"]
    if (ModelConfig.from_dict(cp["model_config"]).fingerprint != base["architecture_hash"]
            or cp.get("base_architecture_hash") != base["architecture_hash"]
            or cp.get("scales_sha256") != scales_digest(cp["scales"])
            or cp["scales_sha256"] != base["scales_sha256"]):
        raise ValueError("Control100 architecture or frozen scales differ")
    if set(cp["model"]) != set(base["ema"]):
        raise ValueError("Only the original U-Net control is compatible; no slow adapters/V13")
    for key, value in cp["model"].items():
        if value.shape != base["ema"][key].shape or not torch.isfinite(value).all():
            raise ValueError("Invalid control EMA tensor: " + key)
    payload = copy.deepcopy(base)
    payload["ema"] = cp["model"]
    return payload, report


def candidate(original, fm, slow_state, fm_name, slow_name, provenance):
    bundle = copy.deepcopy(original)
    bundle["fm"] = fm
    bundle["slow"]["model"] = slow_state
    bundle["fm_state_sha256"] = state_sha(fm["ema"])
    bundle["slow_state_sha256"] = state_sha(slow_state)
    bundle["slow_beta"] = .75
    bundle["reference_qc_r"] = None
    bundle["provenance"].update(provenance, experiment="component_crossover",
                                stable_reference=False, fm_component=fm_name, slow_component=slow_name)
    bundle["provenance"]["original_fm_sha256"] = original["provenance"]["fm_sha256"]
    bundle["provenance"]["original_slow_sha256"] = original["provenance"]["slow_sha256"]
    if fm_name == "control100":
        bundle["provenance"].update(fm_sha256=provenance["control100_checkpoint_sha256"], source_fm_epoch=100)
    if slow_name != "original":
        bundle["provenance"].update(slow_sha256=provenance[slow_name + "_sha256"],
                                    source_slow_epoch=provenance[slow_name + "_epoch"])
    validate_bundle(bundle)
    return bundle


def save_candidate(path, bundle, score, seed):
    # Portable deployment uses the unchanged seed42 protocol. Seed43 is an evaluation probe.
    bundle = copy.deepcopy(bundle)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Candidate bundle already exists: " + str(path))
    if seed == 42:
        bundle["reference_qc_r"] = {k: score[k] for k in ("task1_r_missing11", "task2_r_missing11")}
    bundle["provenance"].update(evaluation_seed=seed, evaluation_score=score)
    validate_bundle(bundle)
    atomic_save(path, bundle)
    write(path.with_suffix(".json"), {"bundle_sha256": sha(path), "sampling": bundle["sampling"],
          "evaluation_seed": seed, "evaluation_score": score, "beta": .75,
          "target_required_at_inference": False, "training_cache_required_at_inference": False,
          "provenance": bundle["provenance"]})
