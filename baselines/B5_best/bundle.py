"""Export a portable, inference-only bundle; never reads training waveforms."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from baselines.B5 import ARCHITECTURE_ID, CONDITION_SCHEMA, PREPROCESSING_VERSION
from baselines.B5.checkpoint import atomic_save, load_checkpoint, scales_digest, model_from_checkpoint
from baselines.B5.config import ModelConfig
from . import BUNDLE_PROTOCOL, LEGACY_BUNDLE_PROTOCOL, SAMPLING, SCORE_PROTOCOL
from .combine import validate_slow_beta
from .common import read, write, sha, state_sha
from .slow import load_slow

DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "configs/b5_best.json"


def verified_source(profile):
    """Resolve only the explicitly pinned portable package; no raw checkpoint fallback."""
    if "default_bundle" not in profile:
        raise ValueError("This legacy profile describes raw components; export a portable bundle first "
                         "and pass it with --checkpoint")
    if (profile.get("bundle_protocol") != BUNDLE_PROTOCOL or profile.get("sampling") != SAMPLING):
        raise ValueError("Default profile must preserve v2 and Heun16/K16/seed42")
    source = Path(profile["default_bundle"])
    if not source.is_file():
        raise FileNotFoundError("Verified default model bundle is missing: " + str(source)
                                + "; keep/copy the verified .pt or pass --checkpoint explicitly")
    if sha(source) != profile["default_bundle_sha256"]:
        raise ValueError("Default model bundle SHA256 differs from the verified package")
    bundle = load_bundle(source)
    if bundle_slow_beta(bundle) != bundle_slow_beta(profile):
        raise ValueError("Default package/profile slow_beta mismatch")
    reference = bundle.get("reference_qc_r")
    tolerance = float(profile.get("reference_tolerance", 2e-5))
    if reference is None or any(abs(reference[k] - v) > tolerance for k, v in profile["reference_qc_r"].items()):
        raise ValueError("Default package/profile reference r mismatch")
    for component, field in (("fm", "fm_sha256"), ("slow", "slow_sha256")):
        expected = profile.get("components", {}).get(component + "_sha256")
        if expected and bundle["provenance"].get(field) != expected:
            raise ValueError("Default package component provenance mismatch: " + component)
    return source.resolve(), bundle


def load_inference_bundle(checkpoint=None, profile_path=DEFAULT_PROFILE):
    """Explicit --checkpoint remains backward compatible; omitted means the promoted default."""
    if checkpoint is not None:
        path = Path(checkpoint).resolve()
        return load_bundle(path), path
    path, bundle = verified_source(read(profile_path))
    return bundle, path


def export_profile(profile_path, output):
    """Copy a verified portable model byte-for-byte, or export an explicit legacy profile."""
    profile, output = read(profile_path), Path(output)
    if "default_bundle" not in profile:
        return export_bundle(profile["fm_checkpoint"], profile["slow_checkpoint"],
                             profile["prepared_info"], output, profile)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new bundle path; existing weights are never overwritten")
    source, bundle = verified_source(profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    shutil.copyfile(source, temporary)
    if sha(temporary) != profile["default_bundle_sha256"]:
        raise ValueError("Copied default bundle checksum mismatch")
    os.replace(temporary, output)
    write(output.with_suffix(".json"), {k: v for k, v in bundle.items() if k not in ("fm", "slow")} |
          {"bundle_sha256": sha(output), "verified_default_source": str(source),
           "deployment_verification": profile.get("deployment_verification"),
           "training_started": False, "optimizer_updates": 0})
    return bundle


def export_bundle(fm_path, slow_path, prepared_info, output, profile, candidate=False):
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new bundle path; existing weights are never overwritten")
    if profile.get("bundle_protocol") != BUNDLE_PROTOCOL or profile.get("sampling") != SAMPLING:
        raise ValueError("Stable profile must use the B5-best protocol and Heun16/K16/seed42")
    beta = bundle_slow_beta(profile)
    fm_hash, slow_hash = sha(fm_path), sha(slow_path)
    if not candidate and (fm_hash != profile["fm_sha256"] or slow_hash != profile["slow_sha256"]):
        raise ValueError("Stable components differ from the verified pair; use --candidate and a new path for experiments")
    fm = load_checkpoint(fm_path)
    slow = torch.load(slow_path, map_location="cpu", weights_only=True)
    info = read(prepared_info)
    if (info.get("protocol") != "B5-slow-probe-v1" or not info.get("completed") or
            info.get("target_means_removed") is not False or info.get("native_rate_hz") != 50 or
            info.get("slow_width") != 501):
        raise ValueError("Expected the original absolute-voltage S50 prepared-info JSON")
    if slow["config"]["prepared_sha256"] != sha(prepared_info):
        raise ValueError("Slow checkpoint and prepared-info fingerprint disagree")
    if not np.array_equal(np.asarray(fm["scales"]["d12"], np.float32), np.asarray(info["scale_uV"], np.float32)):
        raise ValueError("FM and slow predictor must use identical frozen scales")
    if not candidate and fm_hash != info["checkpoint_sha256"]:
        raise ValueError("Stable FM must be the source bound to the slow experiment")
    variant = {"C": "slow", "F": "slow_f"}.get(slow.get("config", {}).get("group"))
    load_slow(slow, torch.device("cpu"), variant)
    # Exclude optimizers/RNG/targets/prepared paths from the deployable model payload.
    keep = ("architecture_id", "architecture_hash", "model_config", "condition_schema",
            "preprocessing_version", "scales", "scales_sha256", "ema")
    fm_payload = {key: fm[key] for key in keep}
    slow_payload = {"protocol": slow["protocol"], "config": {k: slow["config"][k]
                    for k in ("context_seconds", "group", "model_channels")}, "model": slow["model"]}
    provenance = {"fm_sha256": fm_hash, "slow_sha256": slow_hash,
                  "prepared_info_sha256": sha(prepared_info), "source_fm_epoch": fm.get("epoch"),
                  "source_slow_epoch": slow.get("epoch"), "slow_variant": variant,
                  "source_manifests": fm.get("manifests", {}),
                  "slow_source_checks": info.get("source_checks", {}),
                  "stable_reference": not candidate}
    payload = {"bundle_protocol": BUNDLE_PROTOCOL, "score_protocol": SCORE_PROTOCOL, "slow_beta": beta,
               "sampling": dict(SAMPLING), "fm": fm_payload, "slow": slow_payload,
               "fm_state_sha256": state_sha(fm_payload["ema"]),
               "slow_state_sha256": state_sha(slow_payload["model"]), "provenance": provenance,
               "reference_qc_r": profile["reference_qc_r"] if not candidate else None,
               "reference_tolerance": float(profile.get("reference_tolerance", 2e-5)),
               "target_used_at_inference": False, "prepared_data_required_at_inference": False}
    validate_bundle(payload)
    atomic_save(output, payload)
    write(output.with_suffix(".json"), {k: v for k, v in payload.items() if k not in ("fm", "slow")} |
          {"bundle_sha256": sha(output), "training_started": False, "optimizer_updates": 0})
    return payload


def bundle_slow_beta(bundle):
    protocol = bundle.get("bundle_protocol")
    if protocol == LEGACY_BUNDLE_PROTOCOL:
        beta = validate_slow_beta(bundle.get("slow_beta", 1.0))
        if beta != 1.0:
            raise ValueError("Legacy v1 bundles must retain slow_beta=1")
        return beta
    if protocol != BUNDLE_PROTOCOL or "slow_beta" not in bundle:
        raise ValueError("A v2 bundle/profile must explicitly declare slow_beta")
    return validate_slow_beta(bundle["slow_beta"])


def validate_bundle(bundle):
    if bundle.get("bundle_protocol") not in (BUNDLE_PROTOCOL, LEGACY_BUNDLE_PROTOCOL) or bundle.get("score_protocol") != SCORE_PROTOCOL:
        raise ValueError("Expected a B5-best output-combination bundle")
    bundle_slow_beta(bundle)
    if bundle.get("sampling") != SAMPLING:
        raise ValueError("This bundle requires Heun16/K16/seed42")
    fm = bundle["fm"]
    if (fm.get("architecture_id") != ARCHITECTURE_ID or fm.get("condition_schema") != CONDITION_SCHEMA or
            fm.get("preprocessing_version") != PREPROCESSING_VERSION or
            fm.get("architecture_hash") != ModelConfig.from_dict(fm["model_config"]).fingerprint):
        raise ValueError("FM architecture/metadata/preprocessing mismatch")
    if fm.get("scales_sha256") != scales_digest(fm["scales"]):
        raise ValueError("FM scale digest mismatch")
    scale = np.asarray(fm["scales"]["d12"], np.float32)
    if (scale.shape != (12,) or not np.isfinite(scale).all() or np.any(scale <= 0) or
            not np.array_equal(scale[:1], np.asarray(fm["scales"]["ecg_machine_i"], np.float32))):
        raise ValueError("Invalid d12/anchor scale")
    if state_sha(fm["ema"]) != bundle["fm_state_sha256"] or state_sha(bundle["slow"]["model"]) != bundle["slow_state_sha256"]:
        raise ValueError("Embedded component state digest mismatch")
    if bundle.get("target_used_at_inference") is not False or bundle.get("prepared_data_required_at_inference") is not False:
        raise ValueError("Bundle must be deployable without hidden target/prepared arrays")


def load_bundle(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_bundle(payload)
    return payload


def models(bundle, device):
    validate_bundle(bundle)
    fm = model_from_checkpoint(bundle["fm"], device)
    slow = load_slow(bundle["slow"], device, bundle["provenance"]["slow_variant"])
    return fm, slow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(DEFAULT_PROFILE))
    p.add_argument("--fm-checkpoint")
    p.add_argument("--slow-checkpoint")
    p.add_argument("--prepared-info", help="Only prepared.json is read; no .npz/targets are loaded")
    p.add_argument("--output", required=True)
    p.add_argument("--candidate", action="store_true", help="Explicitly export a different compatible B5-U/S50 pair; no historical score claim")
    args = p.parse_args()
    cfg = read(args.profile)
    overrides = any((args.fm_checkpoint, args.slow_checkpoint, args.prepared_info, args.candidate))
    if "default_bundle" in cfg:
        if overrides:
            p.error("Current default uses a verified portable package. For raw B5-U/S50 sources use "
                    "--profile configs/b5_best_legacy_beta075.json with explicit --candidate overrides")
        export_profile(args.profile, args.output)
    else:
        export_bundle(args.fm_checkpoint or cfg["fm_checkpoint"], args.slow_checkpoint or cfg["slow_checkpoint"],
                      args.prepared_info or cfg["prepared_info"], args.output, cfg, args.candidate)
    print({"bundle": args.output, "training_started": False, "optimizer_updates": 0}, flush=True)


if __name__ == "__main__":
    main()
