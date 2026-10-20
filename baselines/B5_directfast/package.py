"""Export an inference-only package with candidate, frozen P/I and frozen body."""
import argparse
from pathlib import Path
import torch
from baselines.B5.checkpoint import atomic_save
from baselines.B5_best.bundle import load_bundle, validate_bundle
from baselines.B5_best.common import write, sha, state_sha
from baselines.B5_best_context.adapters import load_settings
from baselines.B5_final.profile import BASE_SHA, BODY_SHA, check_payloads
from . import PROTOCOL, PACKAGE_PROTOCOL
from .common import load_study, load_candidate, code_sha


def load_package(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (value.get("package_protocol") != PACKAGE_PROTOCOL or value.get("code_sha256") != code_sha()
            or value.get("base_sha256") != BASE_SHA or value.get("body_sha256") != BODY_SHA
            or value.get("target_used_at_inference") is not False
            or value.get("prepared_data_required_at_inference") is not False):
        raise ValueError("Wrong inference package or source code")
    validate_bundle(value["original_bundle"])
    check_payloads(value["original_bundle"], value["body"])
    if (state_sha(value["body"]["model"]) != value["body_state_sha256"]
            or state_sha(value["candidate"]["model"]) != value["fast_state_sha256"]
            or value["candidate"].get("protocol") != PROTOCOL
            or value["candidate"]["scale_uV"] != value["original_bundle"]["fm"]["scales"]["d12"]
            or value["candidate"]["model_config"] != value["original_bundle"]["fm"]["model_config"]):
        raise ValueError("Embedded model state or scale mismatch")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, help="New .pt file, never an original checkpoint")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Use a new package filename")
    study = load_study(args.run, verify_records=False)
    _, candidate = load_candidate(args.candidate, torch.device("cpu"), study, Path(args.run) / "study.json")
    for key, digest in (("base_checkpoint", BASE_SHA), ("body_checkpoint", BODY_SHA)):
        if sha(study[key]) != digest:
            raise ValueError("Original source asset changed")
    bundle = load_bundle(study["base_checkpoint"])
    original_body = torch.load(study["body_checkpoint"], map_location="cpu", weights_only=True)
    check_payloads(bundle, original_body)
    body_keys = ("protocol", "group", "model", "sampling", "slow_beta", "training_seed", "epoch",
                 "base_checkpoint_sha256", "cache_sha256", "body_adaptation", "feature_encoding",
                 "history_preprocessing", "history_policy", "body_feature_encoding", "record_gate")
    # Feature/history metadata names are preserved exactly, while optimizer/RNG
    # state and training references never enter the portable inference payload.
    body = {k: v for k, v in original_body.items() if k in body_keys}
    from baselines.B5_best_context.body_features import feature_encoding
    from baselines.B5_best_context.body_history import history_policy
    body["feature_encoding"] = feature_encoding(original_body)
    # Copy the exact original keys used by policy readers (see body_history).
    for key in ("body_history_preprocessing", "history_variant", "history"):
        if key in original_body:
            body[key] = original_body[key]
    if history_policy(body) != history_policy(original_body):
        raise ValueError("Portable body metadata lost its saved history policy")
    check_payloads(bundle, body)
    keep = ("protocol", "group", "model_config", "scale_uV", "model", "epoch", "updates", "selection_route")
    payload = {"package_protocol": PACKAGE_PROTOCOL, "code_sha256": code_sha(),
        "base_sha256": BASE_SHA, "body_sha256": BODY_SHA, "candidate_sha256": sha(args.candidate),
        "candidate": {k: candidate[k] for k in keep}, "original_bundle": bundle, "body": body,
        "context_settings": load_settings(), "body_state_sha256": state_sha(body["model"]),
        "fast_state_sha256": state_sha(candidate["model"]), "body_transfer": "frozen_body_transfer",
        "target_used_at_inference": False, "prepared_data_required_at_inference": False,
        "selection_validation": candidate["validation"]}
    atomic_save(output, payload)
    write(output.with_suffix(".json"), {k: payload[k] for k in
        ("package_protocol", "code_sha256", "base_sha256", "body_sha256", "candidate_sha256",
         "body_state_sha256", "fast_state_sha256", "body_transfer", "target_used_at_inference",
         "prepared_data_required_at_inference")} | {"package_sha256": sha(output), "group": candidate["group"],
                                                   "epoch": candidate["epoch"], "training_started": False})


if __name__ == "__main__":
    main()
