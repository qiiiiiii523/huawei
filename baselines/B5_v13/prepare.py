"""Verify the fixed pair/data and run a CPU zero-init forward; never train."""
import argparse
from pathlib import Path

import torch

from baselines.B5.checkpoint import checkpoint_preprocessor
from baselines.B5.config import ModelConfig
from baselines.B5.data import preprocessing_config
from baselines.B5.metadata import encode_demographics
from baselines.B5.objective import loss_signature
from baselines.B5.runtime import build_datasets, seed_all
from baselines.B5_best import SAMPLING, SCORE_PROTOCOL
from baselines.B5_best.bundle import DEFAULT_PROFILE, export_bundle
from baselines.B5_best.common import read, write, sha
from baselines.B5_best.validate import verify_reference_inputs
from . import PROTOCOL
from .common import code_digest
from .model import initialized_model


def zero_check(bundle):
    seed_all(42, True)
    control = initialized_model(bundle, "control", torch.device("cpu")).eval()
    experiment = initialized_model(bundle, "v13", torch.device("cpu")).eval()
    fields = {k: torch.from_numpy(v[None]) for k, v in encode_demographics({"age": 40}).items()}
    anchor, state = torch.randn(1, 1, 5000), torch.randn(1, 11, 5000)
    differences = []
    with torch.no_grad():
        for value in (0., .5, 1.):
            time = torch.tensor([value])
            a, ai = control(state, time, anchor, **fields)
            b, bi = experiment(state, time, anchor, **fields)
            differences.append(float((a - b).abs().max()))
            if not torch.equal(ai, bi):
                raise ValueError("I head changed during initialization")
    if max(differences) != 0.:
        raise ValueError("Zero residual did not reproduce original FM velocity")
    return {"cpu_forward_shape": [1, 11, 5000], "tested_times": [0., .5, 1.],
            "velocity_max_abs_difference": max(differences),
            "control_parameters": sum(p.numel() for p in control.parameters()),
            "experiment_parameters": sum(p.numel() for p in experiment.parameters()),
            "new_branch_parameters": sum(p.numel() for p in experiment.local_v13.parameters()),
            "affected_missing_velocity_indices": [5, 7], "affected_leads": ["V1", "V3"]}


def prepare(root, output, epochs=20, learning_rate=None):
    output, root = Path(output).resolve(), Path(root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty shared output; do not tee logs inside it")
    if epochs < 1 or (learning_rate is not None and learning_rate <= 0):
        raise ValueError("Need positive epochs/learning rate")
    output.mkdir(parents=True, exist_ok=True)
    profile = read(DEFAULT_PROFILE)
    bundle = export_bundle(profile["fm_checkpoint"], profile["slow_checkpoint"],
                           profile["prepared_info"], output / "components.pt", profile)
    info = read(profile["prepared_info"])
    config = read(Path(profile["fm_checkpoint"]).parent / "resolved_config.json")
    config["repository_root"] = str(root)
    if ModelConfig.from_dict(config["model"]).fingerprint != bundle["fm"]["architecture_hash"]:
        raise ValueError("Source data config and FM architecture disagree")
    loss_signature(config["loss"])
    checks = info.get("source_checks", {})
    required = {"strict_index", "demographics_csv", "subject_split_csv", "device_interpretation_qc_csv",
                "task1_train_target", "task2_train_target", "task1_validation_target", "task2_validation_target"}
    if not required.issubset(checks):
        raise ValueError("Original prepared-info lacks required training/data fingerprints")
    for key, item in checks.items():
        print("Checking original data: " + key, flush=True)
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Original data changed: " + key)
    verify_reference_inputs(bundle, config)
    preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(config))
    train, validation = build_datasets(config, preprocessor)
    manifests = {"train": train.manifest_digest, **{k: v.manifest_digest for k, v in validation.items()}}
    if any(bundle["provenance"]["source_manifests"].get(k) != v for k, v in manifests.items()):
        raise ValueError("Train/validation population differs from the original FM run")
    train_subjects = {r["subject_id"] for r in train.rows}
    val_subjects = {r["subject_id"] for ds in validation.values() for r in ds.rows}
    if train_subjects & val_subjects:
        raise ValueError("Subject leakage between training and validation")
    training = dict(config["training"], epochs=epochs, seed=42, deterministic=True, validate_every=5)
    if learning_rate is not None:
        training["learning_rate"] = learning_rate
    config["training"], config["sampling"] = training, dict(SAMPLING)
    config["validation"]["selection_metric"] = "main_correlation"
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        checked = zero_check(bundle)
    finally:
        torch.set_num_threads(old_threads)
    study = {"protocol": PROTOCOL, "score_protocol": SCORE_PROTOCOL, "sampling": dict(SAMPLING),
             "beta": 1.0, "bundle_sha256": sha(output / "components.pt"),
             "source_checks": checks, "manifests": manifests, "data_config": config,
             "code_sha256": code_digest(), "source_provenance": bundle["provenance"],
             "train_windows": len(train), "train_records": len({r["target_record_id"] for r in train.rows}),
             "validation_unique_targets": len({r["target_record_id"] for ds in validation.values() for r in ds.rows}),
             "train_validation_subject_overlap": 0}
    write(output / "study.json", study)
    write(output / "preflight.json", {"ready": True, "training_started": False, "optimizer_updates": 0,
          "same_base_weights": True, "slow_predictor_frozen": True, "beta": 1.0,
          "training": training, "sampling": dict(SAMPLING), **checked})
    print({"ready": True, "study": str(output / "study.json"), **checked}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", default="/home/qht/huawei")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, help="Default: preserve the source FM run learning rate")
    args = p.parse_args()
    prepare(args.repository_root, args.output, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
