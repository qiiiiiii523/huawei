"""Strict shared inputs, checkpoint identities and reproducible pair settings."""
from pathlib import Path

import torch

from baselines.B5.checkpoint import checkpoint_preprocessor
from baselines.B5.data import preprocessing_config
from baselines.B5.objective import loss_signature
from baselines.B5.runtime import build_datasets
from baselines.B5_best import SAMPLING, SCORE_PROTOCOL
from baselines.B5_best.bundle import load_bundle, bundle_slow_beta
from baselines.B5_best.common import read, sha
from baselines.B5_best.validate import verify_reference_inputs
from . import PROTOCOL, ARCHITECTURE_ID
from .model import initialized_model, KERNELS


def code_digest():
    import hashlib
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob("*.py"))
    for folder in ("baselines/B5", "baselines/B5_best", "ecg12gen"):
        paths += list((root / folder).glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_study(path, verify_data=True):
    path = Path(path).resolve()
    study = read(path)
    if study.get("protocol") != PROTOCOL or study.get("sampling") != SAMPLING:
        raise ValueError("Wrong paired experiment or sampling protocol")
    if study.get("score_protocol") != SCORE_PROTOCOL or study.get("beta") != .75:
        raise ValueError("This paired experiment fixes beta=0.75; rerun prepare in a NEW output for an old beta=1 study")
    if study.get("code_sha256") != code_digest():
        raise ValueError("Code changed after preflight; use the same commit or prepare a new study")
    if sha(path.parent / "components.pt") != study["bundle_sha256"]:
        raise ValueError("Shared component bundle changed")
    bundle = load_bundle(path.parent / "components.pt")
    if bundle_slow_beta(bundle) != study["beta"]:
        raise ValueError("Study and component bundle must use the same beta=0.75")
    config = study["data_config"]
    loss_signature(config["loss"])
    if verify_data:
        for key, item in study["source_checks"].items():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Original data changed: " + key)
        verify_reference_inputs(bundle, config)
    preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(config))
    return study, bundle, preprocessor


def datasets_for(study, bundle, preprocessor):
    train, validation = build_datasets(study["data_config"], preprocessor)
    manifests = {"train": train.manifest_digest,
                 **{k: v.manifest_digest for k, v in validation.items()}}
    if manifests != study["manifests"]:
        raise ValueError("Current train/validation selection differs from the paired study")
    source = bundle["provenance"]["source_manifests"]
    if any(source.get(k) != v for k, v in manifests.items()):
        raise ValueError("Population differs from the original FM run")
    return train, validation


def load_experiment_checkpoint(path, study_path, device, use_ema=True, verify_data=True):
    study, bundle, preprocessor = load_study(study_path, verify_data=verify_data)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("protocol") != PROTOCOL or
            checkpoint.get("study_sha256") != sha(study_path) or
            checkpoint.get("sampling_config") != study["sampling"] or
            checkpoint.get("score_protocol") != SCORE_PROTOCOL):
        raise ValueError("Checkpoint belongs to a different study/protocol")
    group = checkpoint["group"]
    expected = ARCHITECTURE_ID if group == "v13" else "B5-U-v1"
    if checkpoint.get("architecture_id") != expected or checkpoint.get("kernels") != list(KERNELS):
        raise ValueError("Experimental architecture mismatch")
    model = initialized_model(bundle, group, device)
    model.load_state_dict(checkpoint["ema"] if use_ema else checkpoint["model"], strict=True)
    return checkpoint, study, bundle, preprocessor, model.eval()
