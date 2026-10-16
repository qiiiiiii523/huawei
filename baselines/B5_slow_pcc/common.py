"""Immutable paired settings and one fixed FM cache, with exact-resume identity."""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from baselines.B5_best import SAMPLING, SCORE_PROTOCOL
from baselines.B5_best.bundle import load_bundle, bundle_slow_beta
from baselines.B5_best.common import read, sha, write
from baselines.B5_best.slow import load_slow
from . import PROTOCOL, CACHE_PROTOCOL, GROUPS


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, value)
    os.replace(temporary, path)


def validation_key(task, pair_id):
    """Pair-specific historical FM output, with physical target identity kept separately."""
    return str(task) + ":" + str(pair_id)


def code_digest():
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob("*.py"))
    for folder in ("baselines/B5", "baselines/B5_best", "ecg12gen"):
        paths += list((root / folder).glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_study(path, verify_data=False):
    path = Path(path).resolve()
    study = read(path)
    if (study.get("protocol") != PROTOCOL or study.get("sampling") != SAMPLING or
            study.get("score_protocol") != SCORE_PROTOCOL):
        raise ValueError("Wrong SlowPCC paired protocol/sampling")
    if study.get("code_sha256") != code_digest():
        raise ValueError("Code changed after preflight; use the same implementation or a new study")
    if sha(path.parent / "components.pt") != study["bundle_sha256"]:
        raise ValueError("Fixed component bundle changed")
    bundle = load_bundle(path.parent / "components.pt")
    if study.get("beta") != bundle_slow_beta(bundle):
        raise ValueError("Both arms must preserve the shared B5-best slow coefficient")
    if bundle["provenance"]["slow_variant"] != "slow_f":
        raise ValueError("This experiment starts from S50-F")
    if verify_data:
        for key, item in study["source_checks"].items():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Original data changed: " + key)
    return study, bundle


def slow_model(bundle, device, trainable=True):
    model = load_slow(bundle["slow"], device, "slow_f")
    return model.requires_grad_(trainable)


def load_cache(path, study_path):
    path = Path(path).resolve()
    if not (path / "cache.json").is_file():
        raise FileNotFoundError("Fixed FM cache is incomplete: cache.json is missing. Finish/resume the cache stage before training.")
    manifest = read(path / "cache.json")
    if (manifest.get("protocol") != CACHE_PROTOCOL or not manifest.get("completed") or
            manifest.get("study_sha256") != sha(study_path)):
        raise ValueError("Cache is incomplete or belongs to another study")
    for split in ("train", "validation"):
        for item in manifest[split]:
            if sha(path / item["file"]) != item["sha256"]:
                raise ValueError("Fixed FM cache changed: " + item["file"])
    keys = [item["cache_key"] for item in manifest["validation"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate validation pair cache key")
    lookup = {item["cache_key"]: item for item in manifest["validation"]}
    for task, rows in manifest["validation_rows"].items():
        for row in rows:
            item = lookup.get(validation_key(task, row["pair_id"]))
            if item is None or item["target_record_id"] != row["target_record_id"]:
                raise ValueError("Validation pair/target identity differs from the cache")
    return manifest


class Records(Dataset):
    def __init__(self, root, entries):
        self.root, self.entries = Path(root), entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        with np.load(self.root / self.entries[i]["file"], allow_pickle=False) as data:
            return {key: torch.from_numpy(np.asarray(data[key]).copy()) for key in data.files}


def load_checkpoint(path, study_path, device):
    study, bundle = load_study(study_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("protocol") != PROTOCOL or checkpoint.get("study_sha256") != sha(study_path) or
            checkpoint.get("group") not in GROUPS or checkpoint.get("sampling_config") != SAMPLING):
        raise ValueError("Checkpoint belongs to another experiment/study")
    model = slow_model(bundle, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, study, bundle, model
