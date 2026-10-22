"""Read-only parent caches, small rate-input sidecars and bound checkpoints."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

from baselines.B5_best.common import read, sha
from baselines.B5_final.profile import BASE_SHA, BODY_SHA, SAMPLING
from baselines.B5_slow120.common import load_study as load_parent
from . import PROTOCOL, RATES
from .model import RateCorrection


def code_sha():
    digest = hashlib.sha256()
    for folder in (Path(__file__).parent.parent / "B5_slow120", Path(__file__).parent):
        for path in sorted(folder.glob("*.py")):
            digest.update((folder.name + "/" + path.name).encode())
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def within(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Cache path escapes its root")
    return path


def load_study(root, verify_records=True):
    root = Path(root).resolve()
    study = read(root / "study.json")
    if (study.get("protocol") != PROTOCOL or study.get("completed") is not True
            or study.get("code_sha256") != code_sha() or study.get("rates_hz") != RATES
            or study.get("base_sha256") != BASE_SHA or study.get("body_sha256") != BODY_SHA
            or study.get("sampling") != SAMPLING):
        raise ValueError("Rate study incomplete or code/assets/protocol changed")
    source = Path(study["source_shared"])
    if sha(source / "study.json") != study["source_study_sha256"]:
        raise ValueError("Parent manifest changed")
    parent = load_parent(source, verify_records=verify_records)
    for split in ("train", "validation"):
        expected = [(e["file"], e["sha256"]) for e in parent[split]]
        actual = [(e["source_file"], e["source_sha256"]) for e in study[split]]
        if expected != actual:
            raise ValueError("Parent record selection changed")
        if verify_records:
            for entry in study[split]:
                if sha(within(root, entry["file"])) != entry["sha256"]:
                    raise ValueError("Rate sidecar changed")
    return study


@contextmanager
def record_arrays(root, study, entry, group):
    with np.load(within(study["source_shared"], entry["source_file"]), allow_pickle=False) as raw:
        with np.load(within(root, entry["file"]), allow_pickle=False) as rate:
            yield {**{k: raw[k] for k in raw.files if k != "local_i"}, "global_i": rate[group]}


class Records(Dataset):
    def __init__(self, root, study, group):
        self.root, self.study, self.group = Path(root), study, group

    def __len__(self):
        return len(self.study["train"])

    def __getitem__(self, index):
        with record_arrays(self.root, self.study, self.study["train"][index], self.group) as data:
            return {k: torch.from_numpy(np.asarray(v).copy()) for k, v in data.items() if k != "anchor_uV"}


def load_correction(path, study, study_path, device):
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if (cp.get("protocol") != PROTOCOL or cp.get("study_sha256") != sha(study_path)
            or cp.get("group") not in RATES or cp.get("code_sha256") != code_sha()
            or cp.get("base_sha256") != BASE_SHA or cp.get("body_sha256") != BODY_SHA
            or cp.get("sampling_hz") != RATES[cp["group"]]):
        raise ValueError("Checkpoint belongs to another rate study or code")
    model = RateCorrection(cp["group"], **study["model"]).to(device)
    model.load_state_dict(cp["model"], strict=True)
    return model, cp
