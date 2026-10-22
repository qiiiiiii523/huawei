"""Study fingerprints, immutable record caches and correction-only checkpoints."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from baselines.B5_best.common import read, sha
from baselines.B5_final.profile import BASE_SHA, BODY_SHA, SAMPLING
from . import PROTOCOL
from .model import ResidualSlowCNN


def code_sha():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_study(run, verify_records=True):
    run = Path(run).resolve()
    study = read(run / "study.json")
    if (study.get("protocol") != PROTOCOL or study.get("completed") is not True
            or study.get("base_sha256") != BASE_SHA or study.get("body_sha256") != BODY_SHA
            or study.get("sampling") != SAMPLING or study.get("code_sha256") != code_sha()):
        raise ValueError("Study incomplete, selected assets/code changed, or protocol mismatch")
    if verify_records:
        for split in ("train", "validation"):
            for entry in study[split]:
                path = run / entry["file"]
                if not path.resolve().is_relative_to(run) or sha(path) != entry["sha256"]:
                    raise ValueError("Record cache changed: " + str(path))
    return study


class Records(Dataset):
    def __init__(self, run, entries):
        self.run, self.entries = Path(run), entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        with np.load(self.run / self.entries[index]["file"], allow_pickle=False) as data:
            return {k: torch.from_numpy(np.asarray(data[k]).copy()) for k in data.files if k != "anchor_uV"}


def load_correction(path, study, study_path, device):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (value.get("protocol") != PROTOCOL or value.get("study_sha256") != sha(study_path)
            or value.get("base_sha256") != BASE_SHA or value.get("body_sha256") != BODY_SHA):
        raise ValueError("Correction belongs to a different study/base")
    model = ResidualSlowCNN(value["group"], **study["model"]).to(device)
    model.load_state_dict(value["model"], strict=True)
    return model, value
