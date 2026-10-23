"""Bind all arms to identical P/G120 assets, immutable records and runtime code."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_slow120.common import load_study as load_source
from baselines.B5_final.profile import BASE_SHA, BODY_SHA, SAMPLING
from . import PROTOCOL
from .model import JointSlow


def code_sha():
    digest = hashlib.sha256()
    # Include imported runtime implementations, not just the new wrapper.
    root = Path(__file__).resolve().parents[1]
    files = list(Path(__file__).parent.glob("*.py"))
    for package in ("B5", "B5_slow120", "B5_best", "B5_best_context", "B5_final"):
        files.extend((root / package).glob("*.py"))
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def safe_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes experiment directory")
    return path


def load_study(root, verify_records=True):
    root = Path(root).resolve()
    d = read(root / "study.json")
    if (d.get("protocol") != PROTOCOL or d.get("completed") is not True
            or d.get("code_sha256") != code_sha() or d.get("sampling") != SAMPLING
            or d.get("base_sha256") != BASE_SHA or d.get("body_sha256") != BODY_SHA):
        raise ValueError("Incomplete/changed joint study, code or selected assets")
    if sha(Path(d["source_run"]) / "study.json") != d["source_study_sha256"]:
        raise ValueError("Original G120 study changed")
    original = load_source(d["source_run"], verify_records=verify_records)
    if d["train"] != original["train"] or d["validation"] != original["validation"]:
        raise ValueError("Joint training must keep the original Huawei data and QC")
    if sha(root / "initializer.pt") != d["initializer_sha256"]:
        raise ValueError("Joint initializer changed")
    if sha(d["source_g120_checkpoint"]) != d["source_g120_sha256"]:
        raise ValueError("Source G120 changed")
    return d


def initial_model(root, study, device):
    cp = torch.load(Path(root) / "initializer.pt", map_location="cpu", weights_only=True)
    if cp.get("protocol") != PROTOCOL or state_sha(cp["model"]) != study["initial_weights_sha256"]:
        raise ValueError("Initializer protocol/state mismatch")
    model = JointSlow(study["model"], study["training"]["window_batch"])
    model.load_state_dict(cp["model"], strict=True)
    return model.to(device)


def load_model(path, study, study_path, device):
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if (cp.get("protocol") != PROTOCOL or cp.get("group") not in ("A", "B", "C")
            or cp.get("study_sha256") != sha(study_path)
            or cp.get("source_g120_sha256") != study["source_g120_sha256"]):
        raise ValueError("Checkpoint belongs to another joint study")
    model = JointSlow(study["model"], study["training"]["window_batch"])
    model.load_state_dict(cp["model"], strict=True)
    model.set_group(cp["group"])
    return model.to(device), cp


class Records(Dataset):
    def __init__(self, root, entries, scale_i):
        self.root, self.entries, self.scale_i = Path(root), entries, scale_i

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        with np.load(safe_file(self.root, self.entries[index]["file"]), allow_pickle=False) as z:
            data = {k: np.asarray(z[k]).copy() for k in z.files if k != "anchor_uV"}
            if "anchor" not in data:
                data["anchor"] = z["anchor_uV"].reshape(1, 6000, 10).mean(-1).astype(np.float32) / self.scale_i
        return {k: torch.from_numpy(v) for k, v in data.items()}
