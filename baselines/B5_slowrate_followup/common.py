"""Bind follow-ups to selected checkpoints, caches and reference scores."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

from baselines.B5_best.common import read, sha
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.slow import load_slow
from baselines.B5_slowrate import RATES
from baselines.B5_slowrate.common import load_study as load_rates, load_correction as load_rate, within
from baselines.B5_slowrate.common import record_arrays as rate_arrays
from baselines.B5_slowrate.model import RateCorrection
from . import PROTOCOL, GROUPS
from .model import Followup


def code_sha():
    from baselines.B5_slowrate.common import code_sha as rates_sha
    digest = hashlib.sha256(rates_sha().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def source_checkpoints(rate_run, rate_study):
    rate_run = Path(rate_run).resolve()
    result = {}
    for group in RATES:
        path = rate_run / group / "best.pt"
        _, cp = load_rate(path, rate_study, rate_run / "shared/study.json", "cpu")
        verified = read(rate_run / "verification" / group / "best/evaluation.json")
        if (not verified.get("completed") or verified["summary"]["group"] != group or cp["group"] != group
                or verified["summary"]["epoch"] != cp["epoch"]
                or abs(verified["summary"]["mean_r"] - cp["best_r"]) > 1e-7):
            raise ValueError("Source best checkpoint has not been successfully reevaluated")
        result[group] = {"path": str(path), "sha256": sha(path), "epoch": cp["epoch"],
                         "summary": verified["summary"]}
    return result


def load_study(root, verify_records=True):
    root = Path(root).resolve()
    study = read(root / "study.json")
    if (study.get("protocol") != PROTOCOL or not study.get("completed")
            or study.get("code_sha256") != code_sha() or study.get("reference_group") not in RATES):
        raise ValueError("Follow-up study/code changed or is incomplete")
    rate_root = Path(study["rate_shared"])
    if sha(rate_root / "study.json") != study["rate_study_sha256"]:
        raise ValueError("Parent rate study changed")
    parent = load_rates(rate_root, verify_records=verify_records)
    for key in ("base_sha256", "body_sha256", "population", "scale_uV", "sampling", "train", "validation"):
        if study[key] != parent[key]:
            raise ValueError("Follow-up changed pinned base/data: " + key)
    for member in study["sources"].values():
        if sha(member["path"]) != member["sha256"]:
            raise ValueError("A selected source weight changed")
    return study


@contextmanager
def record_arrays(root, study, entry, group=None):
    # No new waveform copies: merge tiny original rate sidecars on demand.
    rate_root = Path(study["rate_shared"])
    with rate_arrays(rate_root, study, entry, "R5") as data:
        with np.load(within(rate_root, entry["file"]), allow_pickle=False) as signals:
            result = {**data, **{"signal_" + g: signals[g] for g in RATES}}
        raw = np.asarray(data["anchor_uV"], np.float32).reshape(1, 6000, 10).mean(-1)
        result["anchor50"] = raw / np.float32(study["scale_uV"][0])
        yield result


class Records(Dataset):
    def __init__(self, root, study):
        self.root, self.study = root, study

    def __len__(self):
        return len(self.study["train"])

    def __getitem__(self, index):
        with record_arrays(self.root, self.study, self.study["train"][index]) as data:
            return {k: torch.from_numpy(np.asarray(v).copy()) for k, v in data.items()
                    if k not in ("anchor_uV", "global_i")}


def initialize(group, study, device):
    sources = {}
    for rate, member in study["sources"].items():
        if sha(member["path"]) != member["sha256"]:
            raise ValueError("Source checkpoint changed")
        cp = torch.load(member["path"], map_location="cpu", weights_only=True)
        sources[rate] = RateCorrection(rate, **study["model"])
        sources[rate].load_state_dict(cp["model"], strict=True)
    original = None
    if group in ("PTAIL", "PCTX", "PCTX_RF"):
        if sha(study["base_checkpoint"]) != study["base_sha256"]:
            raise ValueError("Original FM/P bundle changed")
        bundle = load_bundle(study["base_checkpoint"])
        original = load_slow(bundle["slow"], "cpu", bundle["provenance"]["slow_variant"])
    return Followup(group, sources, study["reference_group"], original, study["tail_blocks"],
                    study["p_context_seconds"], study["p_extra_dilation"]).to(device)


def load_candidate(path, study, study_path, device):
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if (cp.get("protocol") != PROTOCOL or cp.get("group") not in GROUPS
            or cp.get("study_sha256") != sha(study_path) or cp.get("code_sha256") != code_sha()):
        raise ValueError("Candidate belongs to another follow-up study")
    model = initialize(cp["group"], study, device)
    model.load_state_dict(cp["model"], strict=True)
    return model, cp
