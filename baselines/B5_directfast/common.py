"""Versioned cache/checkpoint loading, explicit inputs and source fingerprints."""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch

from baselines.B5_best.common import read, sha
from baselines.B5_final.profile import BASE_SHA, BODY_SHA, SAMPLING
from . import PROTOCOL, GROUPS
from .model import build_model

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/b5_directfast.json"


def code_sha():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = []
    for folder in ("B5_directfast", "B5", "B5_best", "B5_best_context", "B5_final"):
        paths.extend((root / "baselines" / folder).glob("*.py"))
    paths.extend((root / "ecg12gen").glob("*.py"))
    for name in ("b5_final.json", "b5_best_context.json"):
        paths.append(root / "configs" / name)
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_experiment(path=DEFAULT_CONFIG):
    value = read(path)
    if (value.get("protocol") != PROTOCOL or value.get("slow_beta") != .75
            or value.get("filter_width") != 501
            or value.get("initialization") != "selected_final_fm_ema"
            or value.get("body_transfer") != "explicit_frozen_body_transfer"):
        raise ValueError("Unexpected directfast experiment settings")
    cfg = value["training"]
    for key in ("epochs", "gradient_accumulation", "validate_every"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError("Positive integer required: " + key)
    if not isinstance(cfg["seed"], int) or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("Need uint32 training seed")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("Positive finite training value required: " + key)
    if not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    if cfg["selection_route"] not in ("baseline", "both"):
        raise ValueError("Selection route must be predeclared baseline or both")
    loss = value["direct_loss"]
    if (not math.isfinite(loss["huber_delta"]) or loss["huber_delta"] <= 0
            or not math.isfinite(loss["combo_pcc"]) or loss["combo_pcc"] < 0):
        raise ValueError("Invalid direct loss")
    return value


def load_study(run, verify_records=True):
    root = Path(run).resolve()
    value = read(root / "study.json")
    if (value.get("protocol") != PROTOCOL or value.get("completed") is not True
            or value.get("base_sha256") != BASE_SHA or value.get("body_sha256") != BODY_SHA
            or value.get("sampling") != SAMPLING or value.get("code_sha256") != code_sha()):
        raise ValueError("Study incomplete or source code/weights/protocol changed")
    if not value["train"] or not value["validation"]:
        raise ValueError("Both patient partitions are required")
    if verify_records:
        for entry in value["train"] + value["validation"]:
            path = (root / entry["file"]).resolve()
            if not path.is_relative_to(root) or sha(path) != entry["sha256"]:
                raise ValueError("Record cache changed: " + str(path))
    return value


def load_candidate(path, device, study=None, study_path=None):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (value.get("protocol") != PROTOCOL or value.get("group") not in GROUPS
            or value.get("base_sha256") != BASE_SHA or value.get("body_sha256") != BODY_SHA
            or value.get("code_sha256") != code_sha()):
        raise ValueError("Wrong candidate protocol, source lineage or code")
    if study is not None and (value["study_sha256"] != sha(study_path)
                              or value["model_config"] != study["model_config"]
                              or value["scale_uV"] != study["scale_uV"]):
        raise ValueError("Candidate belongs to another prepared study")
    model = build_model(value["group"], value["model_config"], value["model"], device)
    return model, value


def array_fields(data):
    from . import FIELDS
    # Only explicit observed fields enter a network; labels remain in the caller.
    return {k: np.asarray(data[k]) for k in FIELDS}
