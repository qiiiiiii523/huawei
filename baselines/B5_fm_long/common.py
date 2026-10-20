"""Frozen study identities, strict cache contracts and separate experiment checkpoints."""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch

from baselines.B5_best.common import read, sha, state_sha
from baselines.B5_final.profile import BASE_SHA, SAMPLING
from . import PROTOCOL, FIELDS, WINDOW_SECONDS

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/b5_fm_long.json"


def code_sha():
    root = Path(__file__).resolve().parents[2]
    paths = []
    for folder in ("B5_fm_long", "B5", "B5_best", "B5_final"):
        paths.extend((root / "baselines" / folder).glob("*.py"))
    paths.extend((root / "ecg12gen").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load_experiment(path=DEFAULT_CONFIG):
    cfg = read(path)
    return validate_experiment(cfg)


def validate_experiment(cfg):
    if (cfg.get("protocol") != PROTOCOL or cfg.get("window_seconds") != [10, 40]
            or cfg.get("sampling") != SAMPLING or cfg.get("slow_beta") != .75
            or cfg.get("context") != "disabled"):
        raise ValueError("This experiment fixes 10s/40s, Heun16/K16, beta .75 and no device context")
    training = cfg["training"]
    for key in ("updates", "validate_every", "records_per_update"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError("Positive integer required: " + key)
    if type(training["seed"]) is not int or not 0 <= training["seed"] < 2**32:
        raise ValueError("Training seed must be uint32")
    for key in ("learning_rate", "gradient_clip"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError("Positive finite value required: " + key)
    if not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("Invalid weight_decay")
    if not 0 <= training["ema_decay"] < 1:
        raise ValueError("Invalid EMA decay")
    return cfg


def metadata(data):
    return {key: np.asarray(data[key]) for key in FIELDS}


def visible_digest(anchor, meta):
    digest = hashlib.sha256(np.ascontiguousarray(anchor).tobytes())
    for key in FIELDS:
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(meta[key]).tobytes())
    return digest.hexdigest()


def validate_record(data, training=False):
    if (data["anchor_uV"].shape != (1, 60000) or data["target_uV"].shape != (12, 60000)
            or data["slow_uV"].shape != (11, 60000) or data["mask"].shape != (12,)
            or data["mask"].dtype != np.bool_):
        raise ValueError("Record must contain actual full120s waveforms and boolean lead QC")
    for key in ("anchor_uV", "target_uV", "slow_uV"):
        if not np.isfinite(data[key]).all():
            raise ValueError("Nonfinite record: " + key)
    if training and (not data["mask"][0] or not data["mask"][1:].any()):
        raise ValueError("Training requires reliable I and at least one valid missing lead")
    expected = {"numeric": (3,), "sex": (), "field_mask": (4,), "age_topcoded": (1,)}
    for key, shape in expected.items():
        if np.asarray(data[key]).shape != shape:
            raise ValueError("Invalid demographic field: " + key)


def record_path(root, entry):
    root = Path(root).resolve()
    path = (root / entry["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Cache path escapes study")
    return path


def read_record(root, entry):
    with np.load(record_path(root, entry), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_study(run, deep=True):
    root = Path(run).resolve()
    cfg = read(root / "study.json")
    if (cfg.get("protocol") != PROTOCOL or cfg.get("completed") is not True
            or cfg.get("base_sha256") != BASE_SHA or cfg.get("code_sha256") != code_sha()
            or cfg.get("sampling") != SAMPLING or cfg.get("context") != "disabled"):
        raise ValueError("Study is incomplete, modified, or from another protocol/code version")
    if not cfg["train"] or set(cfg["validation"]) != {"task1", "task2"}:
        raise ValueError("Both tasks and training partition are required")
    validate_experiment(cfg["experiment"])
    train_subjects = {e["subject_id"] for e in cfg["train"]}
    val_subjects = {e["subject_id"] for values in cfg["validation"].values() for e in values}
    if train_subjects & val_subjects:
        raise ValueError("Training and validation patients overlap")
    for task, entries in cfg["validation"].items():
        rows = cfg["validation_rows"][task]
        assigned = set()
        for entry in entries:
            ids = entry["window_indices"]
            if (len(ids) != 12 or assigned.intersection(ids)
                    or any(type(i) is not int or i < 0 or i >= len(rows) for i in ids)):
                raise ValueError("Duplicate/incomplete validation window mapping")
            selected = [rows[i] for i in ids]
            if ([int(r["start_sample_500hz"]) for r in selected] != list(range(0, 60000, 5000))
                    or any(r["target_record_id"] != entry["target_record_id"]
                           or r["subject_id"] != entry["subject_id"]
                           or r["pair_id"] != entry["pair_id"]
                           or r["input_type"] != entry["input_type"] for r in selected)):
                raise ValueError("Validation physical record/patient/pair identity changed")
            assigned.update(ids)
        if assigned != set(range(len(rows))):
            raise ValueError("Validation mapping must retain all original rows")
    if sha(cfg["base_checkpoint"]) != BASE_SHA:
        raise ValueError("Pinned source bundle changed")
    from baselines.B5_best.bundle import load_bundle
    from baselines.B5.config import ModelConfig
    bundle = load_bundle(cfg["base_checkpoint"])
    if (ModelConfig.from_dict(cfg["model_config"]).fingerprint != bundle["fm"]["architecture_hash"]
            or cfg["scale_uV"] != bundle["fm"]["scales"]["d12"]
            or cfg["initial_fm_state_sha256"] != bundle["fm_state_sha256"]
            or cfg["slow_state_sha256"] != bundle["slow_state_sha256"]):
        raise ValueError("Frozen scales/model/source states disagree with Final bundle")
    if deep:
        checked = set()
        for entry in cfg["train"] + [e for entries in cfg["validation"].values() for e in entries]:
            path = record_path(root, entry)
            identity = (str(path), entry["sha256"])
            if identity not in checked:
                if sha(path) != entry["sha256"]:
                    raise ValueError("Record cache changed: " + str(path))
                validate_record(read_record(root, entry), entry in cfg["train"])
                checked.add(identity)
    return cfg


def candidate_payload_matches(value, study, study_path):
    if (value.get("protocol") != PROTOCOL or value.get("window_seconds") not in WINDOW_SECONDS
            or value.get("base_sha256") != BASE_SHA or value.get("code_sha256") != code_sha()
            or value.get("study_sha256") != sha(study_path)
            or value.get("experiment") != study["experiment"]
            or value.get("model_config") != study["model_config"]
            or value.get("scale_uV") != study["scale_uV"]
            or value.get("fm_loss") != study["fm_loss"]
            or value.get("state_sha256") != state_sha(value["ema"])):
        raise ValueError("Candidate does not match this frozen FM40 study")


def load_candidate(path, study, study_path, device):
    from baselines.B5.model import B5UNet
    from baselines.B5.config import ModelConfig
    value = torch.load(path, map_location="cpu", weights_only=True)
    candidate_payload_matches(value, study, study_path)
    model = B5UNet(ModelConfig.from_dict(value["model_config"]))
    model.load_state_dict(value["ema"], strict=True)
    return model.to(device).eval(), value
