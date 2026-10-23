"""Calibrated raw-uV record caches, record-uniform sampling and masked history."""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from baselines.B5_best.common import read, sha

FIELDS = ("numeric", "sex", "field_mask", "age_topcoded")
CONTEXT = ("context", "context_time_mask", "context_window_mask", "context_window_weights")


def encode_history(raw, mask, scale):
    raw, mask = np.asarray(raw, np.float32), np.asarray(mask, bool)
    if raw.ndim != 2 or raw.shape[0] != 6 or mask.shape != raw.shape:
        raise ValueError("History requires raw-uV [6,T] and matching mask")
    mask = mask & np.isfinite(raw)
    n = max(1, (raw.shape[-1] + 4999) // 5000)
    value = np.zeros((6, n * 5000), np.float32)
    valid = np.zeros_like(value, bool)
    for lead in range(6):
        selected = raw[lead, mask[lead]]
        center = np.median(selected) if len(selected) else 0.
        value[lead, :raw.shape[-1]] = np.where(mask[lead], (raw[lead] - center) / scale[lead], 0.)
        valid[lead, :raw.shape[-1]] = mask[lead]
    value = value.reshape(6, n, 5000).transpose(1, 0, 2)
    valid = valid.reshape(6, n, 5000).transpose(1, 0, 2)
    counts = valid.sum((1, 2)).astype(np.float32)
    return dict(context=value.copy(), context_time_mask=valid.copy(), context_window_mask=counts > 0,
                context_window_weights=counts / max(1., counts.sum()))


def tensor_batch(records, starts, scale, device, context=False, history_override=None):
    # Metadata is allowed; labels are deliberately not part of model_inputs().
    values = {k: np.stack([r[k] for r in records]) for k in FIELDS}
    values["anchor"] = np.stack([r["anchor_uV"][:, s:s + 5000] / scale[0] for r, s in zip(records, starts)])
    values["target"] = np.stack([r["target_uV"][:, s:s + 5000] / scale[:, None] for r, s in zip(records, starts)])
    values["quality_mask"] = np.stack([r["mask"] for r in records])
    if context:
        encoded = [encode_history((history_override or records)[i]["history_uV"],
                   (history_override or records)[i]["history_mask"], scale) for i in range(len(records))]
        n = max(len(v["context"]) for v in encoded)
        for k in CONTEXT:
            shape = (len(records), n, 6, 5000) if k in CONTEXT[:2] else (len(records), n)
            values[k] = np.zeros(shape, bool if k in ("context_time_mask", "context_window_mask") else np.float32)
            for i, v in enumerate(encoded): values[k][i, :len(v[k])] = v[k]
    return {k: torch.from_numpy(np.asarray(v).copy()).to(device) for k, v in values.items()}


def model_inputs(batch):
    return {k: batch[k] for k in ("anchor", *FIELDS, *CONTEXT) if k in batch}


def read_record(root, entry):
    root = Path(root).resolve(); path = (root / entry["file"]).resolve()
    if not path.is_relative_to(root): raise ValueError("Cache path escapes root")
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]).copy() for k in data.files}


def load_study(root, verify=True):
    from .common import code_sha
    from . import PROTOCOL
    from .sampling import validate_settings
    root = Path(root)
    d = read(root / "study.json")
    if d.get("protocol") != PROTOCOL or not d.get("completed") or d["code_sha256"] != code_sha():
        raise ValueError("Incomplete study or changed executable code")
    validate_settings(d["sampling"])
    if verify:
        for entries in d["records"].values():
            for entry in entries:
                p = (root / entry["file"]).resolve()
                if not p.is_relative_to(root.resolve()) or sha(p) != entry["sha256"]:
                    raise ValueError("Prepared record changed")
        for path, digest in d["assets"].items():
            if sha(path) != digest: raise ValueError("Pinned source changed: " + path)
    return d
