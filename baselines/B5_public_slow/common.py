"""Immutable study identity, verified source caches and reproducible per-source sampling."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.slow import load_slow
from baselines.B5.scoring import SCORE_PROTOCOL
from . import PROTOCOL


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, value)
    os.replace(temporary, path)


def empty_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Use a new/empty output; put tee logs BESIDE it: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def code_digest():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for folder in ("baselines/B5_public_slow", "baselines/B5_best", "baselines/B5", "ecg12gen"):
        for path in sorted((root / folder).glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def safe_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Cache file escapes its root")
    return path


def read_record(root, entry):
    with np.load(safe_file(root, entry["file"]), allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def verify_entries(root, entries):
    for entry in entries:
        if sha(safe_file(root, entry["file"])) != entry["sha256"]:
            raise ValueError("Cache record changed: " + entry["file"])


def validate_record(record, scale, huawei=False, training=True):
    anchor, target, mask = record["anchor"], record["slow_target"], record["mask"]
    n = anchor.shape[-1]
    if (anchor.shape != (1, n) or target.shape != (11, n) or mask.shape != (12,) or
            n % 500 or not 2500 <= n <= 6000 or not np.isin(mask, (0, 1)).all()):
        raise ValueError("Malformed record/quality mask")
    if training and (not mask[0] or not mask[1:].any()):
        raise ValueError("Training requires reliable I and at least one supervised target")
    for key in ("anchor", "slow_target", "numeric", "age_topcoded"):
        if not np.isfinite(record[key]).all():
            raise ValueError("Nonfinite " + key)
    if (record["numeric"].shape != (3,) or record["field_mask"].shape != (4,) or
            record["age_topcoded"].shape != (1,) or np.asarray(record["sex"]).shape != () or
            int(record["sex"]) not in (0, 1, 2)):
        raise ValueError("Demographic schema mismatch")
    if huawei:
        from .core import slow_numpy
        if n != 6000 or record["target_uV"].shape != (12, 60000) or record["fm_uV"].shape != (12, 60000):
            raise ValueError("Huawei cache must be full real 120 seconds")
        raw = record["target_uV"]
        if not np.isfinite(raw).all() or not np.isfinite(record["fm_uV"]).all():
            raise ValueError("Nonfinite Huawei cache")
        if not np.allclose(anchor, raw[:1].reshape(1, 6000, 10).mean(-1) / scale[0], atol=2e-6):
            raise ValueError("Cached I/scale mismatch")
        expected = slow_numpy(raw[1:]).reshape(11, 6000, 10).mean(-1) / scale[1:, None]
        if not np.allclose(target, expected, atol=2e-6):
            raise ValueError("Cached slow-label definition/scale mismatch")


def load_study(path, deep=False):
    path = Path(path).resolve()
    study = read(path)
    if study.get("protocol") != PROTOCOL or study.get("score_protocol") != SCORE_PROTOCOL:
        raise ValueError("Different CMP/score protocol")
    if study["code_sha256"] != code_digest():
        raise ValueError("Study code changed; finish with the frozen commit or create a new study")
    if sha(path.parent / "components.pt") != study["bundle_sha256"]:
        raise ValueError("Shared initializer changed")
    bundle = load_bundle(path.parent / "components.pt")
    for name in ("huawei", "public"):
        item = study[name]
        if sha(item["manifest"]) != item["sha256"]:
            raise ValueError("Source manifest changed: " + name)
        if deep:
            manifest = read(item["manifest"])
            verify_entries(item["root"], manifest["train"] + manifest["validation"])
    for item in study["external_checks"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Study/source identity changed: " + item["path"])
    return study, bundle


def model_from_bundle(bundle, device):
    return load_slow(bundle["slow"], device, "slow_f").requires_grad_(True)


class SourceSampler:
    """State is serializable: resume restores both draw order and source exposure counts."""
    def __init__(self, entries, seed, patient_balanced=False):
        if not entries:
            raise ValueError("Empty source")
        self.entries, self.rng = entries, random.Random(seed)
        self.patient_balanced = patient_balanced
        self.patients = {}
        for i, entry in enumerate(entries):
            self.patients.setdefault(entry["subject_id"], []).append(i)
        self.ids = sorted(self.patients)
        self.order, self.position, self.counts = [], 0, {}

    def draw(self):
        if self.patient_balanced:
            i = self.rng.choice(self.patients[self.rng.choice(self.ids)])
        else:
            if self.position >= len(self.order):
                self.order = list(range(len(self.entries)))
                self.rng.shuffle(self.order)
                self.position = 0
            i = self.order[self.position]
            self.position += 1
        self.counts[str(i)] = self.counts.get(str(i), 0) + 1
        return self.entries[i]

    def state(self):
        return {"rng": self.rng.getstate(), "order": self.order, "position": self.position, "counts": self.counts}

    def restore(self, state):
        self.rng.setstate(state["rng"])
        self.order, self.position, self.counts = state["order"], state["position"], state["counts"]

    def coverage(self):
        visited = [self.entries[int(i)] for i in self.counts]
        return {"draws": sum(self.counts.values()), "records_visited": len(visited),
                "patients_visited": len({x["subject_id"] for x in visited}),
                "records_available": len(self.entries), "patients_available": len(self.ids),
                "record_draw_counts": {self.entries[int(i)]["file"]: n for i, n in self.counts.items()}}
