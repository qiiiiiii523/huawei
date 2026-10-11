"""Version-bound body-only B overlay; target/anchor and machine context stay canonical A."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .body_scale_detrend import FILTER_SPEC
from .contracts import ContractError
from .record_context import ContextWindows

OVERLAY_PROTOCOL = "independent-body-context-overlay-v1"
OVERLAY_METADATA = "body_scale_context_window_metadata_B.csv"
MANIFEST = "body_scale_ablation_manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def protected_paths(config):
    base = config.path("task2_output")
    return {
        "target_metadata": base / "task2_window_metadata.csv",
        "context_metadata": base / "task2_context_window_metadata.csv",
        "pair_manifest": config.path("task2_pair_manifest_csv"),
        "subject_split": config.path("subject_split_csv"),
        "device_qc": config.path("device_interpretation_qc_csv"),
    }


class BodyScaleBContextStore:
    def __init__(self, config, split):
        self.split = split
        directory = config.path("task2_body_scale_ablation")
        path = directory / MANIFEST
        if not path.is_file():
            raise ContractError("Body B requires the NEW independent-context ablation; rebuild legacy B")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("protocol") != OVERLAY_PROTOCOL or self.manifest.get("filter") != FILTER_SPEC:
            raise ContractError("Body B cache has an incompatible filter/cache protocol")
        if sha256_file(directory / OVERLAY_METADATA) != self.manifest.get("overlay_files_sha256", {}).get(OVERLAY_METADATA):
            raise ContractError("Body B metadata changed after construction")
        for name, source in protected_paths(config).items():
            if sha256_file(source) != self.manifest.get("canonical_metadata_sha256", {}).get(name):
                raise ContractError(f"Stale B overlay: canonical {name} changed; rebuild B")
        filename = f"body_scale_{split}_context_B.npy"
        maskname = f"body_scale_{split}_context_valid_mask_B.npy"
        for name in (filename, maskname):
            if sha256_file(directory / name) != self.manifest.get("overlay_files_sha256", {}).get(name):
                raise ContractError(f"Body B array changed after construction: {name}")
        self.values = np.load(directory / filename, mmap_mode="r")
        self.masks = np.load(directory / maskname, mmap_mode="r")
        self.rows = sorted([r for r in read_rows(directory / OVERLAY_METADATA) if r["split"] == split],
                           key=lambda r: int(r["array_index"]))
        if (self.values.ndim != 3 or self.values.shape[1:] != (6, 5000) or self.masks.dtype != bool or
                self.masks.shape != self.values.shape or
                [int(r["array_index"]) for r in self.rows] != list(range(len(self.values)))):
            raise ContractError("Body B overlay arrays/indices disagree")
        canonical = {int(r["array_index"]): r for r in read_rows(protected_paths(config)["context_metadata"])
                     if r["split"] == split and r["input_type"] == "body_scale_d6"}
        if len(canonical) != len(self.rows):
            raise ContractError("Body B does not cover all canonical body context windows")
        fields = ("subject_id", "split", "input_type", "input_record_id", "context_window_index",
                  "start_sample_500hz", "valid_length", "valid_sample_count", "context_window_valid",
                  "expected_context_window_count")
        used = set()
        self.groups = defaultdict(list)
        for r in self.rows:
            index = int(r["canonical_array_index"])
            original = canonical.get(index)
            if index in used or original is None or any(r[k] != original[k] for k in fields):
                raise ContractError("Body B changes canonical identity, windowing or validity metadata")
            used.add(index)
            self.groups[r["input_record_id"]].append(r)

    def context(self, record_id, subject_id):
        rows = sorted(self.groups.get(record_id, []), key=lambda r: int(r["start_sample_500hz"]))
        if not rows:
            return ContextWindows(np.empty((0, 6, 5000), np.float32), np.empty((0, 6, 5000), bool),
                                  np.empty(0, bool), np.empty(0, np.int64), np.empty(0, np.int64),
                                  np.zeros(6, np.float32))
        if any(r["subject_id"] != subject_id or r["split"] != self.split for r in rows):
            raise ContractError("Body B context crosses patient/split identity")
        starts = np.asarray([int(r["start_sample_500hz"]) for r in rows], dtype=np.int64)
        lengths = np.asarray([int(r["valid_length"]) for r in rows], dtype=np.int64)
        baselines = [np.fromstring(r["input_record_baseline_uV"], sep="|", dtype=np.float32) for r in rows]
        baseline = baselines[0]
        if (not np.array_equal(starts, np.arange(len(rows)) * 5000) or
                np.any(lengths < 1) or np.any(lengths > 5000) or np.any(lengths[:-1] != 5000) or
                baseline.shape != (6,) or not np.isfinite(baseline).all() or
                any(not np.array_equal(baseline, b) for b in baselines)):
            raise ContractError("Invalid Body B context windowing/record medians")
        indices = [int(r["array_index"]) for r in rows]
        masks = np.asarray(self.masks[indices], dtype=bool).copy()
        if any(masks[i, :, n:].any() for i, n in enumerate(lengths)):
            raise ContractError("Body B padding cannot be valid")
        return ContextWindows(np.asarray(self.values[indices]), masks, masks.any(axis=(1, 2)),
                              lengths, starts, baseline)

    def close(self):
        for array in (self.values, self.masks):
            array._mmap.close()
