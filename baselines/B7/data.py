"""Prepared raw-uV records, train-only scale fitting and random real windows."""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from . import GENERATED_INDICES
from .config import read_json
from .leads import validate_scales

DATA_PROTOCOL = "B7-raw-uV-records-v1"
GROUPS = ("task1", "machine", "body")


def read_manifest(path):
    path = Path(path).resolve()
    data = read_json(path)
    if data.get("protocol") != DATA_PROTOCOL or data.get("sample_rate") != 500:
        raise ValueError("Expected a B7 500Hz prepared manifest")
    validate_scales(data["scales_uV"])
    if not data.get("records"):
        raise ValueError("Empty manifest")
    seen, splits, targets = set(), {}, {}
    from .public_qc import PUBLIC_QC_VERSION, PUBLIC_SOURCES
    waveform_ids = set()
    for row in data["records"]:
        required = ("record_id", "subject_id", "target_record_id", "source", "group", "path", "split", "input_reliable")
        if any(key not in row for key in required) or row["split"] not in ("train", "validation", "test"):
            raise ValueError("Record metadata missing or invalid")
        if not isinstance(row["input_reliable"], bool):
            raise ValueError("input_reliable must be a boolean")
        if data.get("public_qc_version") == PUBLIC_QC_VERSION and row["source"] in PUBLIC_SOURCES:
            digest = row.get("public_qc", {}).get("waveform_sha256")
            if not digest or digest in waveform_ids:
                raise ValueError("QC-v2 manifest has missing/duplicate waveform identity")
            waveform_ids.add(digest)
        if row["record_id"] in seen:
            raise ValueError("Duplicate record_id")
        seen.add(row["record_id"])
        subject = row["subject_id"]
        if subject in splits and splits[subject] != row["split"]:
            raise ValueError("Patient leakage across splits")
        splits[subject] = row["split"]
        target = row["target_record_id"]
        owner = (subject, row["split"])
        if target in targets and targets[target] != owner:
            raise ValueError("Physical target leakage or inconsistent patient identity")
        targets[target] = owner
        f = Path(row["path"])
        row["resolved_path"] = str((path.parent / f).resolve() if not f.is_absolute() else f.resolve())
    return data


def load_record(row):
    with np.load(row["resolved_path"], allow_pickle=False) as value:
        anchor = value["anchor_uV"].astype("float32")
        target = value["target_uV"].astype("float32")
        mask = value["lead_mask"]
        metadata = {k: value[k].copy() for k in ("numeric", "sex", "field_mask", "age_topcoded") if k in value}
    if target.ndim != 2 or target.shape[0] != 12 or anchor.shape != (1, target.shape[1]):
        raise ValueError("Record needs anchor[1,T] and target[12,T]")
    if mask.shape != (12,) or not np.isin(mask, [0, 1]).all() or not np.isfinite(anchor).all():
        raise ValueError("Invalid quality mask/anchor")
    mask = mask.astype(bool)
    if not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite supervised leads")
    if metadata and set(metadata) != {"numeric", "sex", "field_mask", "age_topcoded"}:
        raise ValueError("All four metadata arrays must be provided together")
    if "window_start" in row or "window_stop" in row:
        start, stop = row.get("window_start"), row.get("window_stop")
        if (type(start) is not int or type(stop) is not int or start < 0 or stop > target.shape[-1]
                or stop - start != 5000 or start % 5000):
            raise ValueError("Indexed Huawei window must be an aligned real 5000-point slice")
        anchor, target = anchor[:, start:stop], target[:, start:stop]
        if "view_lead_mask" in row:
            extra = np.asarray(row["view_lead_mask"])
            if extra.shape != (12,) or not np.isin(extra, [0, 1]).all():
                raise ValueError("Invalid per-window QC mask")
            if (extra.astype(bool) & ~mask).any():
                raise ValueError("A window cannot re-enable excluded parent leads")
            mask = mask & extra.astype(bool)
    return anchor, target, mask, metadata


def fit_scales(records, minimum=100.0):
    ranges = [[] for _ in range(12)]
    count = 0
    for row in records:
        if row["split"] != "train" or not row["input_reliable"]:
            continue
        _, target, mask, _ = load_record(row)
        count += 1
        for j in range(12):
            if mask[j]:
                p = np.percentile(target[j], [5, 95])
                ranges[j].append(float(p[1] - p[0]))
    if not count or any(not r for r in ranges):
        raise ValueError("Training-only valid data for all twelve scale entries required")
    return [max(float(np.median(r)), minimum) for r in ranges]


def checked_records(rows):
    """Ordered real-array reads, at most64 queued records and8 concurrent readers."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            yield from zip(chunk, pool.map(load_record, chunk))


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, window_seconds=40, boundary_probability=0.15, include_d12=False):
        if window_seconds not in (10, 40) or not 0 <= boundary_probability <= 1:
            raise ValueError("Invalid window configuration")
        self.manifest = manifest
        self.width, self.core = window_seconds * 500, 5000
        self.margin = (self.width - self.core) // 2
        self.boundary_probability = boundary_probability
        self.include_d12 = include_d12
        self.scales = np.asarray(manifest["scales_uV"], dtype="float32")
        self.rows = []
        training_rows = [r for r in manifest['records'] if r['split'] == 'train' and r['input_reliable']]
        for row, (anchor, _, mask, _) in checked_records(training_rows):
            if anchor.shape[-1] >= self.width and mask[list(GENERATED_INDICES)].any():
                self.rows.append(row)
        if not self.rows:
            raise ValueError("No reliable real training records long enough for selected window")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        anchor, target, mask, metadata = load_record(self.rows[index])
        length = anchor.shape[-1]
        if self.margin and np.random.rand() < self.boundary_probability:
            start = -self.margin if np.random.rand() < .5 else length - self.core - self.margin
        else:
            start = int(np.random.randint(0, length - self.width + 1))
        physical = np.arange(start, start + self.width)
        valid = (physical >= 0) & (physical < length)
        positions = physical.clip(0, length - 1)
        # Repeated targets outside the real record only define path padding; no supervision there.
        a = anchor[:, positions] / self.scales[:1, None]
        y = target[list(GENERATED_INDICES)][:, positions] / self.scales[list(GENERATED_INDICES), None]
        result = {"anchor": torch.from_numpy(a.copy()), "target": torch.from_numpy(y.copy()),
                  "lead_mask": torch.from_numpy(mask[list(GENERATED_INDICES)].copy()),
                  "time_mask": torch.from_numpy(valid)}
        if self.include_d12:
            # Preserve measured targets/QC; never synthesize supervision from I/II.
            # Values outside the real record are only padding and have time_mask=False.
            result.update(target_d12_uV=torch.from_numpy(target[:, positions].copy()),
                          lead_mask_d12=torch.from_numpy(mask.copy()),
                          scales_uV=torch.from_numpy(self.scales.copy()))
        if metadata:
            result["metadata"] = {k: torch.as_tensor(v) for k, v in metadata.items()}
        else:
            result["metadata"] = {"numeric": torch.zeros(3), "sex": torch.tensor(2),
                                  "field_mask": torch.zeros(4, dtype=torch.bool), "age_topcoded": torch.zeros(1)}
        from .position import absolute_position
        offset = self.rows[index].get("window_start", 0)
        result["metadata"]["record_offset"] = torch.tensor(offset, dtype=torch.long)
        result["metadata"]["record_position"] = absolute_position(torch.from_numpy(positions), offset)[None]
        return result
