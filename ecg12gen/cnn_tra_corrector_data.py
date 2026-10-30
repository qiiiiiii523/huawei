"""Record-aligned CNN prediction caches for two-stage correction training."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .cnn_tra_data import HuaweiSevenLeadData, LEAD_INDICES, TASKS
from .cnn_tra_model import CNNTransformerSevenLead


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask(text: str) -> np.ndarray:
    values = text.split("|")
    if len(values) != 12 or any(v not in ("0", "1") for v in values):
        raise ValueError("expected 12-lead 0|1 mask")
    return np.asarray([v == "1" for v in values], dtype=bool)


@dataclass
class Record:
    anchor: np.ndarray   # [1,60000] raw uV
    base7: np.ndarray    # [7,60000] raw uV
    target12: np.ndarray # [12,60000] raw uV
    mask12: np.ndarray   # [12] bool
    meta: dict


def _stitch(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3 or windows.shape[0] != 12 or windows.shape[-1] != 5000:
        raise ValueError("expected twelve ordered 10-second windows")
    return np.transpose(windows, (1, 0, 2)).reshape(windows.shape[1], 60000).copy()


def _model_from_checkpoint(path: Path, scale8: np.ndarray,
                           device: torch.device) -> CNNTransformerSevenLead:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("step") != 3000 or state.get("architecture") != "B3_CNN_Transformer_FPN_7lead_v1":
        raise ValueError("expected the completed CNN+TRA step-3000 checkpoint")
    if not np.allclose(state["scale_uV"], scale8, rtol=1e-6, atol=1e-5):
        raise ValueError("CNN checkpoint scales disagree with strict training data")
    config = state["configuration"]
    model = CNNTransformerSevenLead(config["transformer_layers"], config["dropout"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def prepare_train_cache(data: HuaweiSevenLeadData, checkpoint: str | Path,
                        output: str | Path, device: torch.device) -> Path:
    """Generate in-sample train predictions; never touch validation targets."""
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"training prediction cache is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = _model_from_checkpoint(checkpoint, data.scale, device)
    eligible = {(item.task, item.index): item for item in data.train_items}
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _rows(data.root / "metadata" / "d12_strict_pretrain_index.csv"):
        key = (row["source_task_id"], int(row["source_array_index"]))
        if key in eligible:
            grouped.setdefault(row["target_record_id"], []).append(row)
    if len(grouped) != 156:
        raise ValueError(f"expected 156 strict physical train records, got {len(grouped)}")

    quality = {}
    for row in _rows(data.root / "metadata" / "device_interpretation_qc.csv"):
        if row["device_type"] == "ecg_machine_d12":
            quality[row["record_id"]] = _mask(row["target_lead_mask"])
    count = len(grouped)
    pred = np.lib.format.open_memmap(output / "prediction_7_uV.npy", mode="w+",
                                     dtype="float32", shape=(count, 7, 60000))
    anchor = np.lib.format.open_memmap(output / "input_i_uV.npy", mode="w+",
                                       dtype="float32", shape=(count, 1, 60000))
    target = np.lib.format.open_memmap(output / "target_12_uV.npy", mode="w+",
                                       dtype="float32", shape=(count, 12, 60000))
    masks = np.empty((count, 12), dtype=bool)
    index_rows = []
    for position, (record_id, rows) in enumerate(sorted(grouped.items())):
        rows.sort(key=lambda row: int(row["window_index"]))
        if len(rows) != 12 or [int(row["window_index"]) for row in rows] != list(range(12)):
            raise ValueError(f"strict train record is not complete: {record_id}")
        if len({row["source_task_id"] for row in rows}) != 1:
            raise ValueError(f"train record spans tasks: {record_id}")
        task = rows[0]["source_task_id"]
        indices = [int(row["source_array_index"]) for row in rows]
        raw = np.asarray(data.train_arrays[task][indices], dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"nonfinite target in train record {record_id}")
        outputs = []
        for start in range(0, 12, 4):
            x = torch.from_numpy(np.array(raw[start:start + 4, 0:1] / data.scale[0],
                                          copy=True)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                y = model(x)
            outputs.append(y.float().cpu().numpy() * data.scale[None, 1:, None])
        prediction = np.concatenate(outputs, axis=0)
        pred[position] = _stitch(prediction)
        anchor[position] = _stitch(raw[:, 0:1])
        target[position] = _stitch(raw)
        masks[position] = quality[record_id]
        if not masks[position, 0]:
            raise ValueError(f"strict train record has unreliable input I: {record_id}")
        index_rows.append({"target_record_id": record_id, "subject_id": rows[0]["subject_id"],
                           "task": task, "source_array_indices": indices,
                           "mask12": masks[position].astype(int).tolist()})
        if (position + 1) % 20 == 0 or position + 1 == count:
            print(f"[cache] {position + 1}/{count} strict train records", flush=True)
    pred.flush(); anchor.flush(); target.flush()
    np.save(output / "target_quality_mask_12.npy", masks)
    (output / "record_index.json").write_text(json.dumps(index_rows, indent=2) + "\n",
                                              encoding="utf-8")
    manifest = {"schema": "cnn_tra_train_record_predictions_v1",
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "train_records": count, "scale8_uV": data.scale.tolist(),
                "sample_rate_hz": 500, "samples_per_record": 60000,
                "output_leads": ["II", "V1", "V2", "V3", "V4", "V5", "V6"],
                "source": "strict train index only; no validation inputs used"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                          encoding="utf-8")
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return output


def load_train_cache(path: str | Path, checkpoint: str | Path) -> list[Record]:
    path = Path(path)
    if not (path / "COMPLETE").exists():
        raise ValueError("training prediction cache is incomplete")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("training cache was generated with another CNN checkpoint")
    pred = np.load(path / "prediction_7_uV.npy", mmap_mode="r")
    anchor = np.load(path / "input_i_uV.npy", mmap_mode="r")
    target = np.load(path / "target_12_uV.npy", mmap_mode="r")
    masks = np.load(path / "target_quality_mask_12.npy")
    rows = json.loads((path / "record_index.json").read_text(encoding="utf-8"))
    if (pred.shape, anchor.shape, target.shape, masks.shape) != (
            (156, 7, 60000), (156, 1, 60000), (156, 12, 60000), (156, 12)):
        raise ValueError("training cache shapes are unexpected")
    if len(rows) != 156 or any(not mask[0] for mask in masks):
        raise ValueError("training cache index or input QC is invalid")
    return [Record(np.asarray(anchor[i]), np.asarray(pred[i]),
                   np.asarray(target[i]), masks[i], rows[i]) for i in range(156)]


def load_validation_records(data: HuaweiSevenLeadData, cache_dir: str | Path) -> list[Record]:
    """Read the frozen step-3000 validation predictions; targets only for loss/score."""
    cache_dir = Path(cache_dir)
    if not (cache_dir / "COMPLETE").exists():
        raise ValueError("step-3000 validation cache is incomplete")
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["step"] != 3000 or manifest["cache_schema"] != "cnn_tra_7lead_validation_v1":
        raise ValueError("expected the CNN step-3000 seven-lead validation cache")
    records = []
    for task in TASKS:
        folder = cache_dir / task
        pred = np.load(folder / "prediction_7_uV.npy", mmap_mode="r")
        cached_i = np.load(folder / "input_i_uV.npy", mmap_mode="r")
        cached_target = np.load(folder / "target_7_uV.npy", mmap_mode="r")
        rows = data.validation_rows[task]
        raw = data.validation_arrays[task]
        mask = data.validation_masks[task]
        if pred.shape != (len(raw), 7, 5000) or cached_i.shape != (len(raw), 1, 5000):
            raise ValueError(f"{task} validation cache shape mismatch")
        pairs: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            pairs.setdefault(row["pair_id"], []).append(i)
        for pair_id, indices in pairs.items():
            indices.sort(key=lambda i: int(rows[i]["window_index"]))
            if len(indices) != 12 or [int(rows[i]["window_index"]) for i in indices] != list(range(12)):
                raise ValueError(f"incomplete validation pair {pair_id}")
            source = np.asarray(raw[indices], dtype=np.float32)
            if not np.allclose(cached_i[indices], source[:, 0:1], rtol=0, atol=1e-5):
                raise ValueError(f"cached I differs from source I in {pair_id}")
            if not np.allclose(cached_target[indices], source[:, list(LEAD_INDICES)],
                               rtol=0, atol=1e-5):
                raise ValueError(f"cached targets differ from source in {pair_id}")
            if not np.all(mask[indices] == mask[indices[0]]):
                raise ValueError(f"quality mask changes within {pair_id}")
            first = rows[indices[0]]
            records.append(Record(_stitch(source[:, 0:1]),
                                  _stitch(np.asarray(pred[indices])), _stitch(source),
                                  mask[indices[0]].copy(),
                                  {"pair_id": pair_id, "task": task,
                                   "input_type": first["input_type"],
                                   "target_record_id": first["target_record_id"],
                                   "subject_id": first["subject_id"],
                                   "validation_array_indices": indices}))
    return records
