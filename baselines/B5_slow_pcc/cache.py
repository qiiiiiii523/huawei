"""Generate TRAIN-only fixed FM ensembles once; reuse historical validation FM."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from baselines.B5.checkpoint import checkpoint_preprocessor, model_from_checkpoint
from baselines.B5.data import preprocessing_config
from baselines.B5.flow import sample
from baselines.B5.runtime import build_datasets, device_from_name, seed_all
from baselines.B5_best.common import read, sha
from baselines.B5_best.combine import slow_curve
from baselines.B5_best.diagnose import csv_read, align, target_file
from baselines.B5_best.validate import pair_indices
from . import CACHE_PROTOCOL
from .common import atomic_json, load_study, load_cache


def train_groups(rows):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["target_record_id"], []).append(i)
    for key, indices in sorted(groups.items()):
        indices.sort(key=lambda i: int(rows[i]["start_sample_500hz"]))
        if [int(rows[i]["start_sample_500hz"]) for i in indices] != list(range(0, 60000, 5000)):
            raise ValueError("Need twelve real chronological windows: " + key)
        if len({rows[i]["subject_id"] for i in indices}) != 1:
            raise ValueError("Target mixes patients")
        yield indices


def record_data(ds, indices, scales, training):
    first = ds.rows[indices[0]]
    if training:
        samples = [ds.base[i] for i in indices]
        raw = np.concatenate([s.Y_12lead for s in samples], axis=-1).astype(np.float32)
        anchor = raw[:1].copy()
        masks = [np.asarray(s.target_quality_mask, bool) for s in samples]
    else:
        samples = [ds[i] for i in indices]
        raw = np.concatenate([s["target_uV"] for s in samples], axis=-1).astype(np.float32)
        anchor = np.concatenate([ds.base[i].anchor_i_ecg for i in indices], axis=-1).astype(np.float32)
        masks = [s["quality_mask"] for s in samples]
    if raw.shape != (12, 60000) or anchor.shape != (1, 60000):
        raise ValueError("Cache requires complete 120-second real records")
    if not np.isfinite(raw).all() or not np.isfinite(anchor).all() or not np.array_equal(raw[:1], anchor):
        raise ValueError("Expected finite same-record synchronous I/target")
    if not all(np.array_equal(masks[0], v) for v in masks) or not masks[0][1:].any():
        raise ValueError("Inconsistent/empty target QC")
    if training and not masks[0][0]:
        raise ValueError("Training requires usable I")
    meta = ds.demographics.get(first["subject_id"])
    packed = {"anchor": (anchor.reshape(1, 6000, 10).mean(-1) / scales[0]).astype(np.float32),
              "target_uV": raw, "mask": np.asarray(masks[0], bool),
              "slow_target": (slow_curve(raw[1:]).reshape(11, 6000, 10).mean(-1) /
                              scales[1:, None]).astype(np.float32), **meta}
    digest = hashlib.sha256(anchor.tobytes())
    for key, value in sorted(packed.items()):
        digest.update(key.encode()); digest.update(np.asarray(value).tobytes())
    return anchor, meta, packed, digest.hexdigest()


@torch.no_grad()
def generate_fm(model, anchor, meta, key, scales, sampling, device, batch_windows=8):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("FM cache generator must be frozen and eval")
    result = np.empty((12, 60000), np.float32)
    for offset in range(0, 12, batch_windows):
        starts = list(range(offset * 5000, min(offset + batch_windows, 12) * 5000, 5000))
        inputs = {"anchor": torch.from_numpy(np.stack([
            anchor[:, s:s + 5000] / scales[0] for s in starts])).to(device)}
        inputs.update({k: torch.from_numpy(np.stack([v] * len(starts))).to(device) for k, v in meta.items()})
        generated = sample(model, inputs, [key + ":" + str(s) for s in starts], **sampling)
        values = generated.float().cpu().numpy() * scales[None, :, None]
        for i, start in enumerate(starts):
            result[:, start:start + 5000] = values[i]
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite fixed FM prediction")
    return result


def build(study_path, output, device_name, resume=False):
    study_path, output = Path(study_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Use empty cache output or explicit --resume; put log beside it")
    study, bundle = load_study(study_path, verify_data=True)
    if (output / "cache.json").is_file():
        load_cache(output, study_path)
        print("Cache already complete and verified; no new FM generation", flush=True)
        return
    progress_path = output / "progress.json"
    if resume and not progress_path.is_file():
        raise FileNotFoundError("No cache progress.json to resume")
    output.mkdir(parents=True, exist_ok=True)
    progress = read(progress_path) if resume else {"protocol": CACHE_PROTOCOL,
                    "study_sha256": sha(study_path), "records": {}}
    if progress["protocol"] != CACHE_PROTOCOL or progress["study_sha256"] != sha(study_path):
        raise ValueError("Partial cache belongs to a different study")
    atomic_json(progress_path, progress)
    preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(study["data_config"]))
    train, validation = build_datasets(study["data_config"], preprocessor)
    manifests = {"train": train.manifest_digest, **{k: ds.manifest_digest for k, ds in validation.items()}}
    if manifests != study["manifests"]:
        raise ValueError("Train/validation populations changed")
    device = device_from_name(device_name)
    seed_all(42, True)
    fm = model_from_checkpoint(bundle["fm"], device).requires_grad_(False).eval()
    scales = np.asarray(bundle["fm"]["scales"]["d12"], np.float32)
    manifest = {"protocol": CACHE_PROTOCOL, "study_sha256": sha(study_path), "completed": False,
                "FM_optimizer_updates": 0, "sampling": study["sampling"],
                "train": [], "validation": [], "validation_rows": {}}

    def store(split, first, packed, data_sha, make_fm):
        key = split + ":" + first["target_record_id"]
        if key in progress["records"]:
            previous = progress["records"][key]
            if previous["data_sha256"] != data_sha or sha(output / previous["file"]) != previous["sha256"]:
                raise ValueError("Partial record cache/input changed: " + key)
            return previous
        packed["fm_uV"] = make_fm()
        relative = split + "/" + hashlib.sha256(key.encode()).hexdigest()[:24] + ".npz"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **packed)
        os.replace(temporary, path)
        entry = {"target_record_id": first["target_record_id"], "subject_id": first["subject_id"],
                 "file": relative, "sha256": sha(path), "data_sha256": data_sha}
        progress["records"][key] = entry
        atomic_json(progress_path, progress)
        return entry

    for count, indices in enumerate(train_groups(train.rows), 1):
        first = train.rows[indices[0]]
        anchor, meta, packed, data_sha = record_data(train, indices, scales, True)
        entry = store("train", first, packed, data_sha, lambda: generate_fm(
            fm, anchor, meta, first["target_record_id"], scales, study["sampling"], device))
        manifest["train"].append(entry)
        print(f"TRAIN fixed FM cache {count}/{study['train_records']}: {first['target_record_id']}", flush=True)
    del fm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for path, expected in study["source_evaluation_checks"].items():
        if sha(path) != expected:
            raise ValueError("Historical FM validation changed: " + path)
    seen = {}
    # Validation FM is reused only for validation; it never enters the training dataset.
    for task, ds in validation.items():
        from contextlib import ExitStack
        folder = Path(study["source_validation_evaluation"]) / task
        with ExitStack() as stack:
            old_rows = csv_read(folder / "window_metadata.csv")
            order = align(ds.rows, old_rows)
            predicted = np.load(folder / "prediction_uV.npy", mmap_mode="r", allow_pickle=False)
            target = np.load(target_file(folder), mmap_mode="r", allow_pickle=False)
            stack.callback(predicted._mmap.close); stack.callback(target._mmap.close)
            rows = [None] * len(ds)
            for indices in pair_indices(ds.rows):
                first = ds.rows[indices[0]]
                _, _, packed, data_sha = record_data(ds, indices, scales, False)
                ids = order[indices]
                restored = np.concatenate(predicted[ids], axis=-1).astype(np.float32)
                if not np.array_equal(np.concatenate(target[ids], axis=-1), packed["target_uV"]):
                    raise ValueError("Historical and current validation targets differ")
                key = first["target_record_id"]
                if key in seen:
                    previous = seen[key]
                    if previous["data_sha256"] != data_sha:
                        raise ValueError("Same validation target has different I/metadata/labels/QC")
                    with np.load(output / previous["file"], allow_pickle=False) as record:
                        if not np.allclose(record["fm_uV"], restored, atol=.002, rtol=1e-6):
                            raise ValueError("Repeated validation target has different FM ensemble")
                    entry = previous
                else:
                    entry = store("validation", first, packed, data_sha, lambda: restored)
                    manifest["validation"].append(entry)
                    seen[key] = entry
                for i in indices:
                    rows[i] = ds[i]["evaluation_metadata"]
            manifest["validation_rows"][task] = rows
    if {r["subject_id"] for r in manifest["train"]} & {r["subject_id"] for r in manifest["validation"]}:
        raise ValueError("Cache train/validation subject leakage")
    if len(manifest["train"]) != study["train_records"] or len(manifest["validation"]) != study["validation_unique_targets"]:
        raise ValueError("Incomplete cache population")
    manifest["completed"] = True
    atomic_json(output / "cache.json", manifest)
    print({"completed": True, "train_records": len(manifest["train"]),
           "validation_records": len(manifest["validation"]), "FM_optimizer_updates": 0}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute-inference", action="store_true")
    args = p.parse_args()
    if not args.execute_inference:
        p.error("Pass --execute-inference on the server to generate the fixed FM training cache")
    build(args.study, args.output, args.device, args.resume)


if __name__ == "__main__":
    main()
