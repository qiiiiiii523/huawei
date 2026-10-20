"""Prepare full real120s training/validation records and one frozen slow prediction."""
import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
import torch

from baselines.B5.checkpoint import checkpoint_preprocessor, load_checkpoint
from baselines.B5.config import load_config, ModelConfig
from baselines.B5.data import (HuaweiTrainDataset, demographics_table, preprocessing_config,
                              demographics_path, common_config)
from baselines.B5.metadata import read_csv, subject_key
from baselines.B5.objective import validate_mainline_loss
from baselines.B5.runtime import build_validation_datasets, device_from_name, seed_all
from baselines.B5_best.bundle import load_bundle, models
from baselines.B5_best.common import write, sha, state_sha
from baselines.B5_best.validate import pair_indices
from baselines.B5_final.profile import BASE_SHA, SAMPLING, load_profile
from . import PROTOCOL, FIELDS
from .common import (DEFAULT_CONFIG, load_experiment, code_sha, validate_record, visible_digest)
from .inference import frozen_slow


def train_groups(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("source_split") != "train":
            raise ValueError("Strict training rows must be train-only")
        groups[row["target_record_id"]].append(i)
    result = []
    for key in sorted(groups):
        ids = sorted(groups[key], key=lambda i: int(rows[i]["start_sample_500hz"]))
        if ([int(rows[i]["start_sample_500hz"]) for i in ids] != list(range(0, 60000, 5000))
                or len({subject_key(rows[i]["subject_id"]) for i in ids}) != 1):
            raise ValueError("Training record needs12 unique chronological windows from one patient")
        result.append(ids)
    return result


def prepare(args):
    root = Path(args.repository_root).resolve()
    profile = load_profile()
    base = Path(args.checkpoint).resolve() if args.checkpoint else root / profile["base"]["path"]
    if not base.is_file() or sha(base) != BASE_SHA:
        raise ValueError("Use the unchanged B5-final base bundle; specify --checkpoint if moved")
    config_path = Path(args.config).resolve() if args.config else root / profile["validation_config"]
    cfg = load_config(config_path, data_only=True)
    experiment = load_experiment(args.experiment_config)
    bundle = load_bundle(base)
    if (bundle["sampling"] != SAMPLING or bundle.get("slow_beta") != .75
            or ModelConfig.from_dict(cfg["model"]).fingerprint != bundle["fm"]["architecture_hash"]):
        raise ValueError("Data configuration and pinned Final architecture/settings disagree")
    fm_loss = cfg["loss"]
    loss_source = {"kind": "Final_data_run_config", "path": str(config_path), "sha256": sha(config_path)}
    if args.fm_source_checkpoint:
        if sha(args.fm_source_checkpoint) != bundle["provenance"]["fm_sha256"]:
            raise ValueError("Original FM source must match the selected bundle provenance")
        source = load_checkpoint(args.fm_source_checkpoint)
        fm_loss = source.get("loss_config", source.get("config", {}).get("loss"))
        loss_source = {"kind": "matching_raw_FM_checkpoint", "path": str(Path(args.fm_source_checkpoint).resolve()),
                       "sha256": sha(args.fm_source_checkpoint)}
    if fm_loss is None:
        raise ValueError("No auditable original FM loss configuration")
    validate_mainline_loss(fm_loss)
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Use a NEW preparation directory; incomplete prior runs are retained")
    preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(cfg))
    train = HuaweiTrainDataset(cfg, preprocessor, demographics_table(cfg))
    validation = build_validation_datasets(cfg, preprocessor)
    train_subjects = {subject_key(r["subject_id"]) for r in train.rows}
    val_subjects = {subject_key(r["subject_id"]) for ds in validation.values() for r in ds.rows}
    if train_subjects & val_subjects:
        raise ValueError("Train and validation patients overlap")
    split_table = {subject_key(r["subject_id"]): r["split"]
                   for r in read_csv(common_config(cfg).path("subject_split_csv"))}
    if (any(split_table.get(s) != "train" for s in train_subjects)
            or any(split_table.get(s) != "validation" for s in val_subjects)):
        raise ValueError("Patient split disagrees with authoritative metadata")
    device = device_from_name(args.device)
    seed_all(42, True)
    fm, slow_model = models(bundle, device)
    fm.requires_grad_(False)
    slow_model.requires_grad_(False)
    scales = np.asarray(bundle["fm"]["scales"]["d12"], np.float32)
    entries, val_entries, val_rows, seen = [], {}, {}, {}
    out.mkdir(parents=True)

    def save(part, anchor, target, quality, meta, row):
        key = (part, row["target_record_id"])
        digest = visible_digest(anchor, meta)
        target_digest = hashlib.sha256(np.ascontiguousarray(target).tobytes()
                                       + np.ascontiguousarray(quality).tobytes()).hexdigest()
        if key in seen:
            old_digest, old_target, entry = seen[key]
            if old_digest != digest or old_target != target_digest:
                raise ValueError("Repeated physical target has different I/demographics/target/QC")
            return dict(entry)
        record = {k: np.asarray(meta[k]) for k in FIELDS}
        record.update(anchor_uV=np.asarray(anchor, np.float32), target_uV=np.asarray(target, np.float32),
                      mask=np.asarray(quality, bool),
                      slow_uV=frozen_slow(slow_model, anchor, meta, scales, device))
        validate_record(record, training=part == "train")
        relative = f"{part}/{len(seen):05d}.npz"
        path = out / relative
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **record)
        entry = {"file": relative, "sha256": sha(path), "subject_id": subject_key(row["subject_id"]),
                 "target_record_id": row["target_record_id"], "visible_sha256": digest}
        seen[key] = (digest, target_digest, entry)
        print({"prepared": part, "unique_records": len(seen)}, flush=True)
        return dict(entry)

    try:
        for ids in train_groups(train.rows):
            samples = [train.base[i] for i in ids]
            masks = np.stack([s.target_quality_mask for s in samples])
            if not np.all(masks == masks[0]):
                raise ValueError("Training lead QC varies within one physical record")
            target = np.concatenate([s.Y_12lead for s in samples], axis=-1)
            row = train.rows[ids[0]]
            entries.append(save("train", target[:1].copy(), target, masks[0],
                                train.demographics.get(row["subject_id"]), row))
        for task, ds in validation.items():
            val_entries[task], val_rows[task] = [], [None] * len(ds)
            for ids in pair_indices(ds.rows):
                samples = [ds.base[i] for i in ids]
                row = ds.rows[ids[0]]
                target = np.concatenate([s.Y_12lead for s in samples], axis=-1)
                anchor = np.concatenate([s.anchor_i_ecg for s in samples], axis=-1)
                masks = np.stack([s.target_quality_mask for s in samples])
                if not np.all(masks == masks[0]):
                    raise ValueError("Validation lead QC varies within physical record")
                entry = save("validation", anchor, target, masks[0], ds.demographics.get(row["subject_id"]), row)
                entry.update(pair_id=row["pair_id"], input_type=row["input_type"], window_indices=ids)
                val_entries[task].append(entry)
                for i in ids:
                    val_rows[task][i] = ds[i]["evaluation_metadata"]
        common = common_config(cfg)
        sources = {"data_config": sha(config_path), "demographics": sha(demographics_path(cfg)),
                   "subject_split": sha(common.path("subject_split_csv")),
                   "device_qc": sha(common.path("device_interpretation_qc_csv")),
                   "strict_index": sha(common.repository_root / "metadata/d12_strict_pretrain_index.csv")}
        for task in validation:
            directory = common.path(task + "_output")
            for name in (task + "_train_target.npy", task + "_validation_target.npy", task + "_window_metadata.csv"):
                sources[task + "/" + name] = sha(directory / name)
        study = {"protocol": PROTOCOL, "completed": True, "code_sha256": code_sha(),
                 "base_checkpoint": str(base.resolve()), "base_sha256": BASE_SHA,
                 "initial_fm_state_sha256": state_sha(bundle["fm"]["ema"]),
                 "slow_state_sha256": state_sha(bundle["slow"]["model"]),
                 "data_config": str(config_path), "sources": sources,
                 "sampling": SAMPLING, "context": "disabled", "experiment": experiment,
                 "model_config": bundle["fm"]["model_config"], "scale_uV": scales.tolist(),
                 "fm_loss": fm_loss, "fm_loss_source": loss_source,
                 "train": entries, "validation": val_entries, "validation_rows": val_rows,
                 "train_subjects": len(train_subjects), "validation_subjects": len(val_subjects),
                 "true_target_seconds": 120, "target_means_removed": False,
                 "train_FM_ODE_calls": 0, "slow_model_frozen": True}
        write(out / "study.json", study)
        write(out / "preflight.json", {"ready": True, "training_started": False,
              "train_records": len(entries), "validation_pairs": {k: len(v) for k, v in val_entries.items()},
              "train_subjects": len(train_subjects), "validation_subjects": len(val_subjects),
              "source_base_sha256": BASE_SHA, "context": "disabled"})
    finally:
        for ds in validation.values():
            ds.base.close()
        for array in train.base.targets.values():
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default="/home/qht/huawei")
    parser.add_argument("--checkpoint", help="Unchanged Final base.pt; SHA remains pinned")
    parser.add_argument("--config", help="Existing Huawei run config, used for read-only data access")
    parser.add_argument("--fm-source-checkpoint", help="Optional provenance-matching original FM training checkpoint")
    parser.add_argument("--experiment-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
