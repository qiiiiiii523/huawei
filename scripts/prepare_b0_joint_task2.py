"""Prepare and audit B0 Task2 joint-anchor data for body-scale variants A/B.

Place in huawei/scripts/.  The script fits scales from training subjects only,
never rewrites source arrays, and keeps A/B as separate experiment variants.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


VARIANTS = ("A_raw_window", "B_detrend_0p2Hz_then_window")
SOURCES = ("ecg_machine_d6", "body_scale_d6")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sample_identity(sample):
    return (
        str(sample.subject_id), str(sample.pair_id), str(sample.target_record_id),
        str(sample.window_id), str(sample.input_type),
    )


def audit_sample(sample, expected_split, subject_split):
    sid = str(sample.subject_id)
    if sample.split != expected_split or subject_split.get(sid) != expected_split:
        raise ValueError(f"Task2 subject split mismatch: {sid}")
    if sample.context_source_type not in SOURCES or sample.input_type != sample.context_source_type:
        raise ValueError("Unexpected Task2 context source")
    if sample.anchor_source_type != "ecg_machine_i":
        raise ValueError("Task2 anchor must be ecg_machine_i")
    if sample.context_ecg.shape != (6, 5000) or sample.Y_12lead.shape != (12, 5000):
        raise ValueError("Task2 sample shape mismatch")
    if not np.isfinite(sample.context_ecg).all() or not np.isfinite(sample.Y_12lead).all():
        raise ValueError("Non-finite Task2 waveform")
    if not np.array_equal(sample.anchor_i_ecg, sample.Y_12lead[:1]):
        raise ValueError("Task2 anchor is not same-window target I")
    anchor_mask = np.asarray(sample.anchor_lead_mask, dtype=bool)
    context_mask = np.asarray(sample.context_lead_mask, dtype=bool)
    if anchor_mask.shape != (12,) or not anchor_mask[0] or anchor_mask[1:].any():
        raise ValueError("Only target-time I may be observed")
    if context_mask.shape != (6,) or not context_mask.all():
        raise ValueError("Task2 B0 requires all six context leads")


def collect_variant(dataset, expected_split, subject_split):
    signals = {source: [] for source in SOURCES}
    identities, targets, subjects = [], [], set()
    for sample in dataset:
        audit_sample(sample, expected_split, subject_split)
        identities.append(sample_identity(sample))
        targets.append(sample.Y_12lead.copy())
        signals[sample.context_source_type].append(sample.context_ecg.copy())
        subjects.add(str(sample.subject_id))
    arrays = {
        source: np.stack(values).astype(np.float32, copy=False)
        for source, values in signals.items() if values
    }
    return arrays, identities, targets, subjects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/b0_joint_anchor/task2/preprocessing")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {output}")

    cfg = ECGDataConfig.from_yaml(ROOT / "configs/common.yaml")
    split_rows = read_csv(cfg.path("subject_split_csv"))
    subject_split = {row["subject_id"]: row["split"] for row in split_rows}
    if len(subject_split) != len(split_rows):
        raise ValueError("Duplicate subject IDs in split file")

    strict = StrictD12PretrainDataset(cfg, "d12_i_pretrain")
    if not len(strict):
        raise ValueError("No strict D12 training windows")
    strict_subjects, strict_ids = set(), set()
    d12 = np.empty((len(strict), 12, 5000), dtype=np.float32)
    for index, sample in enumerate(strict):
        sid = str(sample.meta["subject_id"])
        strict_id = str(sample.meta["strict_id"])
        if sample.split != "train" or subject_split.get(sid) != "train":
            raise ValueError(f"Non-training subject in strict index: {sid}")
        if strict_id in strict_ids:
            raise ValueError(f"Duplicate strict ID: {strict_id}")
        if not np.array_equal(sample.X_ecg, sample.Y_12lead[:1]):
            raise ValueError("Strict input is not same-window D12 I")
        strict_ids.add(strict_id)
        strict_subjects.add(sid)
        d12[index] = sample.Y_12lead

    variant_data = {}
    for variant in VARIANTS:
        variant_data[variant] = {}
        for split in ("train", "validation"):
            dataset = JointAnchorDataset(cfg, "task2", split, body_scale_variant=variant)
            if not len(dataset):
                raise ValueError(f"No eligible Task2 {variant} {split} samples")
            variant_data[variant][split] = collect_variant(dataset, split, subject_split)

    for split in ("train", "validation"):
        a_arrays, a_ids, a_targets, _ = variant_data[VARIANTS[0]][split]
        b_arrays, b_ids, b_targets, _ = variant_data[VARIANTS[1]][split]
        if a_ids != b_ids or len(a_targets) != len(b_targets):
            raise ValueError(f"A/B {split} row identity mismatch")
        if any(not np.array_equal(a, b) for a, b in zip(a_targets, b_targets)):
            raise ValueError(f"A/B {split} target mismatch")
        if "ecg_machine_d6" in a_arrays and not np.array_equal(
            a_arrays["ecg_machine_d6"], b_arrays["ecg_machine_d6"]
        ):
            raise ValueError("Machine D6 must be unchanged between A/B")
        if "body_scale_d6" not in a_arrays or "body_scale_d6" not in b_arrays:
            raise ValueError("Both variants require eligible body-scale samples")
        if np.array_equal(a_arrays["body_scale_d6"], b_arrays["body_scale_d6"]):
            raise ValueError("Body-scale A/B arrays are unexpectedly identical")

    train_subjects = variant_data[VARIANTS[0]]["train"][3]
    validation_subjects = variant_data[VARIANTS[0]]["validation"][3]
    if validation_subjects & (strict_subjects | train_subjects):
        raise ValueError("Training/validation subject overlap")

    print("PASS: Task2 A/B identity, target and split checks", flush=True)
    print("PASS: machine D6 unchanged; only body-scale context uses 0.2 Hz variant", flush=True)
    pre_cfg = PreprocessingConfig.from_yaml(cfg.path("preprocessing_config"))
    scales_by_variant = {}
    output.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        train_arrays = variant_data[variant]["train"][0]
        missing = set(SOURCES) - set(train_arrays)
        if missing:
            raise ValueError(f"No training samples for sources: {sorted(missing)}")
        preprocessor = ECGPreprocessor.fit(pre_cfg, {"d12": d12, **train_arrays})
        scales = preprocessor.scale_uV_by_source
        if not np.array_equal(scales["ecg_machine_i"], scales["d12"][:1]):
            raise ValueError("Machine-I scale must equal D12-I scale")
        suffix = "A" if variant == VARIANTS[0] else "B"
        scale_path = output / f"preprocessing_scales_{suffix}.npz"
        np.savez_compressed(scale_path, **scales)
        scales_by_variant[variant] = {
            "path": scale_path.name,
            "sha256": sha(scale_path),
            "values": {key: value.tolist() for key, value in scales.items()},
        }
    if not np.array_equal(
        np.asarray(scales_by_variant[VARIANTS[0]]["values"]["d12"]),
        np.asarray(scales_by_variant[VARIANTS[1]]["values"]["d12"]),
    ):
        raise ValueError("A/B canonical D12 scales differ")
    if not np.array_equal(
        np.asarray(scales_by_variant[VARIANTS[0]]["values"]["ecg_machine_d6"]),
        np.asarray(scales_by_variant[VARIANTS[1]]["values"]["ecg_machine_d6"]),
    ):
        raise ValueError("A/B machine-D6 scales differ")

    source_paths = {
        "common.yaml": ROOT / "configs/common.yaml",
        "preprocessing.yaml": cfg.path("preprocessing_config"),
        "subject_split.csv": cfg.path("subject_split_csv"),
        "d12_strict_pretrain_index.csv": ROOT / "metadata/d12_strict_pretrain_index.csv",
        "pair_manifest_task2.csv": cfg.path("task2_pair_manifest_csv"),
        "task2_window_metadata.csv": cfg.path("task2_output") / "task2_window_metadata.csv",
        "body_scale_b_metadata.csv": cfg.path("task2_body_scale_b_metadata"),
        "training_protocol.yaml": cfg.path("training_protocol_config"),
        "context_fusion_protocol.yaml": ROOT / "configs/context_fusion_protocol.yaml",
        "losses.yaml": ROOT / "configs/losses.yaml",
        "prepare_b0_joint_task2.py": Path(__file__),
    }
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    for name, source in source_paths.items():
        data = source.read_bytes()
        (snapshot / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()

    counts = {}
    for variant in VARIANTS:
        counts[variant] = {}
        for split in ("train", "validation"):
            arrays, identities, _, subjects = variant_data[variant][split]
            counts[variant][split] = {
                "windows": len(identities), "subjects": len(subjects),
                "ecg_machine_d6_windows": len(arrays.get("ecg_machine_d6", [])),
                "body_scale_d6_windows": len(arrays.get("body_scale_d6", [])),
            }
    summary = {
        "completed": True, "task_id": "task2", "protocol": "joint_anchor",
        "git_head": git_head(), "fit_split": "train",
        "strict_train_windows": len(strict), "strict_train_subjects": len(strict_subjects),
        "train_validation_subject_overlap": 0,
        "validation_used_to_fit_scales": False,
        "variants_are_separate_experiments": True,
        "machine_d6_variant": "raw_window_for_both_A_and_B",
        "body_scale_variants": list(VARIANTS),
        "counts": counts, "scales": scales_by_variant, "source_sha256": hashes,
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Strict train: {len(strict)}")
    for variant in VARIANTS:
        print(f"{variant}: {counts[variant]}")
    print(f"Completed. Results: {output}")


if __name__ == "__main__":
    main()
