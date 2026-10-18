"""Freeze a paired PublicFM study. Audits/copies only; never starts an optimizer."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from baselines.B5.checkpoint import load_checkpoint, checkpoint_preprocessor, atomic_save, scales_digest
from baselines.B5.config import load_config, resolve_path, ModelConfig
from baselines.B5.data import preprocessing_config, common_config, demographics_path
from baselines.B5.objective import loss_signature
from baselines.B5.runtime import build_datasets
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best.bundle import load_inference_bundle, DEFAULT_PROFILE
from ecg12gen.contracts import D12_LEADS
from . import PROTOCOL
from .common import ROOT, read, write, sha, state_sha, code_hash, fresh_directory
from .public_data import audit_public

REFERENCE_RUN = "results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/best.pt"


def public_initializer(explicit, reference):
    ref = load_checkpoint(reference) if reference else None
    initialization = {} if ref is None else ref.get("initialization", {})
    candidate = Path(explicit or initialization.get("source_checkpoint", ""))
    if not candidate.is_file():
        raise FileNotFoundError("Pass --public-checkpoint with the existing public-stage B5-U best.pt; "
                                "automatic resolution reads the original Huawei checkpoint's initialization.source_checkpoint")
    if not explicit and initialization.get("source_checkpoint_sha256") and sha(candidate) != initialization["source_checkpoint_sha256"]:
        raise ValueError("Auto-resolved public initializer has changed; pass an intended public checkpoint explicitly")
    checkpoint = load_checkpoint(candidate)
    if checkpoint.get("stage") != "public":
        raise ValueError("Initializer must be public-stage FM, not Huawei control100 or a hybrid bundle")
    return candidate.resolve(), checkpoint


def validate_settings(settings):
    if settings.get("protocol") != PROTOCOL or settings.get("sampling") != {"seed": 42, "solver": "heun", "steps": 16, "samples": 16}:
        raise ValueError("Paired protocol requires Heun16/K16/seed42")
    if settings.get("slow_beta") != 0.75:
        raise ValueError("This paired experiment fixes beta=0.75")
    p, h = settings["public"], settings["huawei"]
    for key in ("updates", "batch_size", "save_every", "validation_records_per_source"):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError("public." + key + " must be a positive integer")
    for key in ("learning_rate", "gradient_clip"):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError("Invalid public " + key)
    if not 0 <= p["ema_decay"] < 1 or not np.isfinite(p["weight_decay"]) or p["weight_decay"] < 0 or not 0 <= p["seed"] < 2**32:
        raise ValueError("Invalid public optimizer/seed settings")
    if type(h["epochs"]) is not int or h["epochs"] < 1 or h["validate_every"] != 5:
        raise ValueError("Positive Huawei epoch budget and validation every 5 epochs are required")


def huawei_sources(config):
    common = common_config(config)
    paths = {"demographics": demographics_path(config),
             "strict_index": Path(config["repository_root"]) / "metadata/d12_strict_pretrain_index.csv",
             "common_config": resolve_path(config, "common_config"),
             "preprocessing_config": resolve_path(config, "preprocessing_config"),
             "subject_split": common.path("subject_split_csv"),
             "device_QC": common.path("device_interpretation_qc_csv")}
    for task in ("task1", "task2"):
        paths[task + "_pairs"] = common.path(task + "_pair_manifest_csv")
        directory = common.path(task + "_output")
        paths[task + "_metadata"] = directory / (task + "_window_metadata.csv")
        for split in ("train", "validation"):
            paths[f"{task}_{split}_target"] = directory / f"{task}_{split}_target.npy"
    checks = {}
    for name, path in paths.items():
        print("Freezing Huawei data: " + name, flush=True)
        checks[name] = {"path": str(path.resolve()), "sha256": sha(path)}
    return checks


def prepare(args):
    settings = read(args.settings)
    validate_settings(settings)
    repository = Path(args.repository_root).resolve()
    reference = Path(args.data_reference or repository / REFERENCE_RUN)
    public_path, public = public_initializer(args.public_checkpoint, reference if reference.is_file() else None)
    fixed, fixed_path = load_inference_bundle(args.fixed_bundle, args.profile)
    if public["architecture_hash"] != fixed["fm"]["architecture_hash"] or public["scales_sha256"] != fixed["fm"]["scales_sha256"]:
        raise ValueError("Public FM and fixed S50 bundle must share architecture and frozen Huawei scales")
    config = load_config(args.data_config or ROOT / "configs/experiments/b5_main_v2_public_finetune_meta.yaml", data_only=True)
    # The code worktree shares /home/qht/huawei's real caches and metadata.
    config["repository_root"] = str(repository)
    if ModelConfig.from_dict(config["model"]).fingerprint != public["architecture_hash"]:
        raise ValueError("Data config and public initializer model mismatch")
    if loss_signature(public.get("loss_config", public.get("config", {}).get("loss", {}))) != loss_signature(config["loss"]):
        raise ValueError("Public initializer and experiment must preserve the original FM loss")
    config["sampling"] = settings["sampling"]
    config["training"].update(settings["huawei"])
    config["training"]["workers"] = 0  # Exact epoch-boundary resume without worker-side RNG.
    if config["training"]["gradient_accumulation"] != 1 or config["training"]["amp"]:
        raise ValueError("This initial pair preserves batch8, accumulation1, full precision")
    preprocessor = checkpoint_preprocessor(public, preprocessing_config(config))
    train, validation = build_datasets(config, preprocessor)
    output = Path(args.output).resolve()
    resume_audit = bool(getattr(args, "resume_audit", False))
    if resume_audit:
        if (output / "study.json").exists():
            raise ValueError("study.json already exists: completed studies cannot resume audit or change code/data")
        if not (output / "public").is_dir():
            raise FileNotFoundError("--resume-audit requires an existing unfinished shared/public directory")
        old_errors = output / "public/errors.json"
        if old_errors.is_file():
            shutil.copyfile(old_errors, old_errors.with_name(f"errors_before_resume_{time.time_ns()}.json"))
    else:
        output = fresh_directory(output)
    sources = huawei_sources(config)
    summary = audit_public(args.ptbxl_root, args.chapman_root, output / "public", args.chapman_patient_map,
                           args.expected_chapman_records, reuse_cached=resume_audit)
    a, b = (summary["counts"][s]["train"]["records"] for s in ("ptbxl", "chapman"))
    probability = b / (a + b) if args.chapman_probability is None else args.chapman_probability
    if not np.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("Expanded Chapman probability must be strictly between 0 and 1")
    # Copy only inference metadata/EMA; both groups start with exactly these weights and fresh optimizers.
    keep = ("architecture_id", "architecture_hash", "model_config", "condition_schema", "preprocessing_version",
            "scales", "scales_sha256", "ema", "stage", "epoch")
    initializer = {k: copy.deepcopy(public[k]) for k in keep}
    initializer["model"] = initializer["ema"]
    initializer["manifests"] = copy.deepcopy(public.get("manifests", {}))
    initializer["loss_config"] = copy.deepcopy(config["loss"])
    initializer["source_checkpoint_sha256"] = sha(public_path)
    atomic_save(output / "public_initializer.pt", initializer)
    shutil.copyfile(fixed_path, output / "fixed_slow_bundle.pt")
    write(output / "data_config.json", config)
    # Public amplitude/mean diagnostics use train records only; no selection on held-out subjects.
    from .public_data import PublicDataset
    distributions = {}
    for source, dataset in [(s, PublicDataset(output / "public", s, "train", preprocessor)) for s in ("ptbxl", "chapman")] + [("huawei", train)]:
        chosen = np.linspace(0, len(dataset) - 1, min(128, len(dataset)), dtype=int)
        values, good, coverage = [], [], []
        for i in chosen:
            item = dataset[int(i)]
            normalized = item["target"]
            # The frozen preprocessor is scale-only in B5. Decode for comparable physical diagnostics.
            raw = normalized * np.asarray(public["scales"]["d12"], np.float32)[:, None]
            values.append(np.stack((raw.mean(-1), np.ptp(raw, axis=-1)), -1))
            good.append(item["quality_mask"])
            coverage.append(item["field_mask"])
        array = np.stack(values)
        masks = np.stack(good)
        distributions[source] = {"train_sample_count": len(chosen), "lead_order": list(D12_LEADS),
                                 "median_mean_uV": [float(np.median(array[masks[:, j], j, 0])) if masks[:, j].any() else None for j in range(12)],
                                 "median_peak_to_peak_uV": [float(np.median(array[masks[:, j], j, 1])) if masks[:, j].any() else None for j in range(12)],
                                 "demographic_field_coverage": np.stack(coverage).mean(0).tolist()}
    write(output / "train_distribution.json", distributions)
    files = ["public_initializer.pt", "fixed_slow_bundle.pt", "data_config.json", "public/manifest.json",
             "public/audit_summary.json", "public/known_header_exclusions.json", "public/audit.csv", "public/errors.json",
             "train_distribution.json"]
    study = {"protocol": PROTOCOL, "completed": True, "training_started": False, "optimizer_updates": 0,
             "implementation_sha256": code_hash(), "settings": settings,
             "expanded_chapman_probability": probability, "mixture_rule": "audited_train_record_proportions" if args.chapman_probability is None else "explicit_probability",
             "public_source": {"path": str(public_path), "sha256": sha(public_path), "ema_sha256": state_sha(public["ema"])},
             "fixed_slow_source": {"bundle": str(fixed_path), "sha256": sha(fixed_path), "state_sha256": fixed["slow_state_sha256"]},
             "huawei_sources": sources, "huawei_manifests": {"train": train.manifest_digest, **{t: ds.manifest_digest for t, ds in validation.items()}},
             "public_counts": summary["counts"], "files": {name: sha(output / name) for name in files},
             "score_protocol": SCORE_PROTOCOL,
             "selection_metric": "main_correlation", "public_selection": "fixed_final_update_EMA",
             "public_validation": "fixed_first_64_per_source_by_default_diagnostic_only_not_checkpoint_selection"}
    write(output / "study.json", study)
    print({"completed": True, "training_started": False, "study": str(output / "study.json"),
           "expanded_chapman_probability": probability, "public_counts": summary["counts"]}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", default="/home/qht/huawei")
    p.add_argument("--ptbxl-root", required=True)
    p.add_argument("--chapman-root", required=True)
    p.add_argument("--public-checkpoint")
    p.add_argument("--data-reference", help="Original Huawei checkpoint, used only to locate its public initializer")
    p.add_argument("--data-config")
    p.add_argument("--fixed-bundle", help="Portable B5-best bundle whose slow component is frozen")
    p.add_argument("--profile", default=str(DEFAULT_PROFILE))
    p.add_argument("--settings", default=str(ROOT / "configs/b5_public_fm.json"))
    p.add_argument("--chapman-probability", type=float)
    p.add_argument("--chapman-patient-map", help="Optional CSV with record_id,patient_id; otherwise official patient-ECG ID assumption is recorded")
    p.add_argument("--expected-chapman-records", type=int, default=45152, help="Full official release header count; change only for an explicitly audited variant")
    p.add_argument("--resume-audit", action="store_true", help="Re-audit an unfinished shared directory, verify/reuse matching waveform caches; refuses an existing study.json")
    p.add_argument("--output", required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
