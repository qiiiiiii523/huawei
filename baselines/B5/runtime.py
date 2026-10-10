"""Explicitly invoked training/validation; no import-time computation or downloads."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg12gen.evaluate import competition_score, evaluate_record_predictions, write_competition_score, write_report
from . import ARCHITECTURE_ID, CONDITION_SCHEMA, PREPROCESSING_VERSION
from .checkpoint import atomic_save, load_checkpoint, pack_rng, restore_rng, scales_digest, scales_payload
from .config import ModelConfig, resolve_path
from .data import HuaweiTrainDataset, HuaweiValidationDataset, common_config, demographics_table, load_preprocessor
from .flow import sample
from .losses import flow_loss
from .model import B5UNet
from .public_adapter import PTBXLDataset, locate_ptbxl
from .objective import file_sha256, validate_resume_loss

INPUT_FIELDS = ("anchor", "numeric", "sex", "field_mask", "age_topcoded")


def initialize_validation_baseline(model, datasets, config, preprocessor, device,
                                   output: Path, selection: str, make_payload) -> float:
    """Save a selectable epoch0 baseline, without any optimizer update."""
    summaries = validate(model, datasets, config, preprocessor, device,
                         output / 'validation' / 'epoch_0000')
    score = float(summaries[selection]['r_missing11'])
    if not math.isfinite(score):
        raise FloatingPointError('Undefined initializer validation selection score')
    (output / 'initial_validation.json').write_text(json.dumps({
        'epoch': 0, 'updates': 0, 'validation': summaries, 'selection_task': selection,
        'selection_value': score}, indent=2), encoding='utf-8')
    payload = make_payload(score)
    atomic_save(output / 'best.pt', payload)
    atomic_save(output / 'last.pt', payload)
    print(json.dumps({'epoch': 0, 'phase': 'initial_validation',
                      'best_validation_r_missing11': score}), flush=True)
    return score


def seed_all(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def seed_worker(_: int) -> None:
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {field: torch.from_numpy(np.stack([np.asarray(row[field]) for row in rows]).copy()) for field in INPUT_FIELDS}
    for field in ("target", "quality_mask"):
        if field in rows[0]:
            result[field] = torch.from_numpy(np.stack([row[field] for row in rows]).copy())
    result["keys"] = [row["key"] for row in rows]
    result["evaluation_metadata"] = [row["evaluation_metadata"] for row in rows]
    if "target_uV" in rows[0]:
        result["target_uV"] = np.stack([row["target_uV"] for row in rows])
    return result


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose --device cpu or configure CUDA on the training machine")
    return device


def build_validation_datasets(config: dict[str, Any], preprocessor: Any) -> dict[str, Any]:
    if config["stage"] == "public":
        root = locate_ptbxl(resolve_path(config, "data_root"), resolve_path(config, "ptbxl_root"))
        return {"public": PTBXLDataset(root, "validation", preprocessor)}
    demographics = demographics_table(config)
    return {task: HuaweiValidationDataset(config, task, preprocessor, demographics)
            for task in config["validation"]["tasks"]}


def build_datasets(config: dict[str, Any], preprocessor: Any) -> tuple[Any, dict[str, Any]]:
    if config["stage"] == "public":
        root = locate_ptbxl(resolve_path(config, "data_root"), resolve_path(config, "ptbxl_root"))
        train = PTBXLDataset(root, "train", preprocessor)
    else:
        train = HuaweiTrainDataset(config, preprocessor, demographics_table(config))
    return train, build_validation_datasets(config, preprocessor)


@torch.no_grad()
def validate(model: B5UNet, datasets: dict[str, Any], config: dict[str, Any], preprocessor: Any,
             device: torch.device, output_dir: Path) -> dict[str, dict[str, Any]]:
    model.eval()
    summaries = {}
    settings = config["sampling"]
    scale = preprocessor.scale_uV_by_source["d12"][None, :, None]
    for name, dataset in datasets.items():
        output = output_dir / name
        output.mkdir(parents=True, exist_ok=True)
        shape = (len(dataset), 12, 5000)
        prediction = np.lib.format.open_memmap(output / "prediction_uV.npy", mode="w+", dtype=np.float32, shape=shape)
        target = np.lib.format.open_memmap(output / "target_uV.npy", mode="w+", dtype=np.float32, shape=shape)
        loader = DataLoader(dataset, batch_size=int(config["validation"]["batch_size"]), shuffle=False,
                            num_workers=int(config["training"]["workers"]), collate_fn=collate,
                            worker_init_fn=seed_worker)
        rows, offset = [], 0
        started = time.perf_counter()
        for batch in loader:
            inputs = to_device({key: batch[key] for key in INPUT_FIELDS}, device)
            generated = sample(model, inputs, batch["keys"], int(settings["seed"]), int(settings["steps"]),
                               settings["solver"], int(settings["samples"]))
            size = len(generated)
            prediction[offset:offset + size] = generated.float().cpu().numpy() * scale
            target[offset:offset + size] = batch["target_uV"]
            rows.extend(batch["evaluation_metadata"])
            offset += size
        prediction.flush()
        target.flush()
        task_id = "task1" if name == "public" else name
        overall, details = evaluate_record_predictions(prediction, target, task_id, rows)
        overall.update({"source_domain": "PTB-XL" if name == "public" else "Huawei", "solver": settings["solver"],
                        "architecture_id": ARCHITECTURE_ID, "condition_schema": CONDITION_SCHEMA,
                        "metadata_enabled": model.config.metadata_enabled,
                        "integration_steps": int(settings["steps"]), "samples": int(settings["samples"]),
                        "velocity_nfe_per_window": int(settings["steps"]) * int(settings["samples"]) *
                        (2 if settings["solver"] == "heun" else 1), "elapsed_seconds": time.perf_counter() - started})
        write_report(output, overall, details, title=f"B5 {name} raw-uV record evaluation")
        with (output / "window_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summaries[name] = overall
        del prediction, target
    if "task1" in summaries and "task2" in summaries:
        score = competition_score(summaries["task1"]["r_missing11"], summaries["task2"]["r_missing11"],
                                  summaries["task2"]["task2_missing_lead_mean_rmse_uV"])
        write_competition_score(output_dir, score)
    return summaries


def run_training(config: dict[str, Any], device_name: str, init_path: str | None = None,
                 resume_path: str | None = None) -> None:
    """Called only by CLI after its explicit --execute-training gate."""
    if init_path and resume_path:
        raise ValueError("Choose either fine-tune initialization or exact resume")
    if config["stage"] == "finetune" and not (init_path or resume_path):
        raise ValueError("Fine-tuning requires --init-checkpoint or --resume")
    device = device_from_name(device_name)
    training = config["training"]
    seed_all(int(training["seed"]), bool(training["deterministic"]))
    preprocessor = load_preprocessor(config)
    model_config = ModelConfig.from_dict(config["model"])
    model = B5UNet(model_config).to(device)
    scale_payload = scales_payload(preprocessor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                 weight_decay=float(training["weight_decay"]))
    amp = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(int(training["seed"]))
    ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
    start_epoch, updates, best_score = 0, 0, -math.inf
    checkpoint = load_checkpoint(resume_path or init_path) if (resume_path or init_path) else None
    initialization = {"type": "random", "source_checkpoint": None}
    if checkpoint:
        if checkpoint["architecture_hash"] != model_config.fingerprint:
            raise ValueError("Cannot initialize a different architecture/condition configuration")
        if checkpoint["scales_sha256"] != scales_digest(scale_payload):
            raise ValueError("Freeze and reuse the checkpoint's Huawei-fitted scale file before transfer")
        if resume_path:
            validate_resume_loss(checkpoint, config['loss'])
            if checkpoint["stage"] != config["stage"]:
                raise ValueError("Exact resume must preserve training stage")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["grad_scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            updates, best_score = int(checkpoint["updates"]), float(checkpoint["best_score"])
            generator.set_state(checkpoint["loader_rng"].cpu())
            ema = {key: value.to(device).clone() for key, value in checkpoint["ema"].items()}
            initialization = checkpoint["initialization"]
        else:
            model.load_state_dict(checkpoint["ema"], strict=True)
            ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
            initialization = {"type": "pretrained_ema", "source_checkpoint": str(Path(init_path).resolve()),
                              "source_checkpoint_sha256": file_sha256(init_path),
                              "source_stage": checkpoint["stage"], "source_epoch": int(checkpoint["epoch"]),
                              "source_manifests": checkpoint["manifests"]}
    train_data, validation_data = build_datasets(config, preprocessor)
    manifests = {"train": train_data.manifest_digest,
                 **{name: data.manifest_digest for name, data in validation_data.items()}}
    if config["stage"] != "public":
        import hashlib
        manifests["demographics_csv"] = hashlib.sha256(resolve_path(config, "demographics").read_bytes()).hexdigest()
        common = common_config(config)
        for key in ("subject_split_csv", "device_interpretation_qc_csv"):
            manifests[key] = hashlib.sha256(common.path(key).read_bytes()).hexdigest()
    selection = "public" if config["stage"] == "public" else config["validation"]["selection_task"]
    if selection not in validation_data:
        raise ValueError("validation.selection_task must be included in validation.tasks")
    if resume_path:
        if checkpoint["manifests"] != manifests or checkpoint["selection_task"] != selection:
            raise ValueError("Resume changed data manifest or checkpoint-selection population")
        for key in ("seed", "batch_size", "gradient_accumulation", "workers", "deterministic", "amp", "ema_decay"):
            if checkpoint["training_config"][key] != training[key]:
                raise ValueError(f"Exact resume changed training.{key}; use --init-checkpoint for a new run")
        restore_rng(checkpoint["rng"])
    output = resolve_path(config, "output_dir")
    if not resume_path and any((output / name).exists() for name in ("last.pt", "best.pt")):
        raise FileExistsError("Output already contains checkpoints; use a new output directory or explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "initialization.json").write_text(json.dumps({
        'initialization': initialization, 'scales_sha256': scales_digest(scale_payload),
        'manifests': manifests, 'selection_task': selection, 'architecture_hash': model_config.fingerprint,
        'loss_config': config['loss'], 'resumed': bool(resume_path)}, indent=2), encoding='utf-8')
    loader = DataLoader(train_data, batch_size=int(training["batch_size"]), shuffle=True, generator=generator,
                        num_workers=int(training["workers"]), collate_fn=collate, worker_init_fn=seed_worker,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    scale_tensor = torch.as_tensor(preprocessor.scale_uV_by_source["d12"], device=device)

    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        return {"format_version": 1, "architecture_id": ARCHITECTURE_ID,
                "architecture_hash": model_config.fingerprint, "model_config": asdict(model_config),
                "condition_schema": CONDITION_SCHEMA, "preprocessing_version": PREPROCESSING_VERSION,
                "scales": scale_payload, "scales_sha256": scales_digest(scale_payload),
                "stage": config["stage"], "epoch": epoch, "updates": updates, "best_score": best_score,
                "selection_task": selection, "selection_metric": "r_missing11", "model": model.state_dict(),
                "ema": ema, "optimizer": optimizer.state_dict(), "grad_scaler": scaler.state_dict(),
                "rng": pack_rng(), "loader_rng": generator.get_state(), "manifests": manifests,
                "training_config": training, "loss_config": config['loss'], "config": config,
                "torch_version": str(torch.__version__), "initialization": initialization}

    if training.get('validate_initial', False) and not resume_path:
        # Seeded target-free ODE validation is identical in both paired runs.
        # epoch=-1 means zero optimizer epochs completed; --resume starts at epoch 1.
        best_score = initialize_validation_baseline(model, validation_data, config, preprocessor,
            device, output, selection, lambda score: {**checkpoint_payload(-1), 'best_score': score})
    for epoch in range(start_epoch, int(training["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, count = {}, 0
        for index, batch in enumerate(loader):
            inputs = to_device(batch, device)
            accumulation = int(training["gradient_accumulation"])
            group_start = (index // accumulation) * accumulation
            divisor = min(accumulation, len(loader) - group_start)
            with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                loss, parts = flow_loss(model, inputs, config["loss"], scale_tensor)
            scaler.scale(loss / divisor).backward()
            for key, value in {"loss": loss, **parts}.items():
                running[key] = running.get(key, 0.) + float(value.detach()) * len(inputs["anchor"])
            count += len(inputs["anchor"])
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                decay = float(training["ema_decay"])
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        ema[key].mul_(decay).add_(value, alpha=1 - decay)
        record: dict[str, Any] = {"epoch": epoch + 1, "updates": updates,
                                  "train": {key: value / count for key, value in running.items()}}
        is_validation_epoch = (epoch + 1) % int(training["validate_every"]) == 0 or epoch + 1 == int(training["epochs"])
        improved = False
        if is_validation_epoch:
            evaluation_model = B5UNet(model_config).to(device)
            evaluation_model.load_state_dict(ema, strict=True)
            summaries = validate(evaluation_model, validation_data, config, preprocessor, device,
                                 output / "validation" / f"epoch_{epoch + 1:04d}")
            score = float(summaries[selection]["r_missing11"])
            if not math.isfinite(score):
                raise FloatingPointError("Undefined validation selection score")
            improved = score > best_score
            best_score = max(best_score, score)
            record.update({"validation": summaries, "selection_task": selection,
                           "selection_metric": "r_missing11", "selection_value": score})
            del evaluation_model
        payload = checkpoint_payload(epoch)
        atomic_save(output / "last.pt", payload)
        if improved:
            atomic_save(output / "best.pt", payload)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"epoch": epoch + 1, "updates": updates, "loss": record["train"]["loss"],
                          "best_validation_r_missing11": best_score}, ensure_ascii=False), flush=True)
