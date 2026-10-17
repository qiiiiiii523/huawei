"""Explicit two-stage paired training. Original FM objective, no gradients through slow CNN."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import (load_checkpoint, checkpoint_preprocessor, model_from_checkpoint,
                                    atomic_save, pack_rng, restore_rng)
from baselines.B5.config import ModelConfig
from baselines.B5.data import preprocessing_config, HuaweiTrainDataset, demographics_table
from baselines.B5.losses import flow_loss
from baselines.B5.model import B5UNet
from baselines.B5.runtime import device_from_name, seed_all, collate, to_device
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best.bundle import load_bundle
from . import PROTOCOL, GROUPS
from .common import read, write, sha, state_sha, load_study, csv_write, training_lock
from .public_data import PublicDataset
from .stream import PairedStream
from .evaluation import HybridEvaluator, public_diagnostic, candidate_bundle


def cpu_state(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def validate_resume(checkpoint, signature, output):
    if checkpoint.get("experiment_protocol") != PROTOCOL or checkpoint.get("job_signature") != signature:
        raise ValueError("Resume changed study/group/stage/settings/initializer; use a new run")
    if Path(checkpoint["output_directory"]).resolve() != Path(output).resolve():
        raise ValueError("Resume must use its original stage output directory")
    if checkpoint.get("checkpoint_role") != "last":
        raise ValueError("Resume requires last.pt, never best or an EMA-only initializer")


def step_update(model, optimizer, ema, batch, config, scale):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, parts = flow_loss(model, batch, config["loss"], scale)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]), error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        for key, value in model.state_dict().items():
            ema[key].mul_(config["ema_decay"]).add_(value, alpha=1 - config["ema_decay"])
    return {"loss": float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()}}


def checkpoint_payload(source, model, ema, optimizer, signature, stage, output, progress, history, extra):
    keep = ("architecture_id", "architecture_hash", "model_config", "condition_schema", "preprocessing_version", "scales", "scales_sha256")
    return {**{k: copy.deepcopy(source[k]) for k in keep}, "format_version": 1,
            "experiment_protocol": PROTOCOL, "score_protocol": SCORE_PROTOCOL,
            "job_signature": signature, "stage": stage, "epoch": progress if stage == "finetune" else 0,
            "updates": extra["updates"], "progress": progress, "checkpoint_role": "last",
            "output_directory": str(Path(output).resolve()), "model": cpu_state(model.state_dict()), "ema": cpu_state(ema),
            "optimizer": optimizer.state_dict(), "rng": pack_rng(), "history": copy.deepcopy(history), **extra}


def start_job(source_path, study_path, study, group, stage, output, device, resume):
    source = load_checkpoint(source_path)
    root = Path(study_path).parent
    config = read(root / "data_config.json")
    settings = study["settings"]["public"] if stage == "public" else config["training"]
    seed_all(int(settings["seed"]), True)
    model = model_from_checkpoint(source, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    signature = {"study_sha256": sha(study_path), "group": group, "stage": stage, "source_sha256": sha(source_path)}
    previous = None
    if resume and (output / "last.pt").is_file():
        previous = load_checkpoint(output / "last.pt")
        validate_resume(previous, signature, output)
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        optimizer_to(optimizer, device)
        ema = {k: v.to(device).clone() for k, v in previous["ema"].items()}
    elif output.exists() and any(output.iterdir()):
        job = output / "job.json"
        if not resume or not job.is_file() or read(job).get("signature") != signature:
            raise FileExistsError(f"{output} is nonempty; use --resume for the same job, otherwise use a new group output")
        print("No saved update boundary yet; restarting this matching stage from its initializer", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "job.json", {"signature": signature, "settings": settings, "loss": config["loss"],
          "slow_used_in_training": False, "sampling": study["settings"]["sampling"]})
    preprocessor = checkpoint_preprocessor(source, preprocessing_config(config))
    return source, config, settings, model, optimizer, ema, signature, previous, preprocessor


def run_public(study_path, study, group, group_output, device, resume=False):
    root, output = Path(study_path).parent, Path(group_output) / "public"
    source_path = root / "public_initializer.pt"
    source, config, settings, model, optimizer, ema, signature, previous, preprocessor = start_job(
        source_path, study_path, study, group, "public", output, device, resume)
    train = {s: PublicDataset(root / "public", s, "train", preprocessor) for s in ("ptbxl", "chapman")}
    validation = {s: PublicDataset(root / "public", s, "validation", preprocessor) for s in train}
    probability = 0.0 if group == "control" else study["expanded_chapman_probability"]
    stream = PairedStream({s: len(ds) for s, ds in train.items()}, probability, settings["seed"])
    progress, history = 0, []
    if previous:
        progress, history = previous["progress"], previous["history"]
        stream.load_state_dict(previous["stream"])
        restore_rng(previous["rng"])
    if progress > settings["updates"]:
        raise ValueError("Checkpoint exceeds fixed public budget")
    scale = torch.as_tensor(source["scales"]["d12"], device=device)
    train_settings = {**settings, "loss": config["loss"]}
    running = []
    payload = previous
    for update in range(progress + 1, settings["updates"] + 1):
        draws = stream.next(settings["batch_size"])
        batch = collate([train[s][i] for s, i in draws])
        parts = step_update(model, optimizer, ema, to_device(batch, device), train_settings, scale)
        running.append(parts["loss"])
        if update % settings["save_every"] == 0 or update == settings["updates"]:
            record = {"updates": update, "mean_loss_since_save": float(np.mean(running)), "coverage": stream.coverage()}
            history.append(record)
            payload = checkpoint_payload(source, model, ema, optimizer, signature, "public", output, update, history,
                {"updates": update, "stream": stream.state_dict(), "loss_config": config["loss"],
                 "manifests": {"public_cache_manifest": sha(root / "public/manifest.json")},
                 "initialization": {"source_checkpoint": str(source_path), "source_checkpoint_sha256": sha(source_path)}})
            atomic_save(output / "last.pt", payload)
            write(output / "history.json", history)
            print({"group": group, "stage": "public", **record}, flush=True)
            running.clear()
    # Final EMA, not the best score on one source, is the shared Huawei-start rule.
    if payload is None or payload["progress"] != settings["updates"]:
        raise ValueError("Public stage did not reach its fixed update budget")
    final = output / "public_final.pt"
    if not final.is_file():
        atomic_save(final, {**payload, "checkpoint_role": "public_fixed_final_EMA"})
    else:
        stored = load_checkpoint(final)
        if stored["job_signature"] != signature or state_sha(stored["ema"]) != state_sha(payload["ema"]):
            raise ValueError("Existing public_final.pt disagrees with completed training")
    marker = output / "public_validation.json"
    if not marker.is_file():
        before = pack_rng()
        evaluation_model = model_from_checkpoint(payload, device)
        public_diagnostic(evaluation_model, validation, study["settings"], device, marker)
        del evaluation_model
        restore_rng(before)
    write(output / "completed.json", {"completed": True, "stage": "public", "group": group, "updates": settings["updates"],
          "final_checkpoint": str(final), "sha256": sha(final), "selection": "fixed_final_update_EMA", "coverage": stream.coverage()})
    return final


def available_evaluation(output, epoch):
    path = output / "evaluations" / f"epoch_{epoch:04d}"
    if path.exists() and any(path.iterdir()):
        # Interrupted evaluation is never mistaken for a completed candidate; preserve it for diagnosis.
        path.rename(path.with_name(path.name + f"_interrupted_{time.time_ns()}"))
    return path


def restore_best_alias(output, best):
    source = output / best["path"]
    if not source.is_file() or sha(source) != best["sha256"]:
        raise ValueError("Saved best-candidate ledger no longer matches its checkpoint")
    for name in ("best.pt", "best_r.pt"):
        shutil.copyfile(source, output / name)


def run_huawei(study_path, study, group, group_output, device, resume=False):
    root, output = Path(study_path).parent, Path(group_output) / "huawei"
    public_output = Path(group_output) / "public"
    source_path = public_output / "public_final.pt"
    completed = read(public_output / "completed.json")
    if not completed.get("completed") or completed["sha256"] != sha(source_path) or completed["group"] != group:
        raise ValueError("Huawei stage requires this group's completed public stage")
    public_identity = load_checkpoint(source_path).get("job_signature", {})
    if public_identity != {"study_sha256": sha(study_path), "group": group, "stage": "public",
                           "source_sha256": sha(root / "public_initializer.pt")}:
        raise ValueError("Public initializer belongs to another study/group")
    source, config, settings, model, optimizer, ema, signature, previous, preprocessor = start_job(
        source_path, study_path, study, group, "finetune", output, device, resume)
    if settings["gradient_accumulation"] != 1 or settings["amp"] or settings["workers"] != 0:
        raise ValueError("Initial PublicFM pair requires accumulation1/full precision/workers0")
    train = HuaweiTrainDataset(config, preprocessor, demographics_table(config))
    if train.manifest_digest != study["huawei_manifests"]["train"]:
        raise ValueError("Huawei training population changed")
    loader_rng = torch.Generator().manual_seed(settings["seed"])
    loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True, generator=loader_rng,
                        num_workers=0, collate_fn=collate)
    evaluator = HybridEvaluator(study_path, study, device)
    history, best, progress, updates = [], None, 0, 0
    if previous:
        history, best, progress, updates = previous["history"], previous["best"], previous["progress"], previous["updates"]
        loader_rng.set_state(previous["loader_rng"].cpu())
        restore_rng(previous["rng"])
        if best is not None:
            restore_best_alias(output, best)
    if progress > settings["epochs"]:
        raise ValueError("Checkpoint exceeds fixed Huawei epoch budget")
    scale = torch.as_tensor(source["scales"]["d12"], device=device)

    def save(epoch, score=None, train_loss=None):
        nonlocal best, history
        record = {"epoch": epoch, "updates": updates, "train_loss": train_loss, "score": score}
        history.append(record)
        extra = {"updates": updates, "loss_config": config["loss"], "training_config": settings,
                 "sampling_config": study["settings"]["sampling"], "config": config,
                 "manifests": study["huawei_manifests"], "loader_rng": loader_rng.get_state(), "best": best,
                 "selection_metric": "main_correlation", "selection_scope": "fixed_slow_hybrid_full120s",
                 "initialization": {"source_checkpoint": str(source_path), "source_checkpoint_sha256": sha(source_path)}}
        payload = checkpoint_payload(source, model, ema, optimizer, signature, "finetune", output, epoch, history, extra)
        if score is not None:
            value = float(score["main_correlation"])
            if not math.isfinite(value):
                raise ValueError("Nonfinite hybrid validation r")
            name = f"candidates/epoch_{epoch:04d}.pt"
            atomic_save(output / name, {**payload, "checkpoint_role": "candidate", "validation_score": score})
            if best is None or value > best["value"]:
                best = {"epoch": epoch, "value": value, "path": name, "sha256": sha(output / name), "score": score}
                restore_best_alias(output, best)
        payload["best"] = best
        atomic_save(output / "last.pt", payload)
        write(output / "history.json", history)
        scored = [{"epoch": r["epoch"], "updates": r["updates"], **r["score"]} for r in history if r["score"] is not None]
        if scored:
            csv_write(output / "validation_metrics.csv", scored)
        print({"group": group, "stage": "huawei", **record, "best_epoch": best["epoch"], "best_r": best["value"]}, flush=True)

    def validate_ema(epoch):
        rng = pack_rng()
        weights = cpu_state(ema)
        evaluation_model = B5UNet(ModelConfig.from_dict(source["model_config"])).to(device)
        evaluation_model.load_state_dict(weights, strict=True)
        score = evaluator.evaluate(evaluation_model, available_evaluation(output, epoch))
        del evaluation_model
        restore_rng(rng)
        return score

    if best is None:
        # Commit an epoch-zero bootstrap before expensive validation, so an interruption
        # during initial Heun/K16 generation can resume without discarding public training.
        bootstrap = checkpoint_payload(source, model, ema, optimizer, signature, "finetune", output, 0, history,
            {"updates": updates, "best": None, "loader_rng": loader_rng.get_state(), "loss_config": config["loss"]})
        atomic_save(output / "last.pt", bootstrap)
        save(0, validate_ema(0))
    for epoch in range(progress + 1, settings["epochs"] + 1):
        running, count = 0.0, 0
        for batch in loader:
            parts = step_update(model, optimizer, ema, to_device(batch, device), {**settings, "loss": config["loss"]}, scale)
            size = len(batch["anchor"])
            running += parts["loss"] * size
            count += size
            updates += 1
        score = validate_ema(epoch) if epoch % settings["validate_every"] == 0 or epoch == settings["epochs"] else None
        save(epoch, score, running / count)
    last = load_checkpoint(output / "last.pt")
    # Export deployable candidates with the original slow network embedded, never targets.
    for name in ("best", "last"):
        path = output / (name + ".pt")
        checkpoint = load_checkpoint(path)
        reference = best["score"] if name == "best" else history[-1]["score"]
        bundle = candidate_bundle(checkpoint, evaluator.fixed, path, reference, study["huawei_manifests"], study["huawei_sources"])
        atomic_save(output / (name + "_bundle.pt"), bundle)
    write(output / "completed.json", {"completed": True, "group": group, "epochs": settings["epochs"], "best": best,
          "last_score": history[-1]["score"], "best_bundle": str(output / "best_bundle.pt"),
          "best_bundle_sha256": sha(output / "best_bundle.pt"), "slow_updated": False,
          "study_sha256": sha(study_path), "score_protocol": SCORE_PROTOCOL})
    return output / "best_bundle.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--output", required=True, help="Group root; stage outputs are public/ and huawei/")
    p.add_argument("--stage", choices=("public", "huawei", "both"), default="both")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true", help="Resume each existing stage from its own last.pt; completed stages are verified and reused")
    p.add_argument("--execute-training", action="store_true")
    args = p.parse_args()
    if not args.execute_training:
        p.error("Training requires explicit --execute-training")
    path, study = load_study(args.study)
    device = device_from_name(args.device)
    with training_lock(args.output):
        if args.stage in ("public", "both"):
            run_public(path, study, args.group, args.output, device, args.resume)
        if args.stage in ("huawei", "both"):
            run_huawei(path, study, args.group, args.output, device, args.resume)


if __name__ == "__main__":
    main()
