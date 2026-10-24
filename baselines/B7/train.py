"""B7-T training from scratch; later stages accept matching B7-T checkpoints only."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from . import ARCHITECTURE_ID, CHECKPOINT_VERSION, GENERATED_LEADS
from .checkpoint import file_sha256, load_checkpoint, pack_rng, restore_rng, save_checkpoint
from .config import LossConfig, ModelConfig, SamplingConfig, read_json
from .data import WindowDataset, read_manifest
from .evaluate import SCORE_PROTOCOL, evaluate
from .losses import flow_loss
from .model import B7Transformer


def code_sha():
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def move(batch, device):
    return {key: move(value, device) if isinstance(value, dict) else value.to(device) for key, value in batch.items()}


def balanced_indices(dataset, count):
    # Equal sources, then equal patients, then equal records. Overlap does not create new patients.
    groups = {}
    for index, row in enumerate(dataset.rows):
        groups.setdefault(row["source"], {}).setdefault(row["subject_id"], []).append(index)
    result = []
    sources = sorted(groups)
    for _ in range(count):
        patients = groups[sources[int(np.random.randint(len(sources)))]]
        subjects = sorted(patients)
        ids = patients[subjects[int(np.random.randint(len(subjects)))]]
        result.append(ids[int(np.random.randint(len(ids)))])
    return result


def validate_settings(cfg):
    allowed = {"stage", "model", "loss", "sampling", "training"}
    if set(cfg) not in (allowed, allowed | {"validation"}) or cfg["stage"] not in ("public10", "public40", "huawei40"):
        raise ValueError("Config requires stage/model/loss/sampling/training; B5 settings are not accepted")
    validation = cfg.get("validation", {"batch_size": 1})
    if (set(validation) != {"batch_size"} or type(validation["batch_size"]) is not int
            or validation["batch_size"] < 1):
        raise ValueError("validation requires a positive integer batch_size")
    model = ModelConfig.from_dict(cfg["model"])
    loss, sampling = LossConfig(**cfg["loss"]), SamplingConfig(**cfg["sampling"])
    expected = 10 if cfg["stage"] == "public10" else 40
    if sampling.window_seconds != expected:
        raise ValueError("Stage and window disagree")
    tr = cfg["training"]
    if set(tr) - {"checkpoint_every"} != {"seed", "updates", "validate_every", "batch_size", "learning_rate", "weight_decay",
                   "ema_decay", "gradient_clip", "boundary_probability"}:
        raise ValueError("Unexpected or missing training settings")
    for key in ("updates", "validate_every", "batch_size"):
        if type(tr[key]) is not int or tr[key] < 1:
            raise ValueError("Positive integer required: " + key)
    if "checkpoint_every" in tr and (type(tr["checkpoint_every"]) is not int or tr["checkpoint_every"] < 1):
        raise ValueError("Positive integer required: checkpoint_every")
    if type(tr["seed"]) is not int or not 0 <= tr["seed"] < 2**32:
        raise ValueError("Invalid training seed")
    if any(not np.isfinite(tr[k]) for k in ("learning_rate", "weight_decay", "ema_decay", "gradient_clip", "boundary_probability")):
        raise ValueError("Nonfinite training configuration")
    if tr["learning_rate"] <= 0 or tr["weight_decay"] < 0 or tr["gradient_clip"] <= 0 or not 0 <= tr["ema_decay"] < 1:
        raise ValueError("Invalid optimizer/EMA configuration")
    if not 0 <= tr["boundary_probability"] <= 1:
        raise ValueError("Invalid boundary sampling probability")
    return model, loss, sampling


def train(args):
    cfg = read_json(args.config)
    mc, lc, sc = validate_settings(cfg)
    manifest = read_manifest(args.manifest)
    tr = cfg["training"]
    if args.resume and args.init_checkpoint:
        raise ValueError("Choose resume or a new B7 stage initialization")
    output = Path(args.output).resolve()
    if args.resume:
        if not (output / "last.pt").is_file():
            raise FileNotFoundError("Resume needs this run's last.pt")
    elif output.exists():
        raise FileExistsError("New stage needs a new output directory")
    if not any(r["split"] == "validation" for r in manifest["records"]):
        raise ValueError("Validation records required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(args.device)
    random.seed(tr["seed"])
    np.random.seed(tr["seed"])
    torch.manual_seed(tr["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tr["seed"])
    torch.use_deterministic_algorithms(True)
    model = B7Transformer(mc).to(device)
    checkpoint_every = tr.get("checkpoint_every", tr["validate_every"])
    print(f"[train] stage={cfg['stage']} batch={tr['batch_size']} window={sc.window_seconds}s "
          f"updates={tr['updates']} validate_every={tr['validate_every']} checkpoint_every={checkpoint_every}", flush=True)
    initialization = {"kind": "random_B7T", "seed": tr["seed"], "B5_weights_used": False, "B7_U7_weights_used": False}
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    step, best = 0, -float("inf")
    last_validated_update, pending_validation, pending_train_parts = -1, False, None
    identity = {"manifest_sha256": file_sha256(args.manifest), "code_sha256": code_sha(), "config": cfg}
    data_sha = {r["record_id"]: file_sha256(r["resolved_path"]) for r in manifest["records"]}
    identity["record_sha256"] = data_sha
    if args.init_checkpoint:
        source = load_checkpoint(args.init_checkpoint)
        if ModelConfig.from_dict(source["model_config"]) != mc or list(source["scales_uV"]) != list(manifest["scales_uV"]):
            raise ValueError("B7 stage initialization must preserve architecture and frozen scales")
        model.load_state_dict(source["ema"], strict=True)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        initialization = {"kind": "previous_B7T_stage", "sha256": file_sha256(args.init_checkpoint),
                          "lineage": source["initialization"], "B5_weights_used": False, "B7_U7_weights_used": False}
    if args.resume:
        old = load_checkpoint(output / "last.pt")
        if old.get("identity") != identity:
            raise ValueError("Resume changed code, config, data identity or budget; use a new stage")
        model.load_state_dict(old["model"], strict=True)
        optimizer.load_state_dict(old["optimizer"])
        ema = {k: v.to(device) for k, v in old["ema"].items()}
        step, best, initialization = old["update"], old["best_r"], old["initialization"]
        last_validated_update = old.get("last_validated_update", step)
        pending_validation = old.get("pending_validation", False)
        pending_train_parts = old.get("pending_train_parts")
        restore_rng(old["rng"])
        print(f"[resume] update={step} best_r={best:.6f} last_validated={last_validated_update} "
              f"pending_validation={pending_validation}", flush=True)
    dataset = WindowDataset(manifest, sc.window_seconds, tr["boundary_probability"])
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    def payload(pending=False, training_parts=None):
        return {"architecture_id": ARCHITECTURE_ID, "format_version": CHECKPOINT_VERSION,
                "generated_leads": GENERATED_LEADS, "model_config": mc.to_dict(), "loss": asdict(lc),
                "sampling": asdict(sc), "scales_uV": manifest["scales_uV"], "stage": cfg["stage"],
                "score_protocol": SCORE_PROTOCOL,
                "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
                "update": step, "best_r": best, "identity": identity, "rng": pack_rng(),
                "last_validated_update": last_validated_update, "pending_validation": pending,
                "pending_train_parts": training_parts if pending else None,
                "initialization": initialization}

    def validate(training_parts=None):
        nonlocal best, last_validated_update
        print(f"[validation] start update={step} K={sc.samples} steps={sc.steps}", flush=True)
        validation_started = time.monotonic()
        evaluated = copy.deepcopy(model).eval()
        evaluated.load_state_dict(ema, strict=True)
        metrics, rows = evaluate(evaluated, manifest, sc, cfg["stage"],
                                 batch_size=cfg.get("validation", {}).get("batch_size", 1))
        validation_seconds = time.monotonic() - validation_started
        last_validated_update = step
        won = metrics["r"] > best
        if won:
            best = metrics["r"]
            best_payload = payload()
            save_checkpoint(output / "best_r.pt", best_payload)
            save_checkpoint(output / "best.pt", best_payload)
        save_checkpoint(output / "last.pt", payload())
        report = {"update": step, "validation": metrics, "train": training_parts, "new_best": won,
                  "validation_seconds": validation_seconds,
                  "validation_batch_size": cfg.get("validation", {}).get("batch_size", 1)}
        with (output / "history.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps(report, ensure_ascii=False) + "\n")
        (output / f"validation_{step:07d}.json").write_text(json.dumps({"metrics": metrics, "records": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        del evaluated

    def save_last(pending=False, training_parts=None):
        started = time.monotonic()
        save_checkpoint(output / "last.pt", payload(pending, training_parts))
        report = {"update": step, "last_validated_update": last_validated_update,
                  "pending_validation": pending, "save_seconds": time.monotonic() - started}
        with (output / "checkpoint_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        print("[checkpoint] " + json.dumps(report), flush=True)

    if not args.resume:
        save_checkpoint(output / "initial.pt", payload())
        save_last(pending=True)
        validate()
    elif pending_validation or (step == tr["updates"] and last_validated_update != step):
        print(f"[resume] completing interrupted validation at update={step}", flush=True)
        validate(pending_train_parts)
    started = time.monotonic()
    for step in range(step + 1, tr["updates"] + 1):
        model.train()
        batch = move(default_collate([dataset[i] for i in balanced_indices(dataset, tr["batch_size"])]), device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = flow_loss(model, batch, lc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for k, value in model.state_dict().items():
                ema[k].mul_(tr["ema_decay"]).add_(value, alpha=1 - tr["ema_decay"])
        due_validation = step % tr["validate_every"] == 0 or step == tr["updates"]
        if step % checkpoint_every == 0 or due_validation:
            logged = {k: float(v.detach()) for k, v in {"total": loss, **parts}.items()}
            logged["window_r"] = 1 - logged["pcc"] if logged["pcc_valid_leads"] else None
            save_last(pending=due_validation, training_parts=logged)
            if due_validation:
                validate(logged)
        if step % 50 == 0:
            print(f"[train] update={step}/{tr['updates']} loss={float(loss.detach()):.6g}", flush=True)
    (output / "completed.json").write_text(json.dumps({"completed": True, "update": step,
        "best_r": best, "elapsed_seconds_this_invocation": time.monotonic() - started}), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--init-checkpoint", help="Optional B7-T checkpoint for a later stage; B5/B7-U7 are rejected")
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
