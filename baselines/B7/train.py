"""B7 training from random initialization; later stages accept B7 checkpoints only."""
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
from .model import B7UNet


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
    if set(cfg) != allowed or cfg["stage"] not in ("public10", "public40", "huawei40"):
        raise ValueError("Config requires stage/model/loss/sampling/training; B5 settings are not accepted")
    model = ModelConfig.from_dict(cfg["model"])
    loss, sampling = LossConfig(**cfg["loss"]), SamplingConfig(**cfg["sampling"])
    expected = 10 if cfg["stage"] == "public10" else 40
    if sampling.window_seconds != expected:
        raise ValueError("Stage and window disagree")
    tr = cfg["training"]
    if set(tr) - {"checkpoint_every", "validation_mode"} != {"seed", "updates", "validate_every", "batch_size", "learning_rate", "weight_decay",
                   "ema_decay", "gradient_clip", "boundary_probability"}:
        raise ValueError("Unexpected or missing training settings")
    if tr.get("validation_mode", "periodic") not in ("periodic", "final_only"):
        raise ValueError("validation_mode must be periodic or final_only")
    if tr.get("validation_mode") == "final_only" and tr["validate_every"] != tr["updates"]:
        raise ValueError("final_only requires validate_every == updates")
    for key in ("updates", "validate_every", "batch_size"):
        if type(tr[key]) is not int or tr[key] < 1:
            raise ValueError("Positive integer required: " + key)
    if "checkpoint_every" in tr and (type(tr["checkpoint_every"]) is not int or tr["checkpoint_every"] < 1):
        raise ValueError("checkpoint_every must be a positive integer")
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
    from .public_qc import PUBLIC_QC_VERSION, PUBLIC_SOURCES
    if (not args.resume and any(r["source"] in PUBLIC_SOURCES for r in manifest["records"])
            and manifest.get("public_qc_version") != PUBLIC_QC_VERSION):
        raise ValueError("New public pretraining requires QC-v2 data; rebuild public10 in a new cache")
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
    model = B7UNet(mc).to(device)
    print(f"[train] stage={cfg['stage']} device={device} batch_size={tr['batch_size']} "
          f"window={sc.window_seconds}s updates={tr['updates']} "
          f"ode_batch={sc.sample_batch_size} validation_batch={sc.validation_batch_size} "
          f"validation_mode={tr.get('validation_mode', 'periodic')} "
          f"pcc_leads={lc.pcc_leads} "
          f"head={mc.output_head} lf_weight={lc.low_frequency_weight} "
          f"lf_cutoff_hz={lc.low_frequency_cutoff_hz} "
          f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    initialization = {"kind": "random_B7", "seed": tr["seed"], "B5_weights_used": False}
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    step, best = 0, -float("inf")
    last_validated_update, pending_validation = -1, False
    identity = {"manifest_sha256": file_sha256(args.manifest), "code_sha256": code_sha(), "config": cfg}
    data_sha = {r["record_id"]: file_sha256(r["resolved_path"]) for r in manifest["records"]}
    identity["record_sha256"] = data_sha
    if args.init_checkpoint:
        source = load_checkpoint(args.init_checkpoint)
        if ModelConfig.from_dict(source["model_config"]) != mc or list(source["scales_uV"]) != list(manifest["scales_uV"]):
            raise ValueError("B7 stage initialization must preserve architecture and frozen scales")
        model.load_state_dict(source["ema"], strict=True)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        initialization = {"kind": "previous_B7_stage", "sha256": file_sha256(args.init_checkpoint),
                          "source_stage": source["stage"], "source_update": source["update"],
                          "source_loss": source.get("loss"), "optimizer_reset": True,
                          "lineage": source["initialization"], "B5_weights_used": False}
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
        restore_rng(old["rng"])
        print(f"[resume] update={step} best_r={best:.6f} validate_every={tr['validate_every']} "
              f"checkpoint_every={tr.get('checkpoint_every', tr['validate_every'])}", flush=True)
    dataset = WindowDataset(manifest, sc.window_seconds, tr["boundary_probability"],
                            include_d12=lc.pcc_leads == 11)
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    def payload(pending=False):
        return {"architecture_id": ARCHITECTURE_ID, "format_version": CHECKPOINT_VERSION,
                "generated_leads": GENERATED_LEADS, "model_config": mc.to_dict(), "loss": asdict(lc),
                "sampling": asdict(sc), "scales_uV": manifest["scales_uV"], "stage": cfg["stage"],
                "score_protocol": SCORE_PROTOCOL,
                "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
                "update": step, "best_r": best, "identity": identity, "rng": pack_rng(),
                "last_validated_update": last_validated_update, "pending_validation": pending,
                "initialization": initialization}

    def validate(training_parts=None):
        nonlocal best, last_validated_update
        evaluated = copy.deepcopy(model).eval()
        evaluated.load_state_dict(ema, strict=True)
        metrics, rows = evaluate(evaluated, manifest, sc, cfg["stage"])
        last_validated_update = step
        won = metrics["r"] > best
        if won:
            best = metrics["r"]
            best_payload = payload()
            save_checkpoint(output / "best_r.pt", best_payload)
            save_checkpoint(output / "best.pt", best_payload)
        save_checkpoint(output / "last.pt", payload())
        report = {"update": step, "validation": metrics, "train": training_parts, "new_best": won}
        with (output / "history.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps(report, ensure_ascii=False) + "\n")
        (output / f"validation_{step:07d}.json").write_text(json.dumps({"metrics": metrics, "records": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        del evaluated

    if not args.resume:
        save_checkpoint(output / "initial.pt", payload())
        if tr.get("validation_mode", "periodic") == "periodic":
            validate()
        else:
            print("[validation] initial and intermediate full validation disabled; final EMA only", flush=True)
    elif pending_validation:
        print(f"[resume] completing interrupted validation at update={step}", flush=True)
        validate()
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
        due_validation = validation_due(tr, step)
        if "checkpoint_every" in tr and (step % tr["checkpoint_every"] == 0 or due_validation):
            save_checkpoint(output / "last.pt", payload(pending=due_validation))
            print(f"[checkpoint] update={step} last_validated={last_validated_update} "
                  f"pending_validation={due_validation}", flush=True)
        if due_validation:
            logged = {k: float(v.detach()) for k, v in {"total": loss, **parts}.items()}
            logged["window_r"] = 1 - logged["pcc"] if logged["pcc_valid_leads"] else None
            validate(logged)
        elif step == 1 or step % 50 == 0:
            details = ""
            if lc.low_frequency_weight:
                details = (f" fm={float(parts['fm'].detach()):.6g}"
                           f" mae={float(parts['mae'].detach()):.6g}"
                           f" pcc={float(parts['pcc'].detach()):.6g}"
                           f" lf={float(parts['low_frequency_mae'].detach()):.6g}")
            print(f"[train] update={step}/{tr['updates']} loss={float(loss.detach()):.6g}{details}", flush=True)
    (output / "completed.json").write_text(json.dumps({"completed": True, "update": step,
        "best_r": best, "selection": "final_ema" if tr.get("validation_mode") == "final_only" else "best_validation_r",
        "elapsed_seconds_this_invocation": time.monotonic() - started}), encoding="utf-8")


def validation_due(training, step):
    return step == training["updates"] or (training.get("validation_mode", "periodic") == "periodic"
                                          and step % training["validate_every"] == 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--init-checkpoint", help="Optional B7 checkpoint for a later stage; B5 is rejected")
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
