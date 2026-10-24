"""Independent Huawei 10s A1 stage initialized from a completed U7 or B7-T P."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BACKENDS = {"B7-U7-raw-I-v1": "baselines.B7", "B7-T-temporal-transformer-v1": "baselines.B7T_A1"}


def read_parent(path):
    import torch
    raw = torch.load(path, map_location="cpu", weights_only=True)
    module = BACKENDS.get(raw.get("architecture_id"))
    if module is None:
        raise ValueError("A1 needs a B7-U7 or B7-T P checkpoint, not B5")
    checkpoint = importlib.import_module(module + ".checkpoint")
    parent = checkpoint.load_checkpoint(path)
    if parent.get("stage") != "public10":
        raise ValueError("A1 must start from P/public10, not A/Q/B/A1")
    marker = Path(path).parent / "completed.json"
    completed = json.loads(marker.read_text(encoding="utf-8"))
    cfg = parent.get("identity", {}).get("config", {})
    if (completed.get("completed") is not True
            or completed.get("update") != cfg.get("training", {}).get("updates")
            or not 0 <= parent.get("update", -1) <= completed["update"]
            or parent.get("best_r") != completed.get("best_r")):
        raise ValueError("P completion marker and selected checkpoint disagree")
    # Validate the complete state without changing or converting architecture.
    checkpoint.model_from_checkpoint(parent)
    return module, parent


def make_config(module, parent, args):
    cfg = {"stage": "huawei10", "model": copy.deepcopy(parent["model_config"]),
        "loss": copy.deepcopy(parent["loss"]),
        "sampling": {"sample_rate": 500, "window_seconds": 10, "core_seconds": 10,
            "solver": "heun", "steps": 16, "samples": 16, "seed": args.seed},
        "training": {"seed": args.seed, "updates": args.updates, "validate_every": args.updates,
            "checkpoint_every": args.checkpoint_every, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": .01, "ema_decay": .999,
            "gradient_clip": 1., "boundary_probability": 0.}}
    if module == "baselines.B7":
        cfg["sampling"].update(sample_batch_size=args.ode_batch, validation_batch_size=args.validation_records)
        cfg["training"]["validation_mode"] = "final_only"
    else:
        cfg["validation"] = {"batch_size": args.ode_batch, "initial": False}
    cfg = json.loads(json.dumps(cfg))
    importlib.import_module(module + ".train").validate_settings(cfg)
    return cfg


def run(args):
    from baselines.B7.pipeline import acquire_lock
    from baselines.B7.checkpoint import file_sha256
    args.output = args.output.resolve()
    args.parent_checkpoint = args.parent_checkpoint.resolve()
    args.manifest = args.manifest.resolve()
    if args.output == args.parent_checkpoint.parent or args.parent_checkpoint.parent in args.output.parents:
        raise ValueError("A1 output must not be the P directory or inside it")
    if any((args.output / name).exists() for name in ("initial.pt", "last.pt", "best_r.pt", "completed.json")):
        raise ValueError("A1 needs its own wrapper output directory, not an existing P/A/Q/B stage")
    module, parent = read_parent(args.parent_checkpoint)
    cfg = make_config(module, parent, args)
    data = importlib.import_module(module + ".data")
    manifest = data.read_manifest(args.manifest)
    if list(parent["scales_uV"]) != list(manifest["scales_uV"]):
        raise ValueError("A1 must reuse P frozen scales")
    # Keep validation on original whole records with original group/QC identities.
    for row in manifest["records"]:
        if not Path(row["resolved_path"]).is_file():
            raise FileNotFoundError(row["resolved_path"])
        if row["split"] == "validation":
            anchor, target, _, _ = data.load_record(row)
            if anchor.shape[-1] != 60000 or target.shape[-1] != 60000:
                raise ValueError("Huawei validation must retain real full120s records")
    if {r["group"] for r in manifest["records"] if r["split"] == "validation"} != set(data.GROUPS):
        raise ValueError("Huawei validation needs task1/machine/body groups")
    if args.dry_run:
        print(json.dumps({"backend": module, "parent": str(args.parent_checkpoint),
            "manifest": str(args.manifest), "output": str(args.output), "gpu": args.gpu,
            "config": cfg, "reprepare": False}, ensure_ascii=False, indent=2))
        return 0
    trainer = importlib.import_module(module + ".train")
    checkpoint = importlib.import_module(module + ".checkpoint")
    records = {r["record_id"]: file_sha256(r["resolved_path"]) for r in manifest["records"]}
    spec = {"backend": module, "config": cfg, "parent_path": str(args.parent_checkpoint),
            "parent_sha256": file_sha256(args.parent_checkpoint), "manifest_path": str(args.manifest),
            "manifest_sha256": file_sha256(args.manifest), "record_sha256": records,
            "code_sha256": trainer.code_sha()}
    spec = json.loads(json.dumps(spec))
    args.output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.output / ".a1.lock")
    try:
        spec_path = args.output / "a1_spec.json"
        if spec_path.exists():
            if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
                raise RuntimeError("A1 code/config/parent/data changed; do not bypass resume identity")
        elif (args.output / "train").exists():
            raise RuntimeError("Training output exists without A1 specification")
        else:
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        config_path = args.output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != cfg:
            raise RuntimeError("Saved A1 config was edited")
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        stage = args.output / "train"
        complete = stage / "completed.json"
        if complete.is_file():
            report = json.loads(complete.read_text(encoding="utf-8"))
            last = checkpoint.load_checkpoint(stage / "last.pt")
            best = checkpoint.load_checkpoint(stage / "best_r.pt")
            expected = {"config": cfg, "code_sha256": spec["code_sha256"],
                "manifest_sha256": spec["manifest_sha256"], "record_sha256": records}
            if (report.get("completed") is not True or report.get("update") != args.updates
                    or last.get("update") != args.updates or last.get("pending_validation") is not False
                    or last.get("last_validated_update") != args.updates
                    or best.get("update") != args.updates or last.get("identity") != expected
                    or best.get("identity") != expected or report.get("best_r") != best.get("best_r")
                    or last.get("best_r") != best.get("best_r")
                    or not math.isfinite(report.get("best_r", float("nan")))):
                raise RuntimeError("A1 completion/checkpoints disagree")
            checkpoint.model_from_checkpoint(best)
            print("[REUSE] A1 complete: " + json.dumps(report), flush=True)
            return 0
        command = [sys.executable, "-u", "-m", module + ".train", "--manifest", str(args.manifest),
            "--config", str(config_path), "--output", str(stage), "--device", "cuda"]
        if stage.exists():
            if not (stage / "last.pt").is_file():
                initial = checkpoint.load_checkpoint(stage / "initial.pt")
                expected = {"config": cfg, "code_sha256": spec["code_sha256"],
                    "manifest_sha256": spec["manifest_sha256"], "record_sha256": records}
                if initial.get("update") != 0 or initial.get("identity") != expected:
                    raise RuntimeError("Unverified partial A1 initial checkpoint")
                checkpoint.save_checkpoint(stage / "last.pt", initial)
            command += ["--resume"]
        else:
            command += ["--init-checkpoint", str(args.parent_checkpoint)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMP_NUM_THREADS", "4")
        print(f"[A1] backend={module} gpu={args.gpu} train=10s evaluation=full120s final_only=True", flush=True)
        process = subprocess.Popen(command, cwd=ROOT, env=env)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        if code:
            print(f"[FAILED] A1 exit={code}; restart the same command to resume", flush=True)
            return code
        if not complete.is_file():
            raise RuntimeError("A1 exited without completed.json")
        print("[DONE] A1: " + complete.read_text(encoding="utf-8"), flush=True)
        return 0
    finally:
        lock.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("/home/qht/huawei_B7/results/B7/auto_seed42/data/huawei/manifest.json"))
    p.add_argument("--output", type=Path, required=True, help="Independent A1 directory, never an existing A/Q/B")
    p.add_argument("--gpu", required=True)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ode-batch", type=int, default=16)
    p.add_argument("--validation-records", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.gpu.isascii() or not args.gpu.isdigit() or str(int(args.gpu)) != args.gpu:
        p.error("gpu must be one physical GPU index")
    if any(getattr(args, k) < 1 for k in ("updates", "batch_size", "checkpoint_every", "ode_batch", "validation_records")):
        p.error("updates and batch/chunk/checkpoint sizes must be positive")
    return args


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
