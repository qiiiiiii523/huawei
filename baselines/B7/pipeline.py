"""One-command B7 data preparation and P -> (A, Q -> B) execution."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


@dataclass
class Job:
    name: str
    dependencies: tuple
    module: str
    arguments: list
    artifact: Path
    gpu: str | None = None


def plan(args):
    output, raw, metadata = args.output.resolve(), args.data_root.resolve(), args.metadata.resolve()
    cache = output / "data"
    public, ptb, huawei = [cache / name / "manifest.json" for name in ("public10", "ptb40", "huawei")]
    index = cache / "huawei_index.json"
    if getattr(args, "reuse_data_from", None):
        previous = args.reuse_data_from.resolve()
        public = previous / "data/public10/manifest.json"
        huawei = previous / "data/huawei/manifest.json"
        ptb = previous / "ptb_recovery/data/manifest.json"
        if not ptb.is_file():
            ptb = previous / "data/ptb40/manifest.json"
    jobs = [
        Job("index", (), "huawei_index", ["--metadata", str(metadata), "--data-root", str(raw),
            "--anchor-source", "same-record-target-I", "--output", str(index)], index),
        Job("prepare_public", (), "prepare", ["--public-source", "public10", "--root", str(raw),
            "--output", str(public.parent)], public),
        Job("prepare_ptb", ("prepare_public",), "prepare", ["--public-source", "ptbdb", "--root", str(args.ptb_root.resolve()),
            "--scale-manifest", str(public), "--output", str(ptb.parent)], ptb),
        Job("prepare_huawei", ("prepare_public", "index"), "prepare", ["--index", str(index),
            "--scale-manifest", str(public), "--output", str(huawei.parent)], huawei),
    ]
    def training(name, dependencies, manifest, config, gpu, parent=None):
        directory = output / name
        command = ["--manifest", str(manifest), "--config", str(Path(getattr(args, "config_dir", "configs")) / config),
                   "--output", str(directory), "--device", "cuda"]
        if parent:
            command += ["--init-checkpoint", str(output / parent / "best_r.pt")]
        return Job(name, dependencies, "train", command, directory / "completed.json", gpu)
    jobs += [training("P", ("prepare_public",), public, "b7_public10.json", args.gpu_p),
             training("A", ("P", "prepare_huawei"), huawei, "b7_huawei40.json", args.gpu_a, "P"),
             training("Q", ("P", "prepare_ptb"), ptb, "b7_public40.json", args.gpu_b, "P"),
             training("B", ("Q", "prepare_huawei"), huawei, "b7_huawei40.json", args.gpu_b, "Q")]
    if getattr(args, "retry_preparation", False):
        for job in jobs:
            if job.module == "prepare":
                job.arguments.append("--resume")
    if getattr(args, "reuse_data_from", None):
        jobs = [job for job in jobs if job.module == "train"]
        for job in jobs:
            job.dependencies = tuple(dep for dep in job.dependencies if dep in ("P", "A", "Q", "B"))
    return jobs


def preparation_only_retry_allowed(args):
    # Code changes cannot be adopted into an existing trained lineage.
    return (getattr(args, "retry_preparation", False)
            and not any((args.output / name).exists() for name in ("P", "A", "Q", "B")))


def fingerprint(args):
    # Lock the code, settings and metadata across reruns. Raw files are read-only;
    # training additionally verifies its prepared data and config on resume.
    h = hashlib.sha256()
    for folder, pattern in ((Path(__file__).parent, "*.py"), (Path(getattr(args, "config_dir", "configs")), "b7_*.json"),
                            (args.metadata, "*.csv")):
        for path in sorted(folder.glob(pattern)):
            h.update(path.name.encode())
            h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    for key in ("data_root", "metadata", "ptb_root", "output"):
        h.update(str(getattr(args, key).resolve()).encode())
    if getattr(args, "reuse_data_from", None):
        h.update(str(args.reuse_data_from.resolve()).encode())
    h.update(str(Path(getattr(args, "config_dir", "configs")).resolve()).encode())
    return h.hexdigest()


def acquire_lock(path):
    handle = path.open("a+b")
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle  # OS releases the lock on exit, including crashes.


def command_for(job, recover_initial=False):
    arguments = job.arguments.copy()
    if job.module == "train" and job.artifact.parent.exists():
        if not (job.artifact.parent / "last.pt").is_file():
            if not recover_initial:
                raise RuntimeError(f"{job.name}: incomplete directory without last.pt; inspect its log and use a new --output")
            recover_initial_checkpoint(job)
        if "--init-checkpoint" in arguments:
            pos = arguments.index("--init-checkpoint")
            del arguments[pos:pos + 2]
        arguments.append("--resume")
    elif job.module != "train" and not job.artifact.is_file():
        # prepare deliberately never overwrites a partial cache.
        destination = Path(arguments[arguments.index("--output") + 1])
        if destination.exists() and not (job.module == "prepare" and "--resume" in arguments):
            raise RuntimeError(f"{job.name}: partial output {destination}; inspect log and use a new --output")
    return [sys.executable, "-u", "-m", f"baselines.B7.{job.module}", *arguments]


def recover_initial_checkpoint(job):
    """Recover a failed first update without fabricating training progress."""
    from .checkpoint import load_checkpoint, save_checkpoint, file_sha256, model_from_checkpoint
    from .train import code_sha
    from .data import read_manifest
    initial_path = job.artifact.parent / "initial.pt"
    value = load_checkpoint(initial_path)
    manifest_path = Path(job.arguments[job.arguments.index("--manifest") + 1])
    config_path = Path(job.arguments[job.arguments.index("--config") + 1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = read_manifest(manifest_path)
    expected = {"manifest_sha256": file_sha256(manifest_path), "code_sha256": code_sha(),
                "config": config, "record_sha256": {
                    r["record_id"]: file_sha256(r["resolved_path"]) for r in manifest["records"]}}
    if (value.get("update") != 0 or value.get("identity") != expected
            or value.get("stage") != config["stage"]
            or list(value["scales_uV"]) != list(manifest["scales_uV"])
            or any(k not in value for k in ("optimizer", "rng", "model", "ema", "initialization"))):
        raise RuntimeError(f"{job.name}: initial checkpoint identity/state changed; cannot recover")
    model_from_checkpoint(value)
    save_checkpoint(job.artifact.parent / "last.pt", value)
    print(f"[RECOVER_INITIAL] {job.name}: verified initial.pt, resuming update=0", flush=True)


def validate_artifact(job):
    if not job.artifact.is_file():
        raise RuntimeError(f"{job.name} returned without its success artifact: {job.artifact}")
    if job.module == "train":
        value = json.loads(job.artifact.read_text(encoding="utf-8"))
        config_path = Path(job.arguments[job.arguments.index("--config") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if value.get("completed") is not True or value.get("update") != config["training"]["updates"]:
            raise RuntimeError(f"{job.name}: not a completed training stage")
        from .checkpoint import load_checkpoint, file_sha256
        from .train import code_sha
        checkpoint = load_checkpoint(job.artifact.parent / "last.pt")
        manifest = Path(job.arguments[job.arguments.index("--manifest") + 1])
        identity = checkpoint.get("identity", {})
        if (identity.get("config") != config or identity.get("code_sha256") != code_sha()
                or identity.get("manifest_sha256") != file_sha256(manifest)):
            raise RuntimeError(f"{job.name}: completed run identity changed")
        from .data import read_manifest
        records = read_manifest(manifest)["records"]
        if identity.get("record_sha256") != {r["record_id"]: file_sha256(r["resolved_path"]) for r in records}:
            raise RuntimeError(f"{job.name}: completed run data changed")
        best_path = job.artifact.parent / "best_r.pt"
        best = load_checkpoint(best_path)
        if (checkpoint.get("update") != value["update"] or best.get("identity") != identity
                or not math.isfinite(value.get("best_r", float("nan")))
                or best.get("best_r") != value["best_r"] or checkpoint.get("best_r") != value["best_r"]):
            raise RuntimeError(f"{job.name}: completion, last and best checkpoints disagree")
        if config["training"].get("validation_mode") == "final_only":
            if (value.get("selection") != "final_ema" or best.get("update") != value["update"]
                    or checkpoint.get("pending_validation") is not False
                    or checkpoint.get("last_validated_update") != value["update"]):
                raise RuntimeError(f"{job.name}: final EMA validation is incomplete")
        from .checkpoint import model_from_checkpoint
        model_from_checkpoint(best)  # Strictly validate the generated model's state tensors.
        return {"best_r": value["best_r"], "best_checkpoint": str(best_path.resolve()),
                "last_checkpoint": str((job.artifact.parent / "last.pt").resolve()),
                "updates": value["update"]}
    elif job.module == "prepare":
        from .data import read_manifest
        value = read_manifest(job.artifact)
        if any(not Path(r["resolved_path"]).is_file() for r in value["records"]):
            raise RuntimeError(f"{job.name}: prepared record file is missing")
    else:
        if not isinstance(json.loads(job.artifact.read_text(encoding="utf-8")), list):
            raise RuntimeError("Huawei index must be a list")


def run(args, jobs=None):
    os.chdir(Path(__file__).resolve().parents[2])
    jobs = plan(args) if jobs is None else jobs
    if args.dry_run:
        for job in jobs:
            print(json.dumps({"job": job.name, "after": job.dependencies, "gpu": job.gpu,
                              "command": [sys.executable, "-u", "-m", f"baselines.B7.{job.module}", *job.arguments]}, ensure_ascii=False))
        return 0
    for directory in (args.data_root, args.metadata, args.ptb_root):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current Python environment")
    if getattr(args, "reuse_data_from", None):
        from .data import read_manifest
        manifests = {}
        for job in jobs:
            path = Path(job.arguments[job.arguments.index("--manifest") + 1])
            manifests[str(path)] = read_manifest(path)
        scales = [m["scales_uV"] for m in manifests.values()]
        if any(value != scales[0] for value in scales):
            raise ValueError("Reused caches have different scales")
        if any(not Path(r["resolved_path"]).is_file() for m in manifests.values() for r in m["records"]):
            raise FileNotFoundError("Reused prepared waveform file is missing")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(output / ".pipeline.lock")
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    spec = {"fingerprint": fingerprint(args), "raw_files_must_remain_unchanged": True}
    if getattr(args, "experiment_suite", False):
        spec["experiments"] = [{"name": j.name, "after": j.dependencies,
            "arguments": j.arguments, "gpu": None if getattr(args, "gpu_is_runtime", False) else j.gpu,
            "config": json.loads(Path(j.arguments[j.arguments.index("--config") + 1]).read_text())}
            for j in jobs]
        spec = json.loads(json.dumps(spec))
    state_path = output / "pipeline_spec.json"
    if getattr(args, "retry_preparation", False) and not preparation_only_retry_allowed(args):
        raise RuntimeError("--retry-preparation is only allowed before any P/A/Q/B training directory exists")
    if state_path.exists() and json.loads(state_path.read_text()) != spec:
        if not preparation_only_retry_allowed(args):
            raise RuntimeError("Pipeline code, metadata, paths or settings changed; use a new --output")
        backup = output / f"pipeline_spec_before_retry_{time.time_ns()}.json"
        backup.write_bytes(state_path.read_bytes())
        print("[RETRY] Updating preparation-only specification; partial caches will be verified against raw data", flush=True)
    if getattr(args, "retry_preparation", False):
        for job in jobs:
            if job.module == "prepare" and job.artifact.exists():
                raise RuntimeError("--retry-preparation requires incomplete caches; completed caches need independent provenance review")
    state_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (output / "data").mkdir(exist_ok=True)
    status = {job.name: "pending" for job in jobs}
    training_results = {}
    running = {}
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            if stopped:
                raise KeyboardInterrupt
            for name, (job, process, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[name]
                try:
                    if code:
                        raise RuntimeError(f"exit code {code}")
                    result = validate_artifact(job)
                    if job.module == "train":
                        training_results[name] = result
                    status[name] = "done"
                    print(f"[DONE] {name}", flush=True)
                except Exception as exc:
                    status[name] = "failed"
                    print(f"[FAILED] {name}: {exc}; log: {logs / (name + '.log')}", flush=True)
            for job in jobs:
                if status[job.name] != "pending":
                    continue
                if any(status[dep] in ("failed", "blocked") for dep in job.dependencies):
                    status[job.name] = "blocked"
                    print(f"[BLOCKED] {job.name}", flush=True)
                    continue
                if not all(status[dep] == "done" for dep in job.dependencies):
                    continue
                gpu_count = sum(j.gpu is not None for j, _, _ in running.values())
                cpu_count = len(running) - gpu_count
                if (job.gpu is not None and args.max_train > 0 and gpu_count >= args.max_train) or (job.gpu is None and cpu_count >= 1):
                    continue
                per_gpu = getattr(args, "max_per_gpu", 0)
                if (job.gpu is not None and per_gpu > 0
                        and sum(j.gpu == job.gpu for j, _, _ in running.values()) >= per_gpu):
                    continue
                try:
                    if job.artifact.is_file():
                        result = validate_artifact(job)
                        if job.module == "train":
                            training_results[job.name] = result
                        status[job.name] = "done"
                        print(f"[REUSE] {job.name}", flush=True)
                        continue
                    command = command_for(job, recover_initial=getattr(args, "recover_initial", False))
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    env.setdefault("OMP_NUM_THREADS", "4")
                    if job.gpu is not None:
                        env["CUDA_VISIBLE_DEVICES"] = job.gpu
                    log = (logs / (job.name + ".log")).open("a", encoding="utf-8")
                    try:
                        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    except Exception:
                        log.close()
                        raise
                    running[job.name] = (job, process, log)
                    status[job.name] = "running"
                    print(f"[START] {job.name}; log: {logs / (job.name + '.log')}", flush=True)
                except Exception as exc:
                    status[job.name] = "failed"
                    print(f"[FAILED] {job.name}: {exc}", flush=True)
            summary = {"jobs": status.copy(), "success": all(v == "done" for v in status.values()),
                       "training_results": training_results.copy()}
            temporary = output / "pipeline_status.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            temporary.replace(output / "pipeline_status.json")
            if not running and all(v != "pending" for v in status.values()):
                return 0 if summary["success"] else 1
            time.sleep(1)
    finally:
        for _, process, _ in running.values():
            process.terminate()
        for _, process, log in running.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ptb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/B7/auto_seed42"))
    parser.add_argument("--gpu-p", default="0")
    parser.add_argument("--gpu-a", default="0")
    parser.add_argument("--gpu-b", default="0")
    parser.add_argument("--max-train", type=int, choices=(0, 1, 2, 3, 4), default=2, help="0 disables the training concurrency limit")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"), help="Independent P/Q/A/B configuration directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-data-from", type=Path, help="Reuse completed data caches from an old pipeline (including repaired PTB); use a NEW --output")
    parser.add_argument("--retry-preparation", action="store_true", help="Before any training, verify and reuse partial preparation after a reader fix")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
