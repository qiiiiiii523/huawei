"""Run both compact candidates with a single global limit of two training jobs."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import pipeline
from .compact import PRESETS, ROOT, config_directory, load_configs


def suite_jobs(args):
    jobs = []
    for preset in PRESETS:
        load_configs(preset)
        child = copy.copy(args)
        child.output = args.output / preset / "seed42"
        child.config_dir = config_directory(preset)
        child.gpu_p = child.gpu_a = child.gpu_b = args.gpu
        for job in pipeline.plan(child):
            job.name = preset + ":" + job.name
            job.dependencies = tuple(preset + ":" + dep for dep in job.dependencies)
            jobs.append(job)
    priority = {"P": 0, "A": 1, "Q": 2, "B": 3}
    return sorted(jobs, key=lambda job: priority[job.name.split(":")[1]])


def run(args):
    if args.max_train not in (1, 2):
        raise ValueError("Suite permits at most two concurrent training jobs")
    args.output = args.output.resolve()
    if args.output == args.reuse_data_from.resolve():
        raise ValueError("Suite output must differ from the cache source")
    args.config_dir = ROOT / "configs/b7_t_compact"
    args.experiment_suite = True
    args.retry_preparation = False
    jobs = suite_jobs(args)
    if args.dry_run:
        return pipeline.run(args, jobs=jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = pipeline.acquire_lock(args.output / ".compact_suite.lock")
    try:
        # Resolve and verify repaired PTB caches before spending time on GPU probes.
        from .data import read_manifest
        scales = None
        for path in pipeline.reused_manifests(args.reuse_data_from).values():
            manifest = read_manifest(path)
            if scales is not None and manifest["scales_uV"] != scales:
                raise ValueError("Reused cache scales disagree")
            scales = manifest["scales_uV"]
            if any(not Path(r["resolved_path"]).is_file() for r in manifest["records"]):
                raise FileNotFoundError("Reused prepared waveform file is missing")
        if args.probe_first:
            for preset in PRESETS:
                # A resumed candidate already passed bootstrap on its original run.
                if (args.output / preset / "seed42/P/last.pt").is_file():
                    continue
                for seconds in (10, 40):
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
                    report = args.output / "probes" / f"{time.time_ns()}_{preset}_{seconds}s.json"
                    command = [sys.executable, "-u", "-m", "baselines.B7.compact_probe",
                        "--preset", preset, "--window-seconds", str(seconds),
                        "--memory-budget-gib", str(args.memory_budget_gib), "--output", str(report)]
                    # Probes are serial; no production training starts until all pass.
                    process = subprocess.Popen(command, cwd=ROOT, env=env)
                    try:
                        code = process.wait()
                        if code:
                            raise RuntimeError(f"GPU probe failed (exit={code}); see {report}")
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
        print(f"[compact-suite] global max_train={args.max_train}; final EMA K16; checkpoint_every=500", flush=True)
        return pipeline.run(args, jobs=jobs)
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/home/qht/huawei/data"))
    p.add_argument("--metadata", type=Path, default=Path("/home/qht/huawei/metadata"))
    p.add_argument("--ptb-root", type=Path, default=Path("/home/qht/huawei/data/PTB-Diagnostic"))
    p.add_argument("--reuse-data-from", type=Path,
                   default=Path("/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42"))
    p.add_argument("--output", type=Path, default=ROOT / "results/B7-T-compact-suite")
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-train", type=int, choices=(1, 2), default=2)
    p.add_argument("--probe-first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory-budget-gib", type=float, default=35.)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
