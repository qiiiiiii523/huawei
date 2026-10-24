"""M2/M3: reuse B7 waveform caches and directly run one bounded experiment graph."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.B7 import pipeline


def suite_jobs(args):
    jobs = []
    for candidate in ("M2", "M3"):
        child = copy.copy(args)
        child.output = args.output.resolve() / (candidate + "_seed42")
        child.config_dir = ROOT / "configs" / ("u7_" + candidate.lower())
        child.gpu_p = child.gpu_a = child.gpu_b = args.gpu
        for job in pipeline.plan(child):
            job.name = candidate + "_" + job.name
            job.dependencies = tuple(candidate + "_" + dep for dep in job.dependencies)
            jobs.append(job)
    # Start both P stages first; downstream stages remain dependency-driven.
    priority = {"P": 0, "A": 1, "Q": 2, "B": 3}
    return sorted(jobs, key=lambda j: priority[j.name.split("_")[1]])


def choose_parallelism(peaks, free_gib, requested):
    # Probe is a planning estimate; keep headroom for workspace/allocator variation.
    per_job = max(peaks) * 1.15 + 2
    slots = min(requested or 4, int(free_gib // per_job))
    if slots < 1:
        raise RuntimeError(f"Insufficient free GPU memory: {free_gib:.1f} GiB; estimated slot {per_job:.1f} GiB")
    return slots


def checked_subprocess(command, log, env):
    from scripts.run_b7_large import execute
    code = execute(command, log, env)
    if code:
        raise RuntimeError(f"GPU check failed (exit={code}); see {log}")
    reports = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
    return reports[-1]


def run(args):
    args.output = args.output.resolve()
    args.config_dir = ROOT / "configs"
    args.experiment_suite = True
    args.retry_preparation = False
    jobs = suite_jobs(args)
    if args.dry_run:
        return pipeline.run(args, jobs=jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = pipeline.acquire_lock(args.output / ".structures.lock")
    try:
        # Fail early on missing caches without re-preparing or changing raw data.
        from baselines.B7.data import read_manifest
        scales = None
        for path in sorted({j.arguments[j.arguments.index("--manifest") + 1] for j in jobs}):
            manifest = read_manifest(path)
            if scales is not None and scales != manifest["scales_uV"]:
                raise ValueError("Reused caches have different frozen scales")
            scales = manifest["scales_uV"]
            if any(not Path(r["resolved_path"]).is_file() for r in manifest["records"]):
                raise FileNotFoundError("Cached waveform missing in " + path)
        if not getattr(args, "gpu_probe", False):
            report = {"gpu_probe": False, "effective_parallelism": args.max_train,
                      "memory_limit_GiB": None}
            (args.output / "gpu_plan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("[GPU_PLAN] short probes disabled; no memory budget limit; "
                  f"max_train={args.max_train}", flush=True)
            return pipeline.run(args, jobs=jobs)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["PYTHONUNBUFFERED"] = "1"
        peaks = []
        for name in ("m2", "m3"):
            for stage in ("public10", "public40"):
                log = args.output / f"probe_{name}_{stage}.log"
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                    "--probe-config", str(ROOT / "configs" / ("u7_" + name) / f"b7_{stage}.json")]
                if args.max_peak_gib is not None:
                    command += ["--max-peak-gib", str(args.max_peak_gib)]
                report = checked_subprocess(command, log, env)
                peaks.append(report["peak_reserved_GiB"])
        free = checked_subprocess([sys.executable, "-u", str(Path(__file__).resolve()),
                                   "--free-memory"], args.output / "gpu_free.log", env)["free_GiB"]
        requested = args.max_train
        args.max_train = choose_parallelism(peaks, free, requested)
        report = {"requested_parallelism": requested, "effective_parallelism": args.max_train,
                  "free_GiB": free, "probe_reserved_GiB": peaks}
        (args.output / "gpu_plan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[GPU_PLAN] " + json.dumps(report), flush=True)
        return pipeline.run(args, jobs=jobs)
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/home/qht/huawei/data"))
    p.add_argument("--metadata", type=Path, default=Path("/home/qht/huawei/metadata"))
    p.add_argument("--ptb-root", type=Path, default=Path("/home/qht/huawei/data/PTB-Diagnostic"))
    p.add_argument("--reuse-data-from", type=Path, default=Path("/home/qht/huawei_B7/results/B7/auto_seed42"))
    p.add_argument("--output", type=Path, default=Path("results/B7-M2-M3"))
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-train", type=int, choices=(0, 1, 2, 3, 4), default=0,
                   help="Maximum concurrent training stages; default 0 has no limit")
    p.add_argument("--gpu-probe", action="store_true", help="Optional GPU short tests; disabled by default")
    p.add_argument("--max-peak-gib", type=float, default=None, help="Optional probe limit; no limit by default")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--probe-config", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--free-memory", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.max_peak_gib is not None and args.max_peak_gib <= 0:
        p.error("max-peak-gib must be positive")
    if args.probe_config:
        from scripts.run_b7_large import probe
        probe(args.probe_config, args.max_peak_gib)
    elif args.free_memory:
        import torch
        print(json.dumps({"free_GiB": torch.cuda.mem_get_info()[0] / 2**30}), flush=True)
    else:
        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
