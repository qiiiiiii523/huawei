"""Independent M4/M5 experiments: 20000-step P, final EMA validation only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.B7 import pipeline


def suite_jobs(args):
    jobs = []
    for candidate in args.models:
        child = copy.copy(args)
        child.output = args.output.resolve() / (candidate + "_seed42")
        child.config_dir = ROOT / "configs" / ("u7_" + candidate.lower())
        gpu = str(getattr(args, "gpu_" + candidate.lower()))
        child.gpu_p = child.gpu_a = child.gpu_b = gpu
        for job in pipeline.plan(child):
            job.name = candidate + "_" + job.name
            job.dependencies = tuple(candidate + "_" + dep for dep in job.dependencies)
            jobs.append(job)
    priority = {"P": 0, "A": 1, "Q": 2, "B": 3}
    return sorted(jobs, key=lambda j: priority[j.name.split("_")[1]])


def run(args):
    args.output = args.output.resolve()
    args.config_dir = ROOT / "configs"
    args.experiment_suite = True
    args.retry_preparation = False
    args.gpu_is_runtime = True
    args.recover_initial = True
    jobs = suite_jobs(args)
    # Validate the experiment contract before allocating a run directory.
    from baselines.B7.config import ModelConfig
    from baselines.B7.train import validate_settings
    for candidate in args.models:
        settings = []
        for stage in ("public10", "public40", "huawei40"):
            path = ROOT / "configs" / ("u7_" + candidate.lower()) / f"b7_{stage}.json"
            cfg = json.loads(path.read_text(encoding="utf-8"))
            validate_settings(cfg)
            if cfg["training"]["validation_mode"] != "final_only":
                raise ValueError("M4/M5 require endpoint-only full validation")
            settings.append(ModelConfig.from_dict(cfg["model"]))
        if any(mc != settings[0] for mc in settings):
            raise ValueError(f"{candidate}: P/Q/A/B must share one architecture")
    if not args.dry_run:
        print("[M45] " + json.dumps({"models": args.models,
            "gpu": {m: str(getattr(args, "gpu_" + m.lower())) for m in args.models},
            "max_train": args.max_train, "max_per_gpu": args.max_per_gpu,
            "gpu_probe": False, "memory_limit_GiB": None,
            "selection": "final_ema"}), flush=True)
    return pipeline.run(args, jobs=jobs)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=("M4", "M5"), default=["M4", "M5"])
    p.add_argument("--gpu-m4", help="Physical GPU for all M4 stages")
    p.add_argument("--gpu-m5", help="Physical GPU for all M5 stages")
    p.add_argument("--data-root", type=Path, default=Path("/home/qht/huawei/data"))
    p.add_argument("--metadata", type=Path, default=Path("/home/qht/huawei/metadata"))
    p.add_argument("--ptb-root", type=Path, default=Path("/home/qht/huawei/data/PTB-Diagnostic"))
    p.add_argument("--reuse-data-from", type=Path, default=Path("/home/qht/huawei_B7/results/B7/auto_seed42"))
    p.add_argument("--output", type=Path, default=Path("results/B7-M4-M5"))
    p.add_argument("--max-train", type=int, choices=range(5), default=2,
                   help="Suite-wide training limit; 0 disables it")
    p.add_argument("--max-per-gpu", type=int, choices=range(5), default=1,
                   help="Per-card training limit within this suite; 0 disables it")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if len(set(args.models)) != len(args.models):
        p.error("models must be unique")
    for model in args.models:
        gpu = getattr(args, "gpu_" + model.lower())
        if gpu is None or not gpu.isascii() or not gpu.isdigit() or str(int(gpu)) != gpu:
            p.error(f"--gpu-{model.lower()} must specify one physical GPU index")
    return args


def main():
    args = parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
