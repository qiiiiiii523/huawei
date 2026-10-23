"""Sequential, resumable experiment suite; no polling or automatic promotion."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from baselines.B5_best.common import read, write, sha, csv_write
from baselines.B5_slowrate import RATES
from baselines.B5_slowrate.common import load_study as load_rates
from baselines.B5_slow120.common import load_study as load_original
from . import GROUPS
from .common import code_sha, load_study


PRESETS = {
    "rate-fusion": {"groups": ("CONT", "FUSION_SAME", "FUSION"), "stages": "all"},
    "cnn-rf": {"groups": ("CONT", "PTAIL", "PCTX", "PCTX_RF"), "stages": "rates,followup"},
    "all": {"groups": GROUPS, "stages": "all"},
}


def resolve_selection(args):
    """Resolve one route before reading data or creating any output."""
    preset = getattr(args, "preset", None)
    explicit = getattr(args, "groups", None)
    if preset is not None and explicit is not None:
        raise ValueError("Choose --preset OR --groups; do not silently override either")
    if explicit is None:
        preset = preset or "rate-fusion"
        args.groups = ",".join(PRESETS[preset]["groups"])
    groups = args.groups.split(",")
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError("Unknown/duplicate follow-up groups")
    if args.stages is None:
        args.stages = PRESETS[preset]["stages"] if preset else "rates"
    args.preset = preset or "custom"
    return args


def describe_plan(args):
    """No model loading, data reads, directory creation or optimizer updates."""
    stages = ["rates", "fixed", "followup"] if args.stages == "all" else args.stages.split(",")
    if not stages or len(set(stages)) != len(stages) or set(stages) - {"rates", "fixed", "followup"}:
        raise ValueError("Unknown/duplicate stages")
    return {"preset": args.preset, "stages": stages,
            "rate_groups": list(RATES) if "rates" in stages else [],
            "followup_groups": args.groups.split(",") if "followup" in stages else [],
            "seeds": [int(s) for s in args.seeds.split(",")], "run": str(Path(args.run).resolve()),
            "training_started": False,
            "cnn_followup_requires_completed_rate_sources": "followup" in stages}


class Runner:
    def __init__(self, root, state):
        self.root, self.state = Path(root).resolve(), state

    def save(self):
        write(self.root / "suite.tmp.json", self.state)
        (self.root / "suite.tmp.json").replace(self.root / "suite.json")

    def retain_partial(self, path):
        path = Path(path).resolve()
        if path == self.root or not path.is_relative_to(self.root):
            raise ValueError("Refuse to move a directory outside this experiment")
        if path.exists():
            saved = path.with_name(path.name + "_interrupted_" + str(time.time_ns()))
            path.rename(saved)
            print("Incomplete output retained: " + str(saved), flush=True)

    def step(self, name, module, argv, marker, output=None, training=False, bound_files=()):
        argv = [str(v) for v in argv]
        signature = hashlib.sha256(json.dumps([module, argv, [sha(p) for p in bound_files]],
                                             sort_keys=True).encode()).hexdigest()
        old = self.state["steps"].get(name, {})
        marker = Path(marker)
        if old.get("signature") and old["signature"] != signature:
            raise ValueError("Step command or bound input changed; use a new suite: " + name)
        if old.get("status") == "completed" and old.get("signature") == signature:
            if marker.is_file() and read(marker).get("completed"):
                if old.get("marker_sha256") != sha(marker):
                    raise ValueError("Completed report changed: " + str(marker))
                print("Already complete: " + name, flush=True)
                return
        actual = list(argv)
        if output is not None and Path(output).exists():
            if training and (Path(output) / "last.pt").is_file():
                actual += ["--resume", str(Path(output) / "last.pt")]
            else:
                self.retain_partial(output)
        record = {"status": "running", "signature": signature, "module": module,
                  "argv": actual, "started_unix": time.time()}
        self.state["steps"][name] = record
        self.state.update(status="running", current_step=name, completed=False)
        self.save()
        log = self.root / "logs" / (name + ".log")
        log.parent.mkdir(exist_ok=True)
        print("Starting: " + name, flush=True)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        with log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen([sys.executable, "-u", "-m", module, *actual],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
            record["pid"] = process.pid
            self.save()
            try:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end="", flush=True)
                result = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                record["status"] = "interrupted"
                self.state["status"] = "interrupted"
                self.save()
                raise
        record.update(exit_code=result, ended_unix=time.time())
        if result or not marker.is_file() or not read(marker).get("completed"):
            record["status"] = "failed"
            self.state["status"] = "failed"
            self.save()
            raise RuntimeError(f"Step failed: {name}. Log: {log}; resume the same suite with --resume")
        record["status"] = "completed"
        record["marker_sha256"] = sha(marker)
        self.save()


def suite_report(root):
    rows = []
    for seed_root in sorted(Path(root).glob("seed*")):
        for stage, pattern in (("rates", "comparison/summary.json"), ("followup", "followup/comparison/summary.json")):
            path = seed_root / pattern
            if path.is_file():
                for row in read(path)["summary"]:
                    rows.append({"seed_run": seed_root.name, "stage": stage, "candidate": row["candidate"],
                                 "epoch": row["epoch"], "mean_r": row["mean_r"]})
        for path in sorted((seed_root / "fixed").glob("*/ensemble.json")):
            row = read(path)["summary"]
            rows.append({"seed_run": seed_root.name, "stage": "fixed", "candidate": path.parent.name,
                         "epoch": 0, "mean_r": row["mean_r"]})
    if rows:
        csv_write(Path(root) / "suite_summary.csv", rows)
    lines = ["# 采样率 / 融合 / P-S50与S100 实验汇总", "",
             "|seed目录|阶段|候选|epoch|平均r|", "|---|---|---|---:|---:|"]
    lines += [f"|{r['seed_run']}|{r['stage']}|{r['candidate']}|{r['epoch']}|{r['mean_r']:.8f}|" for r in rows]
    lines += ["", "各seed的comparison与followup/comparison提供完整任务分数和匹配差值。",
              "固定融合是等权修正输出组合，context只应用一次；未按验证患者拟合权重。",
              "PCTX/PCTX_RF的初始分数可能不同；需同时看initial与S50/长窗口控制。",
              "这些验证候选不自动替换当前最好模型。"]
    (Path(root) / "suite_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    root, source = Path(args.run).resolve(), Path(args.source_shared).resolve()
    load_original(source)
    seeds = [int(s) for s in args.seeds.split(",")]
    stages = ("rates", "fixed", "followup") if args.stages == "all" else tuple(args.stages.split(","))
    groups = args.groups.split(",")
    if (not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups)
            or min(args.rate_epochs, args.followup_epochs, args.rate_validate_every, args.followup_validate_every,
                   args.rate_batch_records, args.followup_batch_records) < 1
            or any(not math.isfinite(v) or v <= 0 for v in (args.learning_rate, args.fusion_learning_rate))
            or not math.isfinite(args.p_preservation) or args.p_preservation < 0
            or args.p_extra_dilation < 1 or (2045 + 8 * args.p_extra_dilation) / 50 + 10 > args.p_context_seconds):
        raise ValueError("Invalid groups, budgets, learning rates or P context/RF combination")
    if (not stages or any(s not in ("rates", "fixed", "followup") for s in stages)
            or len(set(stages)) != len(stages) or len(set(seeds)) != len(seeds)
            or any(not 0 <= s < 2**32 for s in seeds)):
        raise ValueError("Need distinct valid seeds/stages")
    # All settings bind resume. Stages may be added later, settings may not change.
    intent = {k: v for k, v in vars(args).items()
              if k not in ("resume", "stages", "execute_training", "preset", "plan")}
    intent.update(source_shared=str(source), run=str(root), source_sha256=sha(source / "study.json"), code_sha256=code_sha())
    if root.exists():
        if not args.resume or not (root / "suite.json").is_file():
            raise FileExistsError("Use a new suite directory or --resume for its existing suite.json")
        state = read(root / "suite.json")
        if state["intent"] != intent:
            raise ValueError("Resume changed code, source, seeds or budgets; create a new suite")
    else:
        root.mkdir(parents=True)
        state = {"intent": intent, "steps": {}, "status": "starting"}
    lock = root / "active.lock"
    if lock.exists():
        pid = read(lock)["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError(f"Suite already active (PID {pid}); do not launch a duplicate")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid()}, handle)
    runner = Runner(root, state)
    runner.save()
    try:
        for seed in seeds:
            rate = root / f"seed{seed}"
            shared = rate / "shared"
            if "rates" in stages:
                runner.step(f"s{seed}_rates_prepare", "baselines.B5_slowrate.prepare",
                    ["--source-shared", source, "--output", shared, "--training-seed", seed,
                     "--epochs", args.rate_epochs, "--validate-every", args.rate_validate_every,
                     "--batch-records", args.rate_batch_records], shared / "study.json", output=shared)
                load_rates(shared)
                for group in RATES:
                    out = rate / group
                    runner.step(f"s{seed}_{group}_train", "baselines.B5_slowrate.train",
                        ["--run", shared, "--group", group, "--output", out, "--device", args.device, "--execute-training"],
                        out / "completed.json", output=out, training=True, bound_files=[shared / "study.json"])
                    for name in ("best", "last"):
                        verify = rate / "verification" / group / name
                        runner.step(f"s{seed}_{group}_{name}", "baselines.B5_slowrate.evaluate",
                            ["--run", shared, "--checkpoint", out / (name + ".pt"), "--output", verify, "--device", args.device],
                            verify / "evaluation.json", output=verify, bound_files=[out / (name + ".pt")])
                runner.step(f"s{seed}_rates_summary", "baselines.B5_slowrate.summarize",
                    ["--run", rate, "--output", rate / "comparison"], rate / "comparison/summary.json", output=rate / "comparison")
            elif not shared.is_dir():
                raise ValueError("Run rates first, then add fixed/followup with --resume")
            load_rates(shared)
            if "fixed" in stages:
                for groups in (("R2p5", "R5"), ("R5", "R10"), ("R2p5", "R10"), tuple(RATES)):
                    name = "_".join(groups)
                    out = rate / "fixed" / name
                    runner.step(f"s{seed}_fixed_{name}", "baselines.B5_slowrate_followup.ensemble",
                        ["--rate-runs", rate, "--groups", *groups, "--output", out, "--device", args.device],
                        out / "ensemble.json", output=out, bound_files=[rate / g / "best.pt" for g in groups])
            if "followup" in stages:
                follow = rate / "followup"
                fs = follow / "shared"
                runner.step(f"s{seed}_followup_prepare", "baselines.B5_slowrate_followup.prepare",
                    ["--rate-run", rate, "--output", fs, "--groups", args.groups, "--reference-group", args.reference_group,
                     "--epochs", args.followup_epochs, "--validate-every", args.followup_validate_every,
                     "--batch-records", args.followup_batch_records, "--learning-rate", args.learning_rate,
                     "--fusion-learning-rate", args.fusion_learning_rate, "--tail-blocks", args.tail_blocks,
                     "--p-context-seconds", args.p_context_seconds, "--p-extra-dilation", args.p_extra_dilation,
                     "--p-preservation", args.p_preservation], fs / "study.json", output=fs,
                     bound_files=[rate / g / "best.pt" for g in RATES])
                load_study(fs)
                for group in args.groups.split(","):
                    out = follow / group
                    runner.step(f"s{seed}_{group}_followup_train", "baselines.B5_slowrate_followup.train",
                        ["--run", fs, "--group", group, "--output", out, "--device", args.device, "--execute-training"],
                        out / "completed.json", output=out, training=True, bound_files=[fs / "study.json"])
                    for name in ("best", "last"):
                        verify = follow / "verification" / group / name
                        runner.step(f"s{seed}_{group}_followup_{name}", "baselines.B5_slowrate_followup.evaluate",
                            ["--run", fs, "--checkpoint", out / (name + ".pt"), "--output", verify, "--device", args.device],
                            verify / "evaluation.json", output=verify, bound_files=[out / (name + ".pt")])
                runner.step(f"s{seed}_followup_summary", "baselines.B5_slowrate_followup.summarize",
                    ["--run", follow, "--output", follow / "comparison"], follow / "comparison/summary.json", output=follow / "comparison")
        suite_report(root)
        state.update(status="completed", current_step=None, completed=True)
        runner.save()
        print("Completed. Results: " + str(root / "suite_summary.md"), flush=True)
    except BaseException as error:
        # Includes setup/data/subprocess-start failures outside Runner.step's
        # stream loop, so suite.json never remains misleadingly "running".
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     completed=False, error=str(error))
        runner.save()
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-shared", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--seeds", default="43")
    p.add_argument("--preset", choices=tuple(PRESETS), help="Default rate-fusion; cnn-rf selects the CNN controls")
    p.add_argument("--stages", help="rates,fixed,followup or all; can add stages with --resume")
    p.add_argument("--groups", help="Custom groups instead of --preset; default custom stage is rates")
    p.add_argument("--reference-group", choices=("best", *RATES), default="best")
    p.add_argument("--rate-epochs", type=int, default=40)
    p.add_argument("--rate-validate-every", type=int, default=5)
    p.add_argument("--rate-batch-records", type=int, default=2)
    p.add_argument("--followup-epochs", type=int, default=10)
    p.add_argument("--followup-validate-every", type=int, default=1)
    p.add_argument("--followup-batch-records", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--fusion-learning-rate", type=float, default=1e-4)
    p.add_argument("--tail-blocks", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--p-context-seconds", type=int, choices=(60, 80, 100, 120), default=100)
    p.add_argument("--p-extra-dilation", type=int, default=256)
    p.add_argument("--p-preservation", type=float, default=.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute-training", action="store_true")
    p.add_argument("--plan", action="store_true", help="Print resolved route only; no data/files/training")
    args = p.parse_args()
    try:
        resolve_selection(args)
        description = describe_plan(args)
    except ValueError as error:
        p.error(str(error))
    if args.plan:
        print(json.dumps(description, ensure_ascii=False, indent=2))
        return
    if not args.execute_training:
        p.error("No work started; add --execute-training explicitly")
    print(json.dumps(description, ensure_ascii=False), flush=True)
    run(args)


if __name__ == "__main__":
    main()
