"""Recover only a blocked B7 PTB/Q/B route without editing the live B7 package."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def canonical_index(root):
    from baselines.B7.prepare import public_index
    root = Path(root).resolve()
    headers = sorted(root.rglob("*.hea"))
    retained, excluded = [], []
    for path in headers:
        relative = path.relative_to(root)
        if len(relative.parts) == 2 and re.fullmatch(r"patient\d+", relative.parts[0]):
            retained.append(path.resolve())
        elif len(relative.parts) == 1:
            excluded.append(str(relative))
        else:
            raise ValueError(f"Unexpected nested PTB header; review rather than guess: {relative}")
    if not retained:
        raise ValueError("No PTB records found in patient-number directories")
    wanted = set(retained)
    rows = [row for row in public_index("ptbdb", root) if Path(row["path"]).resolve() in wanted]
    for row in rows:
        path = Path(row["path"])
        # Unique physical ID includes the authoritative patient directory.
        row["record_id"] = row["target_record_id"] = f"ptbdb:{path.parent.name}/{path.stem}"
    if len(rows) != len(retained) or len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Canonical PTB records have missing or duplicate identities")
    if not {"train", "validation"}.issubset({r["split"] for r in rows}):
        raise ValueError("PTB index needs train and validation patients")
    return rows, {"headers": len(headers), "retained": len(rows), "excluded_root_headers": excluded,
                  "policy": "only-direct-patient-number-records; root files excluded, not deleted",
                  "splits": dict(Counter(r["split"] for r in rows)), "raw_data_modified": False}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def wait_for(path, output, stage, timeout):
    started = time.monotonic()
    while not path.is_file():
        status_path = output / "pipeline_status.json"
        if status_path.is_file():
            state = json.loads(status_path.read_text())["jobs"].get(stage)
            if state in ("failed", "blocked"):
                raise RuntimeError(f"Original {stage} is {state}; inspect its log, no dependent training started")
        if time.monotonic() - started >= timeout:
            raise TimeoutError(f"Waited too long for {stage}: {path}")
        print(f"[WAIT] {stage}: {path}", flush=True)
        time.sleep(30)


def run(args):
    from baselines.B7.pipeline import Job, acquire_lock, command_for, validate_artifact
    from baselines.B7.data import read_manifest
    os.chdir(ROOT)
    output = args.output.resolve()
    status_path = output / "pipeline_status.json"
    original = json.loads(status_path.read_text())["jobs"]
    if any(original.get(name) != "blocked" for name in ("Q", "B")):
        raise RuntimeError("Use only when original Q and B are blocked; do not race another scheduler")
    recovery = output / "ptb_recovery"
    recovery.mkdir(exist_ok=True)
    lock = acquire_lock(recovery / ".lock")
    state = {"jobs": {}, "training_results": {}, "success": False}
    def save():
        write_json(recovery / "status.json", state)
    def execute(job):
        state["jobs"][job.name] = "running"
        save()
        print(f"[START] {job.name}; log: {recovery / (job.name + '.log')}", flush=True)
        if not job.artifact.exists():
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpu if job.module == "train" else ""
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("OMP_NUM_THREADS", "4")
            with (recovery / (job.name + ".log")).open("a", encoding="utf-8") as log:
                subprocess.run(command_for(job), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = validate_artifact(job)
        if job.module == "train":
            state["training_results"][job.name] = result
        state["jobs"][job.name] = "done"
        save()
        print(f"[DONE] {job.name}", flush=True)
    def training(name, manifest, config, parent):
        return Job(name, (), "train", ["--manifest", str(manifest), "--config", config,
                   "--output", str(output / name), "--device", "cuda",
                   "--init-checkpoint", str(output / parent / "best_r.pt")], output / name / "completed.json", args.gpu)
    try:
        public = output / "data/public10/manifest.json"
        read_manifest(public)
        rows, audit = canonical_index(args.ptb_root)
        index = recovery / "index.json"
        if index.exists() and json.loads(index.read_text(encoding="utf-8")) != rows:
            raise RuntimeError("Recovery index changed; keep original inputs unchanged")
        write_json(index, rows)
        write_json(recovery / "index_audit.json", audit)
        print(json.dumps(audit, ensure_ascii=False), flush=True)
        manifest = recovery / "data/manifest.json"
        execute(Job("prepare_ptb", (), "prepare", ["--index", str(index), "--scale-manifest", str(public),
                    "--output", str(manifest.parent), "--resume"], manifest))
        if read_manifest(manifest)["scales_uV"] != read_manifest(public)["scales_uV"]:
            raise RuntimeError("PTB scales must match P")
        wait_for(output / "P/completed.json", output, "P", args.wait_seconds)
        parent_job = training("P", public, "configs/b7_public10.json", "P")
        state["training_results"]["P"] = validate_artifact(parent_job)
        execute(training("Q", manifest, "configs/b7_public40.json", "P"))
        huawei = output / "data/huawei/manifest.json"
        wait_for(huawei, output, "prepare_huawei", args.wait_seconds)
        execute(training("B", huawei, "configs/b7_huawei40.json", "Q"))
        state["success"] = True
        save()
        return 0
    except BaseException as exc:
        for name, value in list(state["jobs"].items()):
            if value == "running":
                state["jobs"][name] = "failed"
        state["error"] = str(exc)
        save()
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/B7/auto_seed42"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--wait-seconds", type=int, default=30 * 24 * 3600)
    args = parser.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
