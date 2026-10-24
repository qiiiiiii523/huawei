"""Copy a stopped B7-T batch32 run into a new run with sparse validation.

Only scheduling metadata is upgraded. Original checkpoints and caches stay intact.
Run after updating to the committed, tested B7-T code and stopping the old controller.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import torch

from .checkpoint import file_sha256, load_checkpoint, model_from_checkpoint, save_checkpoint
from .config import LossConfig, ModelConfig, SamplingConfig, read_json
from .data import read_manifest
from .train import code_sha, validate_settings

ROOT = Path(__file__).resolve().parents[2]
BASELINE_REVISION = "807b8b074f8dcb1f0439f27d0eb91541678ed463"
BASELINE_CODE_SHA = "964aae60c2f2f4b4d2de131876035d1961d8b9c2ed3ea1f3f0fd36f6842d6247"
STAGES = {"P": "public10", "Q": "public40", "A": "huawei40", "B": "huawei40"}
INTERVALS = {"public10": 2000, "public40": 1000, "huawei40": 500}
CACHES = {"public10": "public10", "public40": "ptb40", "huawei40": "huawei"}


def git(*arguments):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments], cwd=ROOT)


def package_at_revision(revision):
    paths = git("ls-tree", "-r", "--name-only", revision, "--", "baselines/B7").decode().splitlines()
    return {Path(path).name: git("show", f"{revision}:{path}").replace(b"\r\n", b"\n")
            for path in paths if Path(path).parent.as_posix() == "baselines/B7" and path.endswith(".py")}


def package_hash(files):
    h = hashlib.sha256()
    for name, value in sorted(files.items()):
        h.update(name.encode())
        h.update(value)
    return h.hexdigest()


def working_package():
    return {path.name: path.read_bytes().replace(b"\r\n", b"\n")
            for path in Path(__file__).parent.glob("*.py")}


def verify_code_upgrade():
    original = package_at_revision(BASELINE_REVISION)
    if package_hash(original) != BASELINE_CODE_SHA:
        raise ValueError("Unexpected baseline code; cannot verify migration")
    current = working_package()
    if package_at_revision("HEAD") != current:
        raise ValueError("Current B7-T code must be the committed tested version")
    changed = {name for name in set(current) | set(original) if current.get(name) != original.get(name)}
    if changed - {"train.py", "pipeline.py", "reschedule.py"}:
        raise ValueError(f"Model/data/loss/sampling/scoring code changed: {sorted(changed)}")


def without_schedule(config):
    value = copy.deepcopy(config)
    for key in ("validate_every", "checkpoint_every"):
        value["training"].pop(key, None)
    return value


def baseline_config(stage):
    return json.loads(git("show", f"{BASELINE_REVISION}:configs/b7_{stage}.json"))


def stage_configs():
    configs = {}
    for stage, interval in INTERVALS.items():
        cfg = read_json(ROOT / "configs" / f"b7_{stage}.json")
        validate_settings(cfg)
        if (without_schedule(cfg) != without_schedule(baseline_config(stage))
                or cfg["training"]["validate_every"] != interval
                or cfg["training"].get("checkpoint_every") != 500):
            raise ValueError(f"Unexpected {stage} config; only validation/checkpoint intervals may change")
        configs[stage] = cfg
    return configs


def assert_stopped(source):
    """Catch both the controller and independently launched stage processes on Linux."""
    if not Path("/proc").is_dir():
        return
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            parts = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any(module in parts for module in ("baselines.B7.train", "baselines.B7.pipeline")):
                continue
            position = parts.index("--output")
            value = Path(parts[position + 1])
            if not value.is_absolute():
                value = (entry / "cwd").resolve() / value
            if value.resolve() == source or value.resolve().parent == source:
                raise RuntimeError(f"Old process {entry.name} is still running; stop the old tmux controller first")
        except (OSError, ValueError):
            continue


def cache_snapshot(path):
    value = read_manifest(path)
    return {"manifest": str(path.resolve()), "manifest_sha256": file_sha256(path),
            "record_sha256": {row["record_id"]: file_sha256(row["resolved_path"]) for row in value["records"]},
            "scales_uV": value["scales_uV"]}


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
            raise ValueError("Best aliases disagree")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("Best aliases disagree")
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("Best aliases disagree")
        for x, y in zip(a, b):
            assert_same(x, y)
    elif a != b:
        raise ValueError("Best aliases disagree")


def inspect_stage(directory, stage, cfg, cache):
    last = load_checkpoint(directory / "last.pt")
    best = load_checkpoint(directory / "best_r.pt")
    identity = last["identity"]
    saved_cfg = identity["config"]
    if (identity["code_sha256"] != BASELINE_CODE_SHA or read_json(directory / "config.json") != saved_cfg
            or without_schedule(saved_cfg) != without_schedule(cfg)):
        raise ValueError(f"{directory.name}: unknown source code or changed non-schedule config")
    if (identity["manifest_sha256"] != cache["manifest_sha256"]
            or identity["record_sha256"] != cache["record_sha256"]
            or list(last["scales_uV"]) != cache["scales_uV"]):
        raise ValueError(f"{directory.name}: data/manifest/scales changed")
    if (last["stage"] != stage or ModelConfig.from_dict(last["model_config"]) != ModelConfig.from_dict(cfg["model"])
            or last["loss"] != asdict(LossConfig(**cfg["loss"]))
            or last["sampling"] != asdict(SamplingConfig(**cfg["sampling"]))):
        raise ValueError("Checkpoint model/loss/sampling differs from config")
    if any(key not in last for key in ("optimizer", "rng", "model", "ema", "initialization")):
        raise ValueError("Incomplete recovery checkpoint")
    if (type(last["update"]) is not int or not 0 <= last["update"] <= cfg["training"]["updates"]
            or not math.isfinite(last["best_r"]) or best["identity"] != identity
            or best["best_r"] != last["best_r"] or not 0 <= best["update"] <= last["update"]):
        raise ValueError("Best and last lineage/score/step disagree")
    if last.get("pending_validation", False):
        raise ValueError("Expected a fully validated legacy checkpoint")
    checked = model_from_checkpoint(last)
    checked.load_state_dict(last["model"], strict=True)
    optimizer = torch.optim.AdamW(checked.parameters())
    optimizer.load_state_dict(last["optimizer"])
    if last["update"] > 0 and not optimizer.state:
        raise ValueError("Missing AdamW recovery state")
    for parameter, state in optimizer.state.items():
        for name in ("exp_avg", "exp_avg_sq"):
            if name not in state or state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                raise ValueError("Invalid AdamW recovery state")
    model_from_checkpoint(best)
    if (directory / "best.pt").exists():
        assert_same(load_checkpoint(directory / "best.pt"), best)
    completed = directory / "completed.json"
    if completed.exists():
        value = read_json(completed)
        if (value.get("completed") is not True or value.get("update") != last["update"]
                or last["update"] != cfg["training"]["updates"] or value.get("best_r") != last["best_r"]):
            raise ValueError("Completed marker and recovery checkpoint disagree")
    return {"update": last["update"], "best_r": last["best_r"], "completed": completed.exists(),
            "identity": identity, "checkpoint_sha256": {name: file_sha256(directory / name)
                for name in ("last.pt", "best_r.pt", "best.pt") if (directory / name).exists()}}


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def migrate(source, output, *, check_only=False):
    from .pipeline import acquire_lock
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.exists() or source == output or source in output.parents:
        raise ValueError("Need an existing source and a NEW output outside the original run")
    if not (source / ".pipeline.lock").is_file():
        raise ValueError("Source must be an existing pipeline run with its lock file")
    assert_stopped(source)
    lock = acquire_lock(source / ".pipeline.lock")
    try:
        verify_code_upgrade()
        configs = stage_configs()
        caches = {stage: cache_snapshot(source / "data" / name / "manifest.json") for stage, name in CACHES.items()}
        if any(value["scales_uV"] != caches["public10"]["scales_uV"] for value in caches.values()):
            raise ValueError("Cached stage scales disagree")
        stages = {}
        for name, stage in STAGES.items():
            directory = source / name
            if directory.exists():
                stages[name] = inspect_stage(directory, stage, configs[stage], caches[stage])
        if "P" not in stages:
            raise ValueError("No recoverable P checkpoint")
        for name, parent in (("Q", "P"), ("A", "P"), ("B", "Q")):
            if name in stages and (parent not in stages or not stages[parent]["completed"]):
                raise ValueError("Started stage has no completed parent")
        record = {"kind": "B7-T-sparse-validation-v1", "source": str(source), "source_revision": BASELINE_REVISION,
                  "target_code_sha256": code_sha(), "configs": configs, "caches": caches,
                  "stages": stages, "complete": False}
        if check_only:
            print(json.dumps({"status": "CHECKED", "stages": {k: v["update"] for k, v in stages.items()}}), flush=True)
            return record
        output.mkdir(parents=True)
        write_json(output / "migration.json", record)
        for name, inspected in stages.items():
            original, target = source / name, output / name
            target.mkdir()
            cfg = configs[STAGES[name]]
            identity = dict(inspected["identity"], config=cfg, code_sha256=code_sha())
            for checkpoint in ("last.pt", "best_r.pt", "best.pt"):
                original_name = checkpoint if checkpoint in inspected["checkpoint_sha256"] else "best_r.pt"
                if file_sha256(original / original_name) != inspected["checkpoint_sha256"][original_name]:
                    raise ValueError("Source checkpoint changed during migration")
                value = load_checkpoint(original / original_name)
                value["identity"] = identity
                value["schedule_migration"] = {"source": str(original / original_name),
                    "sha256": inspected["checkpoint_sha256"][original_name], "source_revision": BASELINE_REVISION}
                if checkpoint == "last.pt":
                    value.update(last_validated_update=value["update"], pending_validation=False, pending_train_parts=None)
                save_checkpoint(target / checkpoint, value)
            write_json(target / "config.json", cfg)
            for path in original.iterdir():
                if path.name in ("history.jsonl", "checkpoint_history.jsonl", "completed.json") or path.name.startswith("validation_"):
                    if path.is_file() and path.suffix in (".json", ".jsonl"):
                        shutil.copy2(path, target / path.name)
        record["complete"] = True
        write_json(output / "migration.json", record)
        print(json.dumps({"status": "MIGRATED", "output": str(output),
                          "stages": {k: v["update"] for k, v in stages.items()},
                          "original_checkpoints_untouched": True}), flush=True)
        return record
    finally:
        lock.close()


def validate_migration(output, reuse_data_from):
    """Keep the ordinary strict training/pipeline checks after the one-time copy."""
    record = read_json(Path(output) / "migration.json")
    if (record.get("kind") != "B7-T-sparse-validation-v1" or record.get("complete") is not True
            or record["target_code_sha256"] != code_sha() or reuse_data_from is None
            or Path(record["source"]).resolve() != Path(reuse_data_from).resolve()):
        raise ValueError("Incomplete migration or changed code/cache source")
    if record["configs"] != stage_configs():
        raise ValueError("Migrated stage configs changed")
    for cache in record["caches"].values():
        if cache_snapshot(Path(cache["manifest"])) != cache:
            raise ValueError("Migrated cache/manifest changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Stopped original pipeline output")
    parser.add_argument("--output", type=Path, required=True, help="New sibling pipeline output")
    parser.add_argument("--check-only", action="store_true", help="Verify without creating the new run")
    args = parser.parse_args()
    migrate(args.source, args.output, check_only=args.check_only)


if __name__ == "__main__":
    main()
