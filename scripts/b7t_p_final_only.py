"""Continue an already migrated B7-T P, evaluating only at its final update.

This standalone driver leaves the model package and shared configs untouched.
Use this same driver's `run` subcommand for subsequent restarts of the new run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.B7 import pipeline
from baselines.B7.checkpoint import file_sha256, load_checkpoint, model_from_checkpoint, save_checkpoint
from baselines.B7.config import read_json
from baselines.B7.reschedule import assert_same, assert_stopped, cache_snapshot, write_json
from baselines.B7.train import code_sha, validate_settings

REPORT = "p_final_only.json"
STAGES = {"P": "public10", "A": "huawei40", "Q": "public40", "B": "huawei40"}


def project_configs():
    return {stage: read_json(ROOT / "configs" / f"b7_{stage}.json")
            for stage in set(STAGES.values())}


def migrate(source, output, *, skip_pending=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.exists() or source == output or source in output.parents:
        raise ValueError("Use an existing source and a NEW output outside the source")
    assert_stopped(source)
    if not (source / ".pipeline.lock").is_file():
        raise ValueError("Source is not a pipeline run")
    lock = pipeline.acquire_lock(source / ".pipeline.lock")
    try:
        if any((source / name).exists() for name in ("A", "Q", "B")):
            raise ValueError("This tool requires P unfinished and A/Q/B not started")
        provenance = read_json(source / "migration.json")
        if (provenance.get("kind") != "B7-T-sparse-validation-v1"
                or provenance.get("complete") is not True
                or provenance["target_code_sha256"] != code_sha()):
            raise ValueError("Source is not the verified current sparse B7-T run")
        configs = provenance["configs"]
        if configs != project_configs():
            raise ValueError("Shared configs changed; restore the original sparse settings first")
        cache_root = Path(provenance["source"]).resolve()
        for stage, cache in provenance["caches"].items():
            expected = cache_root / "data" / {"public10": "public10", "public40": "ptb40", "huawei40": "huawei"}[stage] / "manifest.json"
            if Path(cache["manifest"]).resolve() != expected or cache_snapshot(expected) != cache:
                raise ValueError("Prepared cache/manifest/source changed")
        old = source / "P"
        values = {name: load_checkpoint(old / name) for name in ("last.pt", "best_r.pt", "best.pt")}
        last, best = values["last.pt"], values["best_r.pt"]
        cfg = configs["public10"]
        identity = last["identity"]
        cache = provenance["caches"]["public10"]
        if (identity["code_sha256"] != code_sha() or identity["config"] != cfg
                or read_json(old / "config.json") != cfg
                or identity["manifest_sha256"] != cache["manifest_sha256"]
                or identity["record_sha256"] != cache["record_sha256"]
                or list(last["scales_uV"]) != cache["scales_uV"]):
            raise ValueError("P checkpoint identity/config/data changed")
        if (last["stage"] != "public10" or not 0 < last["update"] < cfg["training"]["updates"]
                or (old / "completed.json").exists()):
            raise ValueError("P must be unfinished; final validation cannot be skipped")
        pending = last.get("pending_validation", False)
        if pending and not skip_pending:
            raise ValueError("Pending validation requires explicit --skip-pending-validation")
        if any(key not in last for key in ("model", "ema", "optimizer", "rng", "initialization")):
            raise ValueError("Missing full recovery state")
        if (best["identity"] != identity or best["best_r"] != last["best_r"]
                or not math.isfinite(last["best_r"])
                or not 0 <= best["update"] <= last["last_validated_update"] <= last["update"]):
            raise ValueError("Best/last inconsistent; validation may have been interrupted during checkpoint commit")
        for key in ("identity", "update", "best_r", "model", "ema", "optimizer", "rng", "initialization"):
            assert_same(values["best.pt"][key], best[key])
        checked = model_from_checkpoint(last)
        checked.load_state_dict(last["model"], strict=True)
        model_from_checkpoint(best)
        new_configs = copy.deepcopy(configs)
        new_cfg = new_configs["public10"]
        new_cfg["training"]["validate_every"] = new_cfg["training"]["updates"]
        for value in new_configs.values():
            validate_settings(value)
        hashes = {name: file_sha256(old / name) for name in values}
        record = {"kind": "B7-T-P-final-only-v1", "complete": False,
                  "source": str(source), "cache_root": str(cache_root), "code_sha256": code_sha(),
                  "tool_sha256": file_sha256(__file__), "configs": new_configs,
                  "caches": provenance["caches"], "source_checkpoint_sha256": hashes,
                  "resume_update": last["update"], "last_validated_update": last["last_validated_update"],
                  "skipped_validation": last["update"] if pending else None,
                  "best_r": last["best_r"]}
        output.mkdir(parents=True)
        write_json(output / REPORT, record)
        (output / "configs").mkdir()
        for stage, value in new_configs.items():
            write_json(output / "configs" / f"b7_{stage}.json", value)
        target = output / "P"
        target.mkdir()
        new_identity = dict(identity, config=new_cfg)
        for name, value in values.items():
            if file_sha256(old / name) != hashes[name]:
                raise ValueError("Source checkpoint changed during copy")
            value["identity"] = new_identity
            value["final_only_reschedule"] = {"source": str(old / name), "sha256": hashes[name]}
            if name == "last.pt" and pending:
                # Skip evaluation, not training. Keep the genuinely validated step.
                value["pending_validation"] = False
                value["pending_train_parts"] = None
            save_checkpoint(target / name, value)
        write_json(target / "config.json", new_cfg)
        for path in old.iterdir():
            if path.is_file() and (path.suffix == ".jsonl" or path.name.startswith("validation_")):
                shutil.copy2(path, target / path.name)
        if pending:
            write_json(target / f"skipped_validation_{last['update']:07d}.json", {
                "update": last["update"], "reason": "explicit user request; no score computed",
                "last_validated_update": last["last_validated_update"]})
        record["complete"] = True
        write_json(output / REPORT, record)
        print(json.dumps({"status": "MIGRATED", "output": str(output),
                          "resume_update": record["resume_update"], "skipped_validation": record["skipped_validation"],
                          "last_validated_update": record["last_validated_update"], "next_validation": new_cfg["training"]["updates"]}), flush=True)
        return record
    finally:
        lock.close()


def run(args):
    output = args.output.resolve()
    record = read_json(output / REPORT)
    if (record.get("kind") != "B7-T-P-final-only-v1" or record.get("complete") is not True
            or record["code_sha256"] != code_sha() or record["tool_sha256"] != file_sha256(__file__)):
        raise ValueError("Incomplete reschedule or changed model/tool code")
    assert_stopped(Path(record["source"]))
    for stage, cfg in record["configs"].items():
        if read_json(output / "configs" / f"b7_{stage}.json") != cfg:
            raise ValueError("Run-specific configuration changed")
    for cache in record["caches"].values():
        if cache_snapshot(Path(cache["manifest"])) != cache:
            raise ValueError("Reused cache changed")
    args.reuse_data_from = Path(record["cache_root"])
    args.retry_preparation = False
    original_plan, original_fingerprint = pipeline.plan, pipeline.fingerprint

    def planned(value):
        jobs = original_plan(value)
        for job in jobs:
            position = job.arguments.index("--config") + 1
            job.arguments[position] = str(output / "configs" / f"b7_{STAGES[job.name]}.json")
        return jobs

    def fingerprint(value):
        h = hashlib.sha256(original_fingerprint(value).encode())
        h.update(file_sha256(__file__).encode())
        h.update((output / REPORT).read_bytes())
        for path in sorted((output / "configs").glob("b7_*.json")):
            h.update(path.name.encode())
            h.update(path.read_bytes())
        return h.hexdigest()

    # Only this driver process selects run-specific configs. Core files stay
    # unchanged; the driver's code/configs are included in the pipeline lock hash.
    pipeline.plan, pipeline.fingerprint = planned, fingerprint
    try:
        return pipeline.run(args)
    finally:
        pipeline.plan, pipeline.fingerprint = original_plan, original_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    migrate_parser = commands.add_parser("migrate")
    migrate_parser.add_argument("--source", type=Path, required=True)
    migrate_parser.add_argument("--output", type=Path, required=True)
    migrate_parser.add_argument("--skip-pending-validation", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    for name in ("data-root", "metadata", "ptb-root"):
        run_parser.add_argument("--" + name, type=Path, required=True)
    run_parser.add_argument("--gpu-p", default="0")
    run_parser.add_argument("--gpu-a", default="1")
    run_parser.add_argument("--gpu-b", default="1")
    run_parser.add_argument("--max-train", type=int, choices=(1, 2), default=2)
    run_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "migrate":
        migrate(args.source, args.output, skip_pending=args.skip_pending_validation)
    else:
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
