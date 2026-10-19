"""Compare P+d6+old-body versus paired P-aligned O16 A/B body adapters."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, sha, csv_write
from .adapters import _visible_digest
from .body_prepare import load_cache
from .body_features import BODY_PROTOCOL, BEAT16_ENCODING, feature_encoding, relation_features
from .body_history import A_POLICY, B_POLICY, process_history, history_policy
from .body_ab_prepare import check_pair, PROTOCOL
from .body_study import run_study
from .body_alpha_core import SharedCorrection, FixedAlphaAdapter
from .evaluate import load_baseline, HistoryProvider
from .four_routes import check_contract, SELECTED_BODY_SHA256


def check_training_pair(a, b, a_sha, b_sha, *, preflight=False):
    for cfg, digest, policy in ((a, a_sha, A_POLICY), (b, b_sha, B_POLICY)):
        if (cfg.get("protocol") != BODY_PROTOCOL or cfg.get("group") != "context"
                or cfg.get("record_gate") is not None or feature_encoding(cfg) != BEAT16_ENCODING
                or history_policy(cfg) != policy or cfg.get("cache_sha256") != digest):
            raise ValueError("A/B training cache/encoding/history protocol mismatch")
        if preflight and (cfg.get("ready") is not True or cfg.get("optimizer_updates") != 0):
            raise ValueError("Finish both A/B CPU forward/backward preflights")
    left, right = dict(a), dict(b)
    for cfg in (left, right):
        cfg.pop("cache_sha256", None)
        cfg.pop("history_preprocessing", None)
    if left != right:
        raise ValueError("A/B must share initialization, network, seed, loss, budget and supervised samples")


def cases():
    return [("baseline", "none", None, None), ("d6_only", "fir25", None, None),
            ("old_O16_both", "fir25", "old", None),
            ("A_both", "fir25", "A", None), ("B_both", "fir25", "B", None)]


def validate_features(cache_root, cache, item, provider):
    lookup = {r["pair_id"]: r for r in cache["records"] if r["split"] == "validation"}
    rows = [item["rows"][ids[0]] for ids in item["pairs"] if item["rows"][ids[0]]["input_type"] == "body_scale_d6"]
    if {r["pair_id"] for r in rows} != set(lookup):
        raise ValueError("Validation body population changed")
    for row in rows:
        record = lookup[row["pair_id"]]
        history = provider(row)
        if (_visible_digest(history) != record["history_visible_sha256"]
                or row["subject_id"] != record["subject_id"] or row["target_record_id"] != record["target_record_id"]):
            raise ValueError("Validation raw history/identity changed")
        processed = process_history(history, history_policy(cache))
        if "processed_history_sha256" in record and _visible_digest(processed) != record["processed_history_sha256"]:
            raise ValueError("Runtime history preprocessing changed")
        values = relation_features(processed, encoding=feature_encoding(cache))
        with np.load(Path(cache_root) / record["file"], allow_pickle=False) as a:
            if any(not np.array_equal(a[key], value) for key, value in zip(("beat", "stats", "available"), values)):
                raise ValueError("Inference history features differ from training cache")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--a-run", required=True)
    p.add_argument("--b-run", required=True)
    p.add_argument("--old-body-checkpoint", required=True)
    p.add_argument("--old-body-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    prepared = Path(args.prepared)
    roots = {"A": prepared / "A_cache", "B": prepared / "B_cache"}
    a, b, _ = check_pair(roots["A"], roots["B"])
    caches = {"A": a, "B": b}
    runs = {"A": Path(args.a_run), "B": Path(args.b_run)}
    configs = {name: read(root / ("preflight.json" if args.check_only else "config.json")) for name, root in runs.items()}
    digests = {name: sha(root / "cache.json") for name, root in roots.items()}
    check_training_pair(configs["A"], configs["B"], digests["A"], digests["B"], preflight=args.check_only)
    if args.check_only:
        write(args.output, {"ready": True, "same_initial_weights": True, "same_network_loss_seed_budget": True,
            "optimizer_updates": 0, "training_started": False, "initial_weights_sha256": configs["A"]["initial_weights_sha256"]})
        print("A/B preflight pair ready", flush=True)
        return
    if sha(args.old_body_checkpoint) != SELECTED_BODY_SHA256:
        raise ValueError("Use original selected O16 seed42 best27 as the frozen transfer reference")
    frozen = a["frozen"]
    loaded = load_baseline(frozen["baseline_evaluation"])
    if loaded[2] != frozen["baseline_population"] or loaded[0]["checkpoint_sha256"] != frozen["base_checkpoint_sha256"]:
        raise ValueError("P base predictions/population changed since preparation")
    from baselines.B5.runtime import device_from_name
    from .body_model import BodyAdapter
    device = device_from_name(args.device)
    engines = {}
    checkpoint_info = {}
    for name, root in runs.items():
        cfg, done = configs[name], read(root / "completed.json")
        engine = BodyAdapter(root / "best.pt", frozen["base_checkpoint_sha256"], device, expected_group="context")
        cp = engine.payload
        if (done.get("completed") is not True or done.get("epochs") != cfg["epochs"]
                or done.get("training_seed") != cfg["seed"] or done.get("best_epoch") != cp["epoch"]
                or done.get("best_checkpoint_sha256") != sha(root / "best.pt")
                or cp.get("protocol") != BODY_PROTOCOL or cp.get("record_gate") is not None
                or cp.get("cache_sha256") != digests[name] or cp.get("initial_weights_sha256") != cfg["initial_weights_sha256"]
                or cp.get("training_seed") != cfg["seed"] or feature_encoding(cp) != BEAT16_ENCODING
                or history_policy(cp) != history_policy(caches[name])):
            raise ValueError("Incomplete/mismatched A/B checkpoint: " + name)
        engines[name] = engine
        checkpoint_info[name] = {"path": str((root / "best.pt").resolve()), "sha256": sha(root / "best.pt"),
            "best_epoch": cp["epoch"], "history_preprocessing": history_policy(cp), "cache_sha256": digests[name]}
    old_cache = load_cache(args.old_body_cache)
    old = BodyAdapter(args.old_body_checkpoint, old_cache["frozen"]["base_checkpoint_sha256"], device, expected_group="context")
    transfer = check_contract(old.payload, old_cache, sha(Path(args.old_body_cache) / "cache.json"),
                              loaded[0], loaded[2], allow_base_transfer=True)
    engines["old"] = old
    provider = HistoryProvider(args.config, loaded[1]["task2"])
    try:
        for name in ("A", "B"):
            validate_features(roots[name], caches[name], loaded[1]["task2"], provider)
        validate_features(args.old_body_cache, old_cache, loaded[1]["task2"], provider)
        adapters = {name: FixedAlphaAdapter(SharedCorrection(engine), 1.) for name, engine in engines.items()}
        summary = run_study(frozen["baseline_evaluation"], args.output, provider, None, None, loaded=loaded,
            case_specs=cases(), adapter_map=adapters, additive_routes=[],
            notes=["Fixed P predictions and d6 FIR25; body alpha=1, O16 encoding throughout.",
                   "A/B starts from identical random weights; only history preprocessing differs.",
                   "old_O16_both is the original frozen body transferred to P, not a newly trained body."])
        by_name = {r["group"]: r for r in summary}
        for name in ("old_O16_both", "A_both", "B_both"):
            if by_name[name]["machine_r"] != by_name["d6_only"]["machine_r"]:
                raise ValueError("Body A/B contaminated machine route")
        rows = {}
        import csv
        for name in ("baseline", "old_O16_both", "A_both", "B_both"):
            with (Path(args.output) / name / "pair_metrics.csv").open(encoding="utf-8-sig", newline="") as h:
                rows[name] = {r["pair_id"]: r for r in csv.DictReader(h)
                              if r["task_id"] == "task2" and r["input_type"] == "body_scale_d6"
                              and r["input_I_reliable"] == "True" and r["r_II"] != ""}
        if any(set(v) != set(rows["baseline"]) for v in rows.values()):
            raise ValueError("Body II diagnosis populations differ")
        deltas = []
        for key, row in rows["baseline"].items():
            values = {name: float(data[key]["r_II"]) for name, data in rows.items()}
            deltas.append({"pair_id": key, "subject_id": row["subject_id"], "target_record_id": row["target_record_id"],
                **{name + "_II_r": value for name, value in values.items()},
                "B_minus_A_II_r": values["B_both"] - values["A_both"],
                "A_minus_old_II_r": values["A_both"] - values["old_O16_both"],
                "B_minus_old_II_r": values["B_both"] - values["old_O16_both"]})
        csv_write(Path(args.output) / "body_ab_deltas.csv", deltas)
        write(Path(args.output) / "body_ab_study.json", {"completed": True, "protocol": PROTOCOL,
            "training_started": False, "FM_updates": 0, "slow_CNN_updates": 0, "d6_frozen": True,
            "sampling": frozen["sampling"], "slow_beta": .75, "body_alpha": 1.,
            "base_checkpoint_sha256": frozen["base_checkpoint_sha256"], "population": loaded[2],
            "training_configs": configs, "checkpoints": checkpoint_info,
            "old_checkpoint_sha256": sha(args.old_body_checkpoint), "old_body_transfer": transfer,
            "summary": summary, "eligible_II_records": len(deltas),
            "B_vs_A_improved": sum(r["B_minus_A_II_r"] > 1e-6 for r in deltas),
            "B_vs_A_declined": sum(r["B_minus_A_II_r"] < -1e-6 for r in deltas)})
    finally:
        provider.close()
    print("Completed: " + str(Path(args.output) / "summary.csv"), flush=True)


if __name__ == "__main__":
    main()
