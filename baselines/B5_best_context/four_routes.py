"""Four fixed context routes on one base cache, with explicit frozen-body transfer."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, sha
from .adapters import _visible_digest
from .body_features import BODY_PROTOCOL, BEAT16_ENCODING, feature_encoding, relation_features
from .body_prepare import load_cache
from .body_alpha_core import SharedCorrection, FixedAlphaAdapter
from .body_study import run_study
from .evaluate import load_baseline, HistoryProvider


SELECTED_BODY_SHA256 = "c13605994098c015ecf390c49d94358e76446dd0e1e323c3001ad555623c7f8e"
SAMPLING = {"seed": 42, "solver": "heun", "steps": 16, "samples": 16}


def cases():
    return [("baseline", "none", None, None), ("d6_only", "fir25", None, None),
            ("body_only", "none", "body", None), ("both", "fir25", "body", None)]


def check_contract(payload, cache, cache_sha, baseline, population, *, allow_base_transfer):
    """Changing base predictions is deliberate; targets/QC/settings must stay identical."""
    frozen = cache["frozen"]
    training_base = frozen["base_checkpoint_sha256"]
    current_base = baseline["checkpoint_sha256"]
    transferred = current_base != training_base
    if transferred and not allow_base_transfer:
        raise ValueError("Different base model: explicitly pass --allow-base-transfer for a frozen-body transfer study")
    if (payload.get("protocol") != BODY_PROTOCOL or payload.get("group") != "context"
            or payload.get("record_gate") is not None or feature_encoding(payload) != BEAT16_ENCODING
            or payload.get("training_seed") != 42 or payload.get("epoch") != 27
            or payload.get("base_checkpoint_sha256") != training_base
            or payload.get("cache_sha256") != cache_sha
            or payload.get("sampling") != SAMPLING or payload.get("slow_beta") != .75
            or frozen.get("sampling") != SAMPLING or frozen.get("slow_beta") != .75
            or baseline.get("sampling") != SAMPLING or baseline.get("slow_beta") != .75):
        raise ValueError("Use selected O16-seed42-best27, its original cache, and unchanged Heun16/K16/seed42/beta=.75")
    for task in ("task1", "task2"):
        # Prediction hashes are allowed to change when moving to the new P bundle.
        for key in ("metadata_sha256", "target_sha256", "QC_sha256"):
            if population[task][key] != frozen["baseline_population"][task][key]:
                raise ValueError("Changed validation population/target/QC: " + task + "/" + key)
    return {"frozen_body_transfer": transferred, "body_training_base_sha256": training_base,
            "evaluated_base_sha256": current_base, "body_retrained": False,
            "body_alpha": 1., "feature_encoding": BEAT16_ENCODING}


def check_visible_features(cache_root, cache, item, provider):
    lookup = {r["pair_id"]: r for r in cache["records"] if r["split"] == "validation"}
    rows = [item["rows"][ids[0]] for ids in item["pairs"]
            if item["rows"][ids[0]]["input_type"] == "body_scale_d6"]
    if {r["pair_id"] for r in rows} != set(lookup):
        raise ValueError("Body cache and current validation have different pairs")
    for row in rows:
        original = lookup[row["pair_id"]]
        if any(row[key] != original[key] for key in ("subject_id", "target_record_id")):
            raise ValueError("Body validation identity changed")
        history = provider(row)
        if _visible_digest(history) != original["history_visible_sha256"]:
            raise ValueError("Visible body history changed since O16 training")
        features = relation_features(history, encoding=BEAT16_ENCODING)
        with np.load(Path(cache_root) / original["file"], allow_pickle=False) as record:
            if any(not np.array_equal(record[key], value)
                   for key, value in zip(("beat", "stats", "available"), features)):
                raise ValueError("Live O16 encoding differs from its training cache")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Data YAML, not public-slow study.json")
    p.add_argument("--checkpoint", required=True, help="Complete FM+slow bundle used for baseline inference")
    p.add_argument("--baseline-evaluation", required=True)
    p.add_argument("--body-checkpoint", required=True)
    p.add_argument("--body-cache", required=True, help="Original O16 feature cache; never replace its base hash")
    p.add_argument("--allow-base-transfer", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty study directory; keep logs outside it")
    if sha(args.body_checkpoint) != SELECTED_BODY_SHA256:
        raise ValueError("This study pins the selected O16 seed42 best27; supplied body weight is different")
    cache = load_cache(args.body_cache)
    loaded = load_baseline(args.baseline_evaluation)
    if loaded[0]["checkpoint_sha256"] != sha(args.checkpoint):
        raise ValueError("Baseline cache was generated by another complete model bundle")
    if (loaded[0]["checkpoint_sha256"] != cache["frozen"]["base_checkpoint_sha256"]
            and not args.allow_base_transfer):
        raise ValueError("Use --allow-base-transfer explicitly for the new P base")
    from baselines.B5.runtime import device_from_name
    from .body_model import BodyAdapter
    # Validate the immutable body weight against the base on which it WAS trained.
    # Application to the new base is separately authorized, checked and recorded below.
    engine = BodyAdapter(args.body_checkpoint, cache["frozen"]["base_checkpoint_sha256"],
                         device_from_name(args.device), expected_group="context")
    contract = check_contract(engine.payload, cache, sha(Path(args.body_cache) / "cache.json"),
                              loaded[0], loaded[2], allow_base_transfer=args.allow_base_transfer)
    provider = HistoryProvider(args.config, loaded[1]["task2"])
    try:
        check_visible_features(args.body_cache, cache, loaded[1]["task2"], provider)
        shared = SharedCorrection(engine)
        adapter = FixedAlphaAdapter(shared, 1.)
        summary = run_study(args.baseline_evaluation, output, provider, None, None, loaded=loaded,
            case_specs=cases(), adapter_map={"body": adapter},
            notes=["Same base predictions, targets, QC and visible history across all four groups.",
                   "Fixed d6 FIR25; selected frozen O16 seed42 best27, alpha=1. No training.",
                   "A changed base is an explicit frozen-body transfer experiment, not a retrained body result."])
        write(output / "four_routes.json", {"completed": True, "training_started": False,
            "FM_updates": 0, "slow_CNN_updates": 0, "body_updates": 0, **contract,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "body_checkpoint": str(Path(args.body_checkpoint).resolve()),
            "body_checkpoint_sha256": sha(args.body_checkpoint),
            "body_cache": str(Path(args.body_cache).resolve()),
            "body_cache_sha256": sha(Path(args.body_cache) / "cache.json"),
            "population": loaded[2], "sampling": SAMPLING, "slow_beta": .75,
            "shared_body_forward_calls": shared.forward_calls, "summary": summary})
    finally:
        provider.close()
    print("Completed: " + str(output / "summary.csv"), flush=True)


if __name__ == "__main__":
    main()
