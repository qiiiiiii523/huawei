"""New complete base bundle versus fixed d6 and selected B-seed43-best15 context."""
from __future__ import annotations

import argparse
from pathlib import Path

from baselines.B5_best.common import write, sha
from .body_features import BODY_PROTOCOL, BEAT16_ENCODING, feature_encoding
from .body_history import B_POLICY, history_policy
from .body_prepare import load_cache
from .body_ab_study import validate_features
from .body_alpha_core import SharedCorrection, FixedAlphaAdapter
from .body_study import run_study
from .evaluate import load_baseline, HistoryProvider
from .four_routes import SAMPLING

SELECTED_B43_SHA256 = "8f850dad07ec88f63ef1e400ec97dc655174f62e09bd5eadc6c22c53f8a1bef3"


def cases():
    return [("baseline", "none", None, None), ("d6_only", "fir25", None, None),
            ("both", "fir25", "body", None)]


def check_contract(payload, cache, digest, baseline, population, *, allow_base_transfer):
    frozen = cache["frozen"]
    training_base = frozen["base_checkpoint_sha256"]
    current_base = baseline["checkpoint_sha256"]
    transferred = training_base != current_base
    if transferred and not allow_base_transfer:
        raise ValueError("Use --allow-base-transfer explicitly when applying frozen B43 body to a new base")
    if (payload.get("protocol") != BODY_PROTOCOL or payload.get("group") != "context"
            or payload.get("record_gate") is not None or payload.get("training_seed") != 43 or payload.get("epoch") != 15
            or feature_encoding(payload) != BEAT16_ENCODING or feature_encoding(cache) != BEAT16_ENCODING
            or history_policy(payload) != B_POLICY or history_policy(cache) != B_POLICY
            or payload.get("cache_sha256") != digest or payload.get("base_checkpoint_sha256") != training_base
            or any(x.get("sampling") != SAMPLING or x.get("slow_beta") != .75 for x in (payload, frozen, baseline))):
        raise ValueError("Use selected B-seed43-best15 and its original B cache with Heun16/K16/seed42/beta=.75")
    for task in ("task1", "task2"):
        for key in ("metadata_sha256", "target_sha256", "QC_sha256"):
            if population[task][key] != frozen["baseline_population"][task][key]:
                raise ValueError("Changed validation population/target/QC: " + task + "/" + key)
    return {"frozen_body_transfer": transferred, "body_training_base_sha256": training_base,
            "evaluated_base_sha256": current_base, "body_retrained": False,
            "body_training_seed": 43, "body_epoch": 15, "body_alpha": 1.,
            "feature_encoding": BEAT16_ENCODING, "history_preprocessing": B_POLICY}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="Complete new FM+slow bundle, not a single network checkpoint")
    p.add_argument("--baseline-evaluation", required=True)
    p.add_argument("--body-checkpoint", required=True)
    p.add_argument("--body-cache", required=True)
    p.add_argument("--allow-base-transfer", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty study directory; tee logs must stay outside it")
    if sha(args.body_checkpoint) != SELECTED_B43_SHA256:
        raise ValueError("Wrong body weight: this study pins B-seed43-best15")
    cache = load_cache(args.body_cache)
    loaded = load_baseline(args.baseline_evaluation)
    if loaded[0]["checkpoint_sha256"] != sha(args.checkpoint):
        raise ValueError("Baseline predictions were produced by another complete bundle")
    from baselines.B5.runtime import device_from_name
    from .body_model import BodyAdapter
    engine = BodyAdapter(args.body_checkpoint, cache["frozen"]["base_checkpoint_sha256"],
                         device_from_name(args.device), expected_group="context")
    contract = check_contract(engine.payload, cache, sha(Path(args.body_cache)/"cache.json"),
                              loaded[0], loaded[2], allow_base_transfer=args.allow_base_transfer)
    provider = HistoryProvider(args.config, loaded[1]["task2"])
    try:
        validate_features(args.body_cache, cache, loaded[1]["task2"], provider)
        shared = SharedCorrection(engine)
        rows = run_study(args.baseline_evaluation, output, provider, None, None, loaded=loaded,
            case_specs=cases(), adapter_map={"body": FixedAlphaAdapter(shared, 1.)}, additive_routes=[],
            notes=["Same new base predictions, data and QC across baseline/d6_only/both.",
                   "Selected B-seed43-best15 body frozen; d6 FIR25, O16, alpha=1; no training.",
                   "Body history is filtered once from raw visible records by its saved B preprocessing policy."])
        by_name = {r["group"]: r for r in rows}
        if any(r["task1_r"] != rows[0]["task1_r"] for r in rows):
            raise ValueError("Task1 changed across context routes")
        if (by_name["both"]["machine_r"] != by_name["d6_only"]["machine_r"]
                or by_name["d6_only"]["body_r"] != by_name["baseline"]["body_r"]):
            raise ValueError("Context contaminated another device route")
        write(output / "selected_context.json", {"completed": True, "training_started": False,
            "FM_updates": 0, "slow_CNN_updates": 0, "body_updates": 0, **contract,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "body_checkpoint": str(Path(args.body_checkpoint).resolve()), "body_checkpoint_sha256": sha(args.body_checkpoint),
            "body_cache": str(Path(args.body_cache).resolve()), "body_cache_sha256": sha(Path(args.body_cache)/"cache.json"),
            "sampling": SAMPLING, "slow_beta": .75, "population": loaded[2], "summary": rows,
            "d6_delta_mean_r": by_name["d6_only"]["mean_r"] - by_name["baseline"]["mean_r"],
            "body_delta_on_d6_mean_r": by_name["both"]["mean_r"] - by_name["d6_only"]["mean_r"]})
    finally:
        provider.close()
    print("Completed: " + str(output / "summary.csv"), flush=True)


if __name__ == "__main__":
    main()
