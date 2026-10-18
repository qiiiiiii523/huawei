"""Matched old/new body-history encoding evaluation; frozen B5-best and d6."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, sha, csv_write
from .body_encoding_prepare import check_cached_pair, ENCODING_STUDY
from .body_features import BODY_PROTOCOL, LEGACY_ENCODING, SHAPE_ENCODING, feature_encoding, relation_features
from .body_study import run_study
from .evaluate import load_baseline, HistoryProvider
from .adapters import _visible_digest


def cases():
    return [("baseline", "none", None, None), ("d6_only", "fir25", None, None),
        *[(f"{name}_body", "none", name, None) for name in ("O", "E")],
        *[(f"{name}_both", "fir25", name, None) for name in ("O", "E")],
        *[(f"{name}_disabled", "none", name + "_disabled", None) for name in ("O", "E")],
        *[(f"{name}_shuffled_{seed}", "none", name, seed)
          for name in ("O", "E") for seed in (101, 202, 303)]]


def check_training_pair(control, encoded, old_sha, new_sha, *, preflight=False):
    for cfg, encoding, cache_sha in ((control, LEGACY_ENCODING, old_sha), (encoded, SHAPE_ENCODING, new_sha)):
        if (cfg.get("protocol") != BODY_PROTOCOL or cfg.get("group") != "context"
                or cfg.get("record_gate") is not None or feature_encoding(cfg) != encoding
                or cfg.get("cache_sha256") != cache_sha):
            raise ValueError("Wrong encoding/cache/group in paired body configs")
    left, right = dict(control), dict(encoded)
    for cfg in (left, right):
        cfg.pop("feature_encoding", None)
        cfg.pop("cache_sha256", None)
    if left != right:
        raise ValueError("Encoding pair changed network initialization/budget/loss/seed or data counts")
    if preflight:
        for cfg in (control, encoded):
            if not cfg.get("ready") or cfg.get("optimizer_updates") != 0:
                raise ValueError("Both encoding CPU preflights must finish before training")


def cached_diagnostics(output, roots, caches, engines):
    from .body_diagnose import waveform_metrics
    metrics, deltas = [], []
    lookup = {name: {r["pair_id"]: r for r in cache["records"] if r["split"] == "validation"}
              for name, cache in caches.items()}
    for row in caches["O"]["records"]:
        if row["split"] != "validation" or not row["I_II_eligible"]:
            continue
        predictions = {}
        for name in ("O", "E"):
            with np.load(Path(roots[name]) / lookup[name][row["pair_id"]]["file"], allow_pickle=False) as a:
                predictions.setdefault("base", a["base_ii"].copy())
                anchor, target = a["anchor"].copy(), a["target_ii"].copy()
                engines[name].mode = "correct"
                predictions[name] = engines[name].predict_ii_features(anchor, a["base_ii"],
                    a["beat"], a["stats"], a["available"])[0]
        values = waveform_metrics(anchor, target, predictions)
        for name, value in values.items():
            metrics.append({"subject_id": row["subject_id"], "pair_id": row["pair_id"], "method": name, **value})
        def diff(a, b):
            return None if a is None or b is None else a - b
        deltas.append({"subject_id": row["subject_id"], "pair_id": row["pair_id"],
            "base_II_r": values["base"]["r_II"], "O_II_r": values["O"]["r_II"], "E_II_r": values["E"]["r_II"],
            "E_minus_O_II_r": diff(values["E"]["r_II"], values["O"]["r_II"]),
            "E_minus_base_II_r": diff(values["E"]["r_II"], values["base"]["r_II"]),
            "E_minus_O_fast_r": diff(values["E"]["fast_r"], values["O"]["fast_r"]),
            "E_minus_O_slow_r": diff(values["E"]["slow_r"], values["O"]["slow_r"])})
    csv_write(Path(output) / "body_component_metrics.csv", metrics)
    csv_write(Path(output) / "body_encoding_deltas.csv", deltas)
    write(Path(output) / "encoding_effect.json", {"completed": True, "eligible_body_pairs": len(deltas),
        "E_vs_O_improved_above_1e_6": sum(r["E_minus_O_II_r"] is not None and r["E_minus_O_II_r"] > 1e-6 for r in deltas),
        "E_vs_O_declined_below_minus_1e_6": sum(r["E_minus_O_II_r"] is not None and r["E_minus_O_II_r"] < -1e-6 for r in deltas),
        "E_vs_O_undefined": sum(r["E_minus_O_II_r"] is None for r in deltas)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--encoded-cache", required=True)
    p.add_argument("--control-run", required=True)
    p.add_argument("--encoded-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--check-only", action="store_true", help="Compare both CPU preflights without loading torch or training")
    args = p.parse_args()
    old, new = check_cached_pair(args.cache, args.encoded_cache)
    roots = {"O": args.cache, "E": args.encoded_cache}
    caches = {"O": old, "E": new}
    runs = {"O": Path(args.control_run), "E": Path(args.encoded_run)}
    configs = {name: read(root / ("preflight.json" if args.check_only else "config.json")) for name, root in runs.items()}
    check_training_pair(configs["O"], configs["E"], sha(Path(args.cache) / "cache.json"),
                        sha(Path(args.encoded_cache) / "cache.json"), preflight=args.check_only)
    if args.check_only:
        write(args.output, {"ready": True, "training_started": False, "optimizer_updates": 0,
            "same_initial_weights": True, "same_network_loss_budget_seed": True,
            "same_I_predictions_targets_QC_population": True,
            "initial_weights_sha256": configs["O"]["initial_weights_sha256"]})
        print({"ready": True, "same_initial_weights": True}, flush=True)
        return
    from .body_model import BodyAdapter
    from baselines.B5.runtime import device_from_name
    frozen = old["frozen"]
    loaded = load_baseline(frozen["baseline_evaluation"])
    if loaded[2] != frozen["baseline_population"]:
        raise ValueError("Original B5 predictions/validation/QC changed")
    device = device_from_name(args.device)
    engines = {name: BodyAdapter(run / "best.pt", frozen["base_checkpoint_sha256"], device,
                                expected_group="context") for name, run in runs.items()}
    for name, engine in engines.items():
        cfg, payload = configs[name], engine.payload
        if (payload.get("protocol") != BODY_PROTOCOL or payload.get("record_gate") is not None
                or feature_encoding(payload) != feature_encoding(cfg)
                or payload.get("cache_sha256") != cfg["cache_sha256"]
                or payload.get("initial_weights_sha256") != cfg["initial_weights_sha256"]
                or payload.get("training_seed", 42) != cfg["seed"]):
            raise ValueError("Selected body weight does not belong to the encoding pair")
        complete = read(runs[name] / "completed.json")
        if (complete.get("completed") is not True or complete["best_checkpoint_sha256"] != sha(runs[name] / "best.pt")
                or complete["epochs"] != cfg["epochs"] or complete["best_epoch"] != payload["epoch"]):
            raise ValueError("Finish both matching body training runs before evaluation")
    provider = HistoryProvider(args.config, loaded[1]["task2"])
    try:
        for name, cache in caches.items():
            lookup = {r["pair_id"]: r for r in cache["records"] if r["split"] == "validation"}
            for ids in loaded[1]["task2"]["pairs"]:
                row = loaded[1]["task2"]["rows"][ids[0]]
                if row["input_type"] != "body_scale_d6":
                    continue
                record = lookup[row["pair_id"]]
                history = provider(row)
                if _visible_digest(history) != record["history_visible_sha256"]:
                    raise ValueError("Body validation history changed")
                visible = relation_features(history, encoding=feature_encoding(cache))
                with np.load(Path(roots[name]) / record["file"], allow_pickle=False) as a:
                    if any(not np.array_equal(a[k], v) for k, v in zip(("beat", "stats", "available"), visible)):
                        raise ValueError("Live checkpoint encoding does not reproduce training features")
        summary = run_study(frozen["baseline_evaluation"], args.output, provider, None, None, loaded=loaded,
            case_specs=cases(), adapter_map={**engines, "O_disabled": engines["O"], "E_disabled": engines["E"]},
            additive_routes=[(name + "_both", "d6_only", name + "_body") for name in ("O", "E")],
            notes=["O与E只改变历史特征编码；同网络、同初始化、同预算、同loss、同seed，无门控。",
                   "E_both减O_both是新编码增量；同时看关闭context、全部三个打乱结果和逐记录II/快慢r。",
                   "旧编码仍是默认；本实验不会自动切换部署权重。"])
        cached_diagnostics(args.output, roots, caches, engines)
        write(Path(args.output) / "encoding_checkpoints.json", {"completed": True, "protocol": ENCODING_STUDY,
            "d6_frozen": True, "FM_updates": 0, "slow_CNN_updates": 0, "record_gate": None,
            "training_configs": configs, "summary": summary,
            "checkpoints": {name: {"path": str((run / "best.pt").resolve()), "sha256": sha(run / "best.pt"),
                "selected_epoch": engines[name].payload["epoch"], "feature_encoding": feature_encoding(engines[name].payload)}
                for name, run in runs.items()}})
    finally:
        provider.close()


if __name__ == "__main__":
    main()
