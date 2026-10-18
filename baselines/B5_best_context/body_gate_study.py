"""Score source body vs matched C/G continuations with the d6 route frozen."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, sha, csv_write
from .body_prepare import load_cache
from .body_gate_features import GATE_PROTOCOL
from .body_features import BODY_PROTOCOL
from .body_study import run_study
from .evaluate import load_baseline, HistoryProvider
from .adapters import _visible_digest


def cases():
    return [("baseline", "none", None, None), ("d6_only", "fir25", None, None),
        ("source_body", "none", "source", None), ("source_both", "fir25", "source", None),
        ("C_body", "none", "C", None), ("C_both", "fir25", "C", None),
        ("G_body", "none", "G", None), ("G_both", "fir25", "G", None),
        ("G_disabled", "none", "G_disabled", None),
        *[(f"G_shuffled_{seed}", "none", "G", seed) for seed in (101, 202, 303)]]


def check_pair(left, right, source_sha, cache_sha):
    for mode, config in (("C", left), ("G", right)):
        if (config.get("protocol") != GATE_PROTOCOL or config.get("record_gate") != mode
                or config.get("init_body_checkpoint_sha256") != source_sha or config.get("cache_sha256") != cache_sha):
            raise ValueError("C/G config/source/cache mismatch")
    a, b = dict(left), dict(right)
    a.pop("record_gate"); b.pop("record_gate")
    if a != b:
        raise ValueError("C/G changed more than whether the record gate learns")


def diagnostics(output, cache_root, cache, source, control, gated):
    """Cached waveform diagnostics; no B5 generation or target-conditioned gate."""
    from .body_diagnose import waveform_metrics
    metrics, deltas = [], []
    for row in cache["records"]:
        if row["split"] != "validation" or not row["I_II_eligible"]:
            continue
        with np.load(Path(cache_root) / row["file"], allow_pickle=False) as a:
            arrays = {k: a[k].copy() for k in a.files}
        predictions = {"base": arrays["base_ii"]}
        for name, adapter in (("source", source), ("C", control), ("G", gated)):
            adapter.mode = "correct"
            predictions[name] = adapter.predict_ii_features(arrays["anchor"], arrays["base_ii"],
                arrays["beat"], arrays["stats"], arrays["available"])[0]
        values = waveform_metrics(arrays["anchor"], arrays["target_ii"], predictions)
        for name, value in values.items():
            metrics.append({"subject_id": row["subject_id"], "pair_id": row["pair_id"], "method": name, **value})
        def difference(a, b):
            return None if a is None or b is None else a - b
        deltas.append({"subject_id": row["subject_id"], "pair_id": row["pair_id"],
            "source_r_II": values["source"]["r_II"], "C_r_II": values["C"]["r_II"], "G_r_II": values["G"]["r_II"],
            "G_minus_C_r_II": difference(values["G"]["r_II"], values["C"]["r_II"]),
            "G_minus_source_r_II": difference(values["G"]["r_II"], values["source"]["r_II"]),
            "G_minus_C_fast_r": difference(values["G"]["fast_r"], values["C"]["fast_r"]),
            "G_minus_C_slow_r": difference(values["G"]["slow_r"], values["C"]["slow_r"])})
    csv_write(Path(output) / "body_component_metrics.csv", metrics)
    csv_write(Path(output) / "body_gate_deltas.csv", deltas)
    with (Path(output) / "G_body/postprocess_records.csv").open(encoding="utf-8-sig", newline="") as handle:
        statuses = {r["pair_id"]: r for r in csv.DictReader(handle) if r["input_type"] == "body_scale_d6"}
    alpha = [float(statuses[r["pair_id"]]["record_alpha"]) for r in deltas]
    write(Path(output) / "gate_effect.json", {"completed": True, "eligible_body_pairs": len(deltas),
        "G_vs_C_improved": sum(r["G_minus_C_r_II"] is not None and r["G_minus_C_r_II"] > 0 for r in deltas),
        "G_vs_C_declined": sum(r["G_minus_C_r_II"] is not None and r["G_minus_C_r_II"] < 0 for r in deltas),
        "G_vs_C_undefined": sum(r["G_minus_C_r_II"] is None for r in deltas),
        "alpha_min": min(alpha), "alpha_median": float(np.median(alpha)), "alpha_max": max(alpha),
        "alpha_equal_1_count": sum(v == 1 for v in alpha),
        "note": "If alpha stays1 or gains do not repeat, the current gate has not solved the failure mode. Do not set validation-patient switches."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--control-checkpoint", required=True)
    p.add_argument("--gate-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    from .body_model import BodyAdapter
    from baselines.B5.runtime import device_from_name
    cache = load_cache(args.cache)
    frozen = cache["frozen"]
    loaded = load_baseline(frozen["baseline_evaluation"])
    if loaded[2] != frozen["baseline_population"]:
        raise ValueError("Base validation predictions/QC changed")
    source_sha, cache_sha = sha(args.source_checkpoint), sha(Path(args.cache) / "cache.json")
    configs = [read(Path(path).parent / "config.json") for path in (args.control_checkpoint, args.gate_checkpoint)]
    check_pair(*configs, source_sha, cache_sha)
    device = device_from_name(args.device)
    base_sha = frozen["base_checkpoint_sha256"]
    engines = {name: BodyAdapter(path, base_sha, device, expected_group="context") for name, path in (
        ("source", args.source_checkpoint), ("C", args.control_checkpoint), ("G", args.gate_checkpoint))}
    if engines["source"].payload["protocol"] != BODY_PROTOCOL:
        raise ValueError("Source must be the original body context checkpoint")
    for name, engine in engines.items():
        if engine.payload["cache_sha256"] != cache_sha:
            raise ValueError("Adapter trained on a different cache")
        if name in ("C", "G") and (engine.payload.get("record_gate") != name
                or engine.payload.get("init_body_checkpoint_sha256") != source_sha
                or engine.payload["initial_weights_sha256"] != configs[0]["initial_weights_sha256"]):
            raise ValueError("C/G checkpoint and paired initialization mismatch")
    provider = HistoryProvider(args.config, loaded[1]["task2"])
    try:
        cached = {r["pair_id"]: r for r in cache["records"] if r["split"] == "validation"}
        for ids in loaded[1]["task2"]["pairs"]:
            row = loaded[1]["task2"]["rows"][ids[0]]
            if row["input_type"] == "body_scale_d6" and _visible_digest(provider(row)) != cached[row["pair_id"]]["history_visible_sha256"]:
                raise ValueError("Body history changed since cache preparation")
        summary = run_study(frozen["baseline_evaluation"], args.output, provider, None, None, loaded=loaded,
            case_specs=cases(), adapter_map={**engines, "G_disabled": engines["G"]},
            additive_routes=[(name + "_both", "d6_only", name + "_body") for name in ("source", "C", "G")],
            notes=["G_both − C_both 是门控的配对增量；C_both − source_both 是继续训练的影响。",
                "同时检查G_disabled、三个预设错配、逐患者alpha及快分量是否受损。",
                "B5-best/d6保持固定，仅体脂秤II修正器继续学习。"])
        diagnostics(args.output, args.cache, cache, engines["source"], engines["C"], engines["G"])
        write(Path(args.output) / "gate_checkpoints.json", {"completed": True,
            "source": str(Path(args.source_checkpoint).resolve()), "source_sha256": source_sha,
            "C": str(Path(args.control_checkpoint).resolve()), "C_sha256": sha(args.control_checkpoint),
            "G": str(Path(args.gate_checkpoint).resolve()), "G_sha256": sha(args.gate_checkpoint),
            "selected_epochs": {name: engine.payload["epoch"] for name, engine in engines.items()},
            "d6_frozen": True, "FM_updates": 0, "slow_CNN_updates": 0, "loss_unchanged": True,
            "training_configs": {"C": configs[0], "G": configs[1]}, "summary": summary})
    finally:
        provider.close()


if __name__ == "__main__":
    main()
