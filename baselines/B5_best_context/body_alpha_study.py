"""Scan five fixed alphas on one frozen legacy C checkpoint; no optimizer or FM generation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, csv_write, sha
from .adapters import _visible_digest
from .body_features import LEGACY_ENCODING, feature_encoding
from .body_gate_features import GATE_PROTOCOL
from .body_prepare import load_cache
from .body_study import run_study
from .body_alpha_core import (ALPHAS, FixedAlphaAdapter, SharedCorrection, cases,
                              group_name, target_mean, oracle_diagnostic)
from .evaluate import load_baseline, HistoryProvider


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def reference_check(records, reference, tolerance=1e-7):
    actual = {r["pair_id"]: r for r in records if r["alpha"] == 1.}
    expected = {r["pair_id"]: r for r in reference["pairs"]}
    if len(actual) != len([r for r in records if r["alpha"] == 1.]) or set(actual) != set(expected):
        raise ValueError("C checkpoint validation and scan have different eligible II pairs")
    maximum = 0.
    for pair_id, row in actual.items():
        other = expected[pair_id]
        if row["target_record_id"] != other["target_record_id"]:
            raise ValueError("Reference target identity changed")
        a, b = row["r_II"], other["r_II"]
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError("Reference II validity changed")
        else:
            maximum = max(maximum, abs(a - b))
    score = target_mean(list(actual.values()))
    if score is None or maximum > tolerance or abs(score - reference["validation_full120s_II_r"]) > tolerance:
        raise ValueError("Alpha1 does not reproduce the selected C validation result")
    return {"reproduced": True, "alpha1_II_r": score,
        "reference_II_r": reference["validation_full120s_II_r"],
        "max_pair_r_difference": maximum, "tolerance": tolerance}


def summarize(output, cache, reference, summary):
    output = Path(output)
    eligible = {r["pair_id"]: r for r in cache["records"] if r["split"] == "validation" and r["I_II_eligible"]}
    records = []
    for alpha in ALPHAS:
        metrics = {r["pair_id"]: r for r in table(output / group_name(alpha) / "pair_metrics.csv")
                   if r["task_id"] == "task2" and r["input_type"] == "body_scale_d6"}
        for pair_id, row in eligible.items():
            current = metrics[pair_id]
            if current["input_I_reliable"] != "True" or current["target_record_id"] != row["target_record_id"]:
                raise ValueError("Eligible body II identity/QC changed")
            value = float(current["r_II"]) if current["r_II"] else None
            records.append({"subject_id": row["subject_id"], "pair_id": pair_id,
                "target_record_id": row["target_record_id"], "alpha": alpha, "r_II": value})
    reproduction = reference_check(records, reference)
    anchors = {(r["pair_id"], r["alpha"]): r["r_II"] for r in records}
    for row in records:
        for name, alpha in (("delta_vs_alpha1", 1.), ("delta_vs_alpha0", 0.)):
            other = anchors[row["pair_id"], alpha]
            row[name] = None if row["r_II"] is None or other is None else row["r_II"] - other
    csv_write(output / "record_alpha.csv", records)
    oracle, choices = oracle_diagnostic(records)
    csv_write(output / "oracle_records.csv", [{**row, "oracle_alpha_diagnostic_only": choices.get(row["target_record_id"])}
        for row in records if row["alpha"] == 1.])
    write(output / "oracle_summary.json", oracle)
    summaries = {r["group"]: r for r in summary}
    zero, d6 = summaries[group_name(0.)], summaries["d6_only"]
    fields = ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")
    if any(abs(zero[f] - d6[f]) > 1e-12 for f in fields):
        raise AssertionError("Alpha0 must exactly reproduce d6-only scoring")
    rows = []
    for alpha in ALPHAS:
        group = summaries[group_name(alpha)]
        if group["task1_r"] != d6["task1_r"] or group["machine_r"] != d6["machine_r"]:
            raise AssertionError("Alpha scan changed Task1 or machine scores")
        valid = [r for r in records if r["alpha"] == alpha]
        rows.append({**group, "alpha": alpha, "body_II_r": target_mean(valid),
            "delta_mean_r_vs_alpha1": group["mean_r"] - summaries[group_name(1.)]["mean_r"],
            "improved_pairs_vs_alpha1": sum(r["delta_vs_alpha1"] is not None and r["delta_vs_alpha1"] > 1e-6 for r in valid),
            "declined_pairs_vs_alpha1": sum(r["delta_vs_alpha1"] is not None and r["delta_vs_alpha1"] < -1e-6 for r in valid),
            "undefined_II_pairs": sum(r["r_II"] is None for r in valid)})
    csv_write(output / "summary.csv", rows)
    lines = ["# 固定body修正强度诊断", "", "同一C权重与旧编码；d6固定；不训练。", "",
        "|alpha|体脂秤II r|体脂秤组r|Task2 r|平均r|相对alpha1|", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        ii = "undefined" if row["body_II_r"] is None else f"{row['body_II_r']:.8f}"
        lines.append(f"|{row['alpha']}|{ii}|{row['body_r']:.8f}|{row['task2_r']:.8f}|{row['mean_r']:.8f}|{row['delta_mean_r_vs_alpha1']:+.8f}|")
    lines += ["", "oracle_summary.json仅为使用真实目标挑alpha的诊断上限，不能用于测试或当作正式成绩。",
        "全局alpha候选仍由本验证集筛选，需要独立复验；不自动修改默认模型。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, reproduction


def render_examples(output, cache_root, cache, subjects, start, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from baselines.B5_best.diagnostics_core import moving_average
    if start < 0 or seconds <= 0 or start + seconds > 120:
        raise ValueError("Detail interval must stay within 120s")
    rows = [r for r in cache["records"] if r["split"] == "validation" and r["I_II_eligible"] and r["subject_id"] in subjects]
    missing = set(subjects) - {r["subject_id"] for r in rows}
    files = []
    metadata = table(Path(output) / group_name(1.) / "task2/window_metadata.csv")
    indices = {}
    for j, row in enumerate(metadata):
        indices.setdefault(row["pair_id"], []).append(j)
    predictions = {a: np.load(Path(output) / group_name(a) / "task2/prediction_uV.npy", mmap_mode="r") for a in ALPHAS}
    try:
        for row in rows:
            with np.load(Path(cache_root) / row["file"], allow_pickle=False) as item:
                truth, base = item["target_ii"], item["base_ii"]
            ids = sorted(indices[row["pair_id"]], key=lambda j: int(metadata[j]["start_sample_500hz"]))
            predicted = {a: np.concatenate([v[j, 1] for j in ids]) for a, v in predictions.items()}
            t = np.arange(60000) / 500
            for suffix, crop in (("overview", slice(None)), ("detail", slice(int(start * 500), int((start + seconds) * 500)))):
                fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
                for ax, target, values, label in ((axes[0], truth, predicted, "Original II (uV), no centering"),
                        (axes[1], moving_average(truth), {a: moving_average(v) for a, v in predicted.items()}, "Slow diagnostic component, full-record 501-point moving average")):
                    ax.plot(t[crop], target[crop], color="black", lw=1., label="Target")
                    for a, value in values.items():
                        ax.plot(t[crop], value[crop], lw=.7, alpha=.8, label=f"alpha={a:g}")
                    ax.set_title(label, loc="left", fontsize=10)
                axes[2].plot(t[crop], (truth - base)[crop], color="black", lw=.8, label="Target minus base (diagnostic only)")
                for a, value in predicted.items():
                    axes[2].plot(t[crop], (value - base)[crop], lw=.7, label=f"alpha={a:g}")
                axes[2].set_title("Applied correction (uV)", loc="left", fontsize=10)
                for ax in axes:
                    ax.grid(alpha=.2)
                    ax.set_ylabel("uV")
                    for boundary in range(10, 120, 10):
                        ax.axvline(boundary, ls=":", color=".7", lw=.5)
                axes[0].legend(ncol=3, fontsize=8)
                axes[2].set_xlabel("Time (s)")
                fig.suptitle(row["subject_id"] + " | " + row["pair_id"])
                fig.tight_layout()
                dest = Path(output) / "plots" / (row["pair_id"] + "_" + suffix + ".png")
                dest.parent.mkdir(exist_ok=True)
                fig.savefig(dest, dpi=130)
                plt.close(fig)
                files.append(str(dest.relative_to(output)))
    finally:
        for value in predictions.values():
            value._mmap.close()
    return {"files": files, "missing_QC_eligible_subjects": sorted(missing), "detail_start_s": start, "detail_seconds": seconds}


def run(args):
    from .body_model import BodyAdapter
    from baselines.B5.runtime import device_from_name
    output, checkpoint = Path(args.output).resolve(), Path(args.checkpoint).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; keep logs adjacent")
    cache = load_cache(args.cache)
    if feature_encoding(cache) != LEGACY_ENCODING:
        raise ValueError("Use original body_v1/cache, not the E/E0 encoded cache")
    frozen = cache["frozen"]
    loaded = load_baseline(frozen["baseline_evaluation"])
    if (loaded[2] != frozen["baseline_population"]
            or loaded[0]["checkpoint_sha256"] != frozen["base_checkpoint_sha256"]):
        raise ValueError("Base predictions/validation/QC changed since cache preparation")
    engine = BodyAdapter(checkpoint, frozen["base_checkpoint_sha256"], device_from_name(args.device), expected_group="context")
    payload = engine.payload
    if (payload.get("protocol") != GATE_PROTOCOL or payload.get("record_gate") != "C"
            or feature_encoding(payload) != LEGACY_ENCODING or payload.get("cache_sha256") != sha(Path(args.cache) / "cache.json")
            or payload.get("epoch") != args.expected_epoch):
        raise ValueError("Use the selected legacy C checkpoint at --expected-epoch (default26), never G/E/E0")
    reference_path = checkpoint.parent / "validation" / f"epoch_{payload['epoch']:03d}.json"
    reference = read(reference_path)
    if reference["epoch"] != payload["epoch"]:
        raise ValueError("Selected checkpoint/reference epoch mismatch")
    shared = SharedCorrection(engine)
    adapters = {group_name(a): FixedAlphaAdapter(shared, a) for a in ALPHAS}
    provider = HistoryProvider(args.config, loaded[1]["task2"])
    try:
        cached = {r["pair_id"]: r for r in cache["records"] if r["split"] == "validation"}
        for ids in loaded[1]["task2"]["pairs"]:
            row = loaded[1]["task2"]["rows"][ids[0]]
            if row["input_type"] != "body_scale_d6":
                continue
            record = cached[row["pair_id"]]
            if _visible_digest(provider(row)) != record["history_visible_sha256"]:
                raise ValueError("Visible body history changed")
            with np.load(Path(args.cache) / record["file"], allow_pickle=False) as arrays:
                for name, value in (("anchor", np.concatenate([loaded[1]["task2"]["target"][i, 0] for i in ids])),
                        ("base_ii", np.concatenate([loaded[1]["task2"]["prediction"][i, 1] for i in ids])),
                        ("target_ii", np.concatenate([loaded[1]["task2"]["target"][i, 1] for i in ids])),
                        ("quality", loaded[1]["task2"]["quality"][ids[0]])):
                    if not np.array_equal(arrays[name], value):
                        raise ValueError("Body cache and baseline differ: " + name)
        summary = run_study(frozen["baseline_evaluation"], output, provider, None, None, loaded=loaded,
            case_specs=cases(), adapter_map=adapters, additive_routes=[],
            notes=["固定C权重，只扫描五个alpha；所有组d6相同。", "不训练门控、不重新生成B5预测；oracle仅为诊断。"])
        summary, reproduction = summarize(output, cache, reference, summary)
        plots = render_examples(output, args.cache, cache, args.subjects, args.detail_start, args.detail_seconds) if not args.no_plots else {"skipped": True}
        write(output / "alpha_checkpoints.json", {"completed": True, "training_started": False,
            "optimizer_updates": 0, "FM_updates": 0, "slow_CNN_updates": 0, "d6_frozen": True,
            "B5_inference_executed": False, "gate_trained": False,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "selected_epoch": payload["epoch"],
            "feature_encoding": feature_encoding(payload), "cache": str(Path(args.cache).resolve()),
            "cache_sha256": sha(Path(args.cache) / "cache.json"), "config": str(Path(args.config).resolve()),
            "base_checkpoint_sha256": frozen["base_checkpoint_sha256"], "sampling": frozen["sampling"],
            "slow_beta": frozen["slow_beta"], "alphas": ALPHAS, "body_forward_calls": shared.forward_calls,
            "unique_visible_body_inputs": len(shared.cache), "reference_reproduction": reproduction,
            "target_used_for_fixed_alpha": False, "plots": plots, "summary": summary})
        for row in summary:
            print(row, flush=True)
        print("Completed: " + str(output), flush=True)
    finally:
        provider.close()
        for item in loaded[1].values():
            for name in ("prediction", "target", "quality"):
                item[name]._mmap.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-epoch", type=int, default=26)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--subjects", nargs="+", default=["HNU11135", "HNU11141", "HNU11119", "HNU11104"])
    p.add_argument("--detail-start", type=float, default=58.)
    p.add_argument("--detail-seconds", type=float, default=4.)
    run(p.parse_args())


if __name__ == "__main__":
    main()
