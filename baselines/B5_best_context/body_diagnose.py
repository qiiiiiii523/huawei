"""Plot fixed examples from EXISTING body weights/cache; never runs or updates B5-best."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, sha, csv_write
from baselines.B5_best.diagnostics_core import corr, moving_average
from .body_prepare import load_cache

METHODS = ("base", "no_context", "context")
LABELS = {"base": "B5-best", "no_context": "No-context adapter", "context": "Correct context"}
COLORS = {"base": "#2878b5", "no_context": "#e68a23", "context": "#26935c"}
DEFAULT_SUBJECTS = ("HNU11135", "HNU11141", "HNU11119", "HNU11104")


def waveform_metrics(anchor, target, predictions):
    """Full-record decomposition first, then any display crops. No mean/lag calibration."""
    if np.asarray(anchor).shape != (60000,) or np.asarray(target).shape != (60000,):
        raise ValueError("Diagnosis requires complete 120s I and target II")
    values = [anchor, target, *predictions.values()]
    if any(np.asarray(x).shape != (60000,) or not np.isfinite(x).all() for x in values):
        raise ValueError("All diagnosis waveforms must be finite full records")
    slow_target = moving_average(target, 501)
    fast_target = target - slow_target
    metrics = {}
    for method, predicted in predictions.items():
        slow = moving_average(predicted, 501)
        delta = predicted.astype(float) - predictions["base"]
        error = predicted.astype(float) - target
        metrics[method] = {"r_II": corr(predicted, target),
            "RMSE_uV": float(np.sqrt(np.mean(error ** 2))), "mean_error_uV": float(error.mean()),
            "slow_r": corr(slow, slow_target), "fast_r": corr(predicted - slow, fast_target),
            "slow_RMSE_uV": float(np.sqrt(np.mean((slow - slow_target) ** 2))),
            "fast_RMSE_uV": float(np.sqrt(np.mean(((predicted - slow) - fast_target) ** 2))),
            "correction_RMS_uV": float(np.sqrt(np.mean(delta ** 2))),
            "correction_max_abs_uV": float(np.max(np.abs(delta)))}
    return metrics


def select_examples(rows, subjects=DEFAULT_SUBJECTS, include_near_zero=True):
    selected = []
    for subject in subjects:
        matches = [r for r in rows if r["subject_id"] == subject]
        if not matches:
            raise ValueError("Requested subject has no QC-eligible body II record: " + subject)
        selected.extend(matches)
    selected_ids = {r["pair_id"] for r in selected}
    if include_near_zero:
        remaining = [r for r in rows if r["pair_id"] not in selected_ids]
        if remaining:
            selected.append(min(remaining, key=lambda r: (abs(r["delta_II_r"]), r["pair_id"])))
    return selected


def render_record(output, row, arrays, predictions, *, detail_start=58., detail_seconds=4.):
    """Raw uV, no per-window centering; detail uses the same interval for every method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if detail_start < 0 or detail_seconds <= 0 or detail_start + detail_seconds > 120:
        raise ValueError("Detail interval must stay within the 120s record")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_-]", "_", row["subject_id"] + "_" + row["pair_id"])
    t = np.arange(60000) / 500
    target, base = arrays["target_ii"], predictions["base"]
    slow_target = moving_average(target, 501)
    slow = {k: moving_average(v, 501) for k, v in predictions.items()}
    title = f"{row['subject_id']} | {row['pair_id']} | II | delta r={row['delta_II_r']:+.6f}"
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for ax in axes:
        for x in range(10, 120, 10):
            ax.axvline(x, color=".75", lw=.6, ls=":")
        ax.grid(alpha=.2)
    for ax, truth, predicted, subtitle in ((axes[0], target, predictions, "Original II: full 120s, no centering"),
            (axes[1], slow_target, slow, "Diagnostic slow component: full-record 501-point centered moving average")):
        ax.plot(t, truth, color="black", lw=.8, label="Target")
        for method in METHODS:
            ax.plot(t, predicted[method], color=COLORS[method], lw=.7, alpha=.8, label=LABELS[method])
        ax.set_title(subtitle, loc="left", fontsize=10)
        ax.set_ylabel("uV")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[2].plot(t, target - base, color=".35", lw=.6, label="Target minus B5-best (diagnostic only)")
    for method in ("no_context", "context"):
        axes[2].plot(t, predictions[method] - base, color=COLORS[method], lw=.8, label=LABELS[method] + " correction")
    axes[2].axhline(0, color="black", lw=.5)
    axes[2].set_title("Applied correction versus the original II error", loc="left", fontsize=10)
    axes[2].set_ylabel("uV"); axes[2].set_xlabel("Current record time (s)")
    axes[2].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .96))
    overview = output / (name + "_overview.png")
    fig.savefig(overview, dpi=140); plt.close(fig)

    left, right = int(detail_start * 500), int((detail_start + detail_seconds) * 500)
    part = slice(left, right)
    fig, axes = plt.subplots(5, 1, figsize=(14, 14))
    axes[0].plot(t[part], arrays["anchor"][part], color="black", lw=.8)
    axes[0].set_title("Visible current I; same fixed time interval for all methods", loc="left", fontsize=10)
    axes[0].set_ylabel("uV")
    for ax, truth, pred, subtitle in (
            (axes[1], target, predictions, "Original II, no voltage/lag calibration"),
            (axes[2], target - slow_target, {k: predictions[k] - slow[k] for k in METHODS},
             "Fast residual: raw II minus its own full-record slow component (diagnostic only)")):
        ax.plot(t[part], truth[part], color="black", lw=1., label="Target")
        for method in METHODS:
            ax.plot(t[part], pred[method][part], color=COLORS[method], lw=.8, label=LABELS[method])
        ax.set_title(subtitle, loc="left", fontsize=10); ax.set_ylabel("uV")
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)
    axes[3].plot(t[part], (target - base)[part], color=".35", lw=.8, label="Target minus B5-best")
    for method in ("no_context", "context"):
        axes[3].plot(t[part], (predictions[method] - base)[part], color=COLORS[method], lw=.9, label=method + " correction")
    axes[3].set_title("Correction direction and strength", loc="left", fontsize=10)
    axes[3].set_ylabel("uV"); axes[3].legend(fontsize=8)
    for ax in axes[:4]:
        for x in range(10, 120, 10):
            if detail_start <= x <= detail_start + detail_seconds:
                ax.axvline(x, color=".6", lw=.7, ls=":")
        ax.set_xlim(detail_start, detail_start + detail_seconds)
        ax.set_xlabel("Current record time (s)"); ax.grid(alpha=.2)
    beat_t = (np.arange(300) - 100) / 500
    axes[4].plot(beat_t, arrays["beat"][0], label="Historical body I")
    axes[4].plot(beat_t, arrays["beat"][1], label="Historical body II")
    axes[4].set_title(f"Historical median templates | available={bool(arrays['available'])} | independent time, NOT current target", loc="left", fontsize=10)
    axes[4].set_xlabel("Time relative to history I-only detected event (s)")
    axes[4].set_ylabel("Normalized by history I RMS"); axes[4].grid(alpha=.2); axes[4].legend(fontsize=8)
    fig.suptitle(title + f" | fixed detail {detail_start:g}--{detail_start + detail_seconds:g}s")
    fig.tight_layout(rect=(0, 0, 1, .97))
    detail = output / (name + "_detail.png")
    fig.savefig(detail, dpi=140); plt.close(fig)
    return [overview.name, detail.name]


def run(args):
    output, root, study = Path(args.output).resolve(), Path(args.cache).resolve(), Path(args.study).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new diagnostics directory; tee logs must be adjacent")
    cache = load_cache(root)
    frozen, source = cache["frozen"], read(study / "body_checkpoints.json")
    study_report = read(study / "study.json")
    if (study_report.get("completed") is not True or study_report["population"] != frozen["baseline_population"]
            or study_report["base_checkpoint_sha256"] != frozen["base_checkpoint_sha256"]):
        raise ValueError("Study/cache have different B5-best predictions/QC/population")
    from .body_model import BodyAdapter
    from baselines.B5.runtime import device_from_name
    device = device_from_name(args.device)
    engines = {}
    for method, field, group in (("context", "context", "context"), ("no_context", "no_context", "no_context")):
        if sha(source[field]) != source[field + "_sha256"]:
            raise ValueError("Study checkpoint bytes changed: " + field)
        engine = BodyAdapter(source[field], frozen["base_checkpoint_sha256"], device, expected_group=group)
        if engine.payload["cache_sha256"] != sha(root / "cache.json"):
            raise ValueError("Checkpoint/cache mismatch")
        engines[method] = engine
    if (engines["context"].payload["initial_weights_sha256"] != engines["no_context"].payload["initial_weights_sha256"]
            or engines["context"].payload.get("training_seed", 42) != engines["no_context"].payload.get("training_seed", 42)):
        raise ValueError("Diagnosis requires a matched training pair")
    expected = {}
    for method, name in (("base", "baseline"), ("no_context", "body_no_context"), ("context", "body_only")):
        with (study / name / "pair_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            expected[method] = {r["pair_id"]: float(r["r_II"]) for r in csv.DictReader(handle)
                if r["input_type"] == "body_scale_d6" and r["input_I_reliable"] == "True" and r["r_II"]}
    eligible_pairs = {r["pair_id"] for r in cache["records"] if r["split"] == "validation" and r["I_II_eligible"]}
    if any(set(values) != eligible_pairs for values in expected.values()):
        raise ValueError("Study/cache eligible body II populations differ")
    output.mkdir(parents=True, exist_ok=True)
    records, metrics_rows, memory = [], [], {}
    for record in cache["records"]:
        if record["split"] != "validation" or not record["I_II_eligible"]:
            continue
        with np.load(root / record["file"], allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        predictions = {"base": arrays["base_ii"]}
        for method, engine in engines.items():
            predictions[method] = engine.predict_ii_features(arrays["anchor"], arrays["base_ii"],
                arrays["beat"], arrays["stats"], arrays["available"])[0]
        metrics = waveform_metrics(arrays["anchor"], arrays["target_ii"], predictions)
        for method, values in metrics.items():
            previous = expected[method].get(record["pair_id"])
            if previous is None or values["r_II"] is None or abs(values["r_II"] - previous) > 1e-7:
                raise ValueError("Reconstructed II does not match completed study: " + record["pair_id"])
            metrics_rows.append({"subject_id": record["subject_id"], "pair_id": record["pair_id"], "method": method, **values})
        row = {"subject_id": record["subject_id"], "pair_id": record["pair_id"],
            "target_record_id": record["target_record_id"], "delta_II_r": metrics["context"]["r_II"] - metrics["base"]["r_II"]}
        records.append(row); memory[record["pair_id"]] = arrays, predictions
    selected = select_examples(records, args.subjects, not args.no_near_zero)
    plots = []
    for row in selected:
        arrays, predictions = memory[row["pair_id"]]
        files = render_record(output, row, arrays, predictions, detail_start=args.detail_start, detail_seconds=args.detail_seconds)
        plots.append({**row, "files": files})
        print(f"Plotted: {row['subject_id']}, delta II r={row['delta_II_r']:+.6f}", flush=True)
    csv_write(output / "record_metrics.csv", metrics_rows)
    write(output / "diagnostics.json", {"completed": True, "training_started": False,
        "B5_inference_executed": False, "optimizer_updates": 0, "study": str(study), "cache": str(root),
        "training_seed": engines["context"].payload.get("training_seed", 42),
        "checkpoint_info": source, "all_eligible_II_scores_reproduced": True,
        "detail_start_s": args.detail_start, "detail_duration_s": args.detail_seconds,
        "selected_for_diagnosis_only": plots, "target_used_for_display_and_metrics_only": True,
        "note": "501-point components are diagnostics, not physiological baseline identification; history templates are not aligned to current target."})
    lines = ["# Body II 波形诊断", "", "所有有效验证记录均重建并核对已完成study的II r；仅以下固定案例画图。", "",
        "原始μV，不扣目标均值、不平移；快慢分量先对完整120秒计算，再裁取局部。", "",
        "|患者|配对|ΔII r|", "|---|---|---:|"]
    for r in plots:
        lines.append(f"|{r['subject_id']}|{r['pair_id']}|{r['delta_II_r']:+.6f}|")
    lines += ["", "record_metrics.csv含全部患者的原始/快/慢r、RMSE及修正幅度。",
        "历史模板是输入特征快照，不能证明不同历史片段的关系稳定；需要分段稳定性实验时再读取原历史。",
        "图中使用真实目标仅为事后诊断，不改变测试输出或正式评分。"]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Completed: " + str(output), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--study", required=True, help="Completed body_study including body_checkpoints.json")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    p.add_argument("--no-near-zero", action="store_true")
    p.add_argument("--detail-start", type=float, default=58.)
    p.add_argument("--detail-seconds", type=float, default=4.)
    run(p.parse_args())


if __name__ == "__main__":
    main()
