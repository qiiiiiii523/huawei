"""Compare saved paired evaluations, every lead, and local morphology."""
import argparse
import csv
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from baselines.B5_best import LEADS
from baselines.B5_best.common import csv_write, read, write
from baselines.B5_best.diagnose import align, pairs
from baselines.B5_best.diagnostics_core import corr, moving_average, qrs_candidates, local_shape


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compare(run_root, output, plots=True):
    root, output = Path(run_root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty comparison output")
    output.mkdir(parents=True, exist_ok=True)
    evaluations, summary = {}, []
    shared_study = None
    for group in ("control", "pcc"):
        init = read(root / group / "initialization.json")
        if shared_study is None:
            shared_study = init["study_sha256"]
        if init["study_sha256"] != shared_study:
            raise ValueError("Arms do not share the same study")
        for selection in ("best", "last"):
            directory = Path((root / group / f"latest_{selection}_evaluation.txt").read_text().strip())
            info = read(directory / "experiment.json")
            if info["group"] != group:
                raise ValueError("Evaluation/group mismatch")
            summary.append({"group": group, "checkpoint": selection, "epoch": info["epoch"], **info["score"]})
            if selection == "best":
                evaluations[group] = directory
    csv_write(output / "summary.csv", summary)
    leads = []
    for task in ("task1", "task2"):
        tables = {g: csv_read(p / task / "lead_metrics.csv") for g, p in evaluations.items()}
        lookup = {(r.get("input_type", "watch_ecg"), r["lead"]): r for r in tables["pcc"]}
        for a in tables["control"]:
            device, lead = a.get("input_type", "watch_ecg"), a["lead"]
            b = lookup[(device, lead)]
            if a["n_unique_target_records"] != b["n_unique_target_records"]:
                raise ValueError("Lead-QC population changed")
            leads.append({"task": task, "device": device, "lead": lead,
                          "control_r": float(a["r"]), "pcc_r": float(b["r"]),
                          "delta_r": float(b["r"]) - float(a["r"]),
                          "n_targets": int(a["n_unique_target_records"]),
                          "control_rmse_uV": float(a["rmse_uV"]), "pcc_rmse_uV": float(b["rmse_uV"])})
    csv_write(output / "lead_comparison.csv", leads)
    morphology, plot_candidates = [], []
    with ExitStack() as stack:
        def array(path):
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            stack.callback(value._mmap.close)
            return value
        aroot, broot = evaluations["control"] / "task2", evaluations["pcc"] / "task2"
        rows = csv_read(aroot / "window_metadata.csv")
        other = csv_read(broot / "window_metadata.csv")
        order = align(rows, other)
        a, b = array(aroot / "prediction_uV.npy"), array(broot / "prediction_uV.npy")
        target, target_b = array(aroot / "target_uV.npy"), array(broot / "target_uV.npy")
        qa, qb = array(aroot / "target_quality_mask.npy"), array(broot / "target_quality_mask.npy")
        chosen = {}
        for indices in pairs(rows):
            first = rows[indices[0]]
            key = first["target_record_id"]
            if key not in chosen or first["input_type"] == "ecg_machine_d6":
                chosen[key] = indices
        for key, indices in sorted(chosen.items()):
            matched = order[indices]
            if not np.array_equal(target[indices], target_b[matched]) or not np.array_equal(qa[indices], qb[matched]):
                raise ValueError("Targets/QC differ between arms")
            y = np.concatenate(target[indices], axis=-1)
            ca, cb = np.concatenate(a[indices], axis=-1), np.concatenate(b[matched], axis=-1)
            mask = qa[indices[0]]
            if not all(np.array_equal(mask, qa[i]) for i in indices):
                raise ValueError("QC changes within a target record")
            # This synchronous same-record I is observed at test time; bad I is explicitly skipped.
            anchors = qrs_candidates(y[0]) if mask[0] else np.array([], dtype=int)
            for lead in ("V1", "V3", "V5", "V6"):
                index = LEADS.index(lead)
                if not mask[index]:
                    continue
                yf = y[index] - moving_average(y[index])
                for group, pred in (("control", ca[index]), ("pcc", cb[index])):
                    pf = pred - moving_average(pred)
                    morphology.append({"target_record_id": key, "lead": lead, "group": group,
                          "input_I_good": bool(mask[0]), "full_record_r": corr(pred, y[index]),
                          **local_shape(pf, yf, anchors)})
                if len(anchors):
                    plot_candidates.append((key, lead, y[index], ca[index], cb[index], anchors))
        csv_write(output / "morphology_record_lead.csv", morphology)
        aggregate = []
        for lead in ("V1", "V3", "V5", "V6"):
            for group in ("control", "pcc"):
                entries = [r for r in morphology if r["lead"] == lead and r["group"] == group and r["input_I_good"]]
                fields = ["peak_amplitude_ratio_median", "abs_peak_time_error_ms_median",
                          "half_width_abs_error_ms_median", "qrs_r_unshifted_median"]
                row = {"lead": lead, "group": group, "n_good_I_records": len(entries)}
                for field in fields:
                    valid = [r[field] for r in entries if r[field] is not None]
                    row[field] = float(np.median(valid)) if valid else None
                aggregate.append(row)
        csv_write(output / "morphology_summary.csv", aggregate)
        if plots:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for lead in ("V1", "V3", "V5", "V6"):
                candidates = [r for r in plot_candidates if r[1] == lead]
                candidates.sort(key=lambda r: corr(r[3], r[2]) if corr(r[3], r[2]) is not None else -1.)
                for rank in sorted({0, len(candidates) // 2, len(candidates) - 1}):
                    if not candidates:
                        break
                    key, _, y, ca, cb, anchors = candidates[rank]
                    center = int(anchors[len(anchors) // 2])
                    left, right = max(0, center - 500), min(60000, center + 1000)
                    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
                    for signal, label in ((y, "Target"), (ca, "Control"), (cb, "SlowPCC")):
                        axes[0].plot(np.arange(left, right) / 500, signal[left:right], label=label)
                        axes[1].plot(np.arange(60000) / 500, moving_average(signal), label=label)
                    axes[0].set_title(f"{key} | {lead}: original voltage local detail")
                    axes[1].set_title("Full-record 501-point moving average (diagnostic only)")
                    for ax in axes:
                        ax.set_xlabel("Time (s)"); ax.set_ylabel("uV"); ax.legend(); ax.grid(alpha=.2)
                    fig.tight_layout()
                    fig.savefig(output / f"{key}_{lead}.png", dpi=120)
                    plt.close(fig)
    best = {r["group"]: r for r in summary if r["checkpoint"] == "best"}
    delta = best["pcc"]["main_correlation"] - best["control"]["main_correlation"]
    write(output / "comparison.json", {"completed": True, "study_sha256": shared_study,
          "best_delta_mean_r": delta, "scores": summary,
          "morphology_is_diagnostic_only": True, "predictions_shifted_for_scoring": False})
    text = ["# 慢CNN最终组合PCC实验比较", "", "|组别|权重|轮次|Task1 r|Task2 r|平均r|",
            "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        text.append(f"|{r['group']}|{r['checkpoint']}|{r['epoch']}|{r['task1_r_missing11']:.6f}|{r['task2_r_missing11']:.6f}|{r['main_correlation']:.6f}|")
    text += ["", f"实验组best相对对照best的平均r差：{delta:+.6f}。", "",
             "lead_comparison.csv保留全部导联及Task2两设备；morphology_summary.csv为正常I记录的局部诊断，不是正式评分。",
             "best可能仍为第0轮；若如此，不能据此说新增组合PCC已带来改善，需同时查看last和训练曲线。",
             "局部峰谷、时间偏差和代表图只解释错误，不做目标参与的校准或平移。RMSE仅诊断，不参与选模。"]
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print({"completed": True, "best_delta_mean_r": delta, "report": str(output / "report.md")}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    compare(args.run_root, args.output, plots=not args.no_plots)


if __name__ == "__main__":
    main()
