"""Diagnose EXISTING B5/S50-F evaluation arrays. CPU only; no weights are loaded."""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from ecg12gen.evaluate import evaluate_record_predictions
from . import LEADS, SAMPLING
from .common import read, write, sha
from .diagnostics_core import (corr, moving_average, component_metrics, boundary_metrics,
                               qrs_candidates, local_shape)

PROTOCOL = "b5-best-existing-array-diagnostics-v1"


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_save(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def identity(row):
    return (row["pair_id"], row["target_record_id"], int(row["start_sample_500hz"]))


def align(reference, other):
    keys = [identity(r) for r in other]
    if len(set(keys)) != len(keys) or len({identity(r) for r in reference}) != len(reference):
        raise ValueError("Duplicate pair/target/window key")
    lookup = {k: i for i, k in enumerate(keys)}
    if set(lookup) != {identity(r) for r in reference}:
        raise ValueError("Evaluation populations differ; do not align by array position")
    order = [lookup[identity(r)] for r in reference]
    for a, i in zip(reference, order):
        for key in ("subject_id", "input_type"):
            if a.get(key) != other[i].get(key):
                raise ValueError("Evaluation identity differs: " + key)
    return np.asarray(order)


def pairs(rows):
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["pair_id"], []).append(i)
    out = []
    for ids in groups.values():
        ids.sort(key=lambda i: int(rows[i]["start_sample_500hz"]))
        if [int(rows[i]["start_sample_500hz"]) for i in ids] != list(range(0, 60000, 5000)):
            raise ValueError("Requires twelve real consecutive 10-second windows per pair")
        for key in ("target_record_id", "subject_id", "input_type"):
            if len({rows[i][key] for i in ids}) != 1:
                raise ValueError("Mixed identities within a pair: " + key)
        out.append(ids)
    return out


def record(array, ids):
    return np.concatenate([array[i] for i in ids], axis=-1).astype(float)


def target_file(folder):
    p = folder / "target_uV.npy"
    if p.is_file():
        return p
    ref = read(folder / "target_reference.json")
    p = Path(ref["path"])
    if not p.is_absolute():
        p = folder / p
    if not p.is_file() or sha(p) != ref["sha256"]:
        raise ValueError("Target reference missing or changed: " + str(p))
    return p


@dataclass
class Task:
    name: str
    rows: list
    fm: np.ndarray
    hybrid: np.ndarray
    target: np.ndarray
    quality: np.ndarray
    groups: list
    paths: dict
    resources: ExitStack | None = None

    def close(self):
        if self.resources is not None:
            self.resources.close()
            self.resources = None


def load_task(task, fm_dir, hybrid_dir, qc_dir, inherited_hybrid_order=False):
    with ExitStack() as resources:
        return _load_task(task, fm_dir, hybrid_dir, qc_dir, inherited_hybrid_order, resources)


def _load_task(task, fm_dir, hybrid_dir, qc_dir, inherited_hybrid_order, resources):
    def mapped(path):
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        resources.callback(value._mmap.close)
        return value
    a, b, c = (Path(d) / task for d in (fm_dir, hybrid_dir, qc_dir))
    rows = csv_read(a / "window_metadata.csv")
    fm = mapped(a / "prediction_uV.npy")
    target = mapped(target_file(a))
    hybrid = mapped(b / "prediction_uV.npy")
    if (b / "window_metadata.csv").is_file():
        order = align(rows, csv_read(b / "window_metadata.csv"))
        if not np.array_equal(order, np.arange(len(rows))):
            hybrid = hybrid[order]
    elif not inherited_hybrid_order:
        raise ValueError("Hybrid has no metadata: require verified original slow-evaluation provenance")
    qc_rows = csv_read(c / "window_metadata.csv")
    order = align(rows, qc_rows)
    quality = np.load(c / "target_quality_mask.npy", allow_pickle=False)[order]
    if quality.shape != (len(rows), 12) or not np.isin(quality, (0, 1)).all():
        raise ValueError("QC requires explicit boolean/0-1 [N,12] mask")
    qc_target_path = target_file(c)
    qc_target = mapped(qc_target_path)
    expected = (len(rows), 12, 5000)
    if any(x.shape != expected for x in (fm, hybrid, target, qc_target)):
        raise ValueError("Expected matching [N,12,5000] arrays in " + task)
    for i, j in enumerate(order):
        if not np.array_equal(target[i], qc_target[j], equal_nan=True):
            raise ValueError("QC directory targets differ from original FM targets")
    groups = pairs(rows)
    for ids in groups:
        if not np.all(quality[ids] == quality[ids[0]]):
            raise ValueError("QC changes between windows")
    return Task(task, rows, fm, hybrid, target, quality.astype(bool), groups,
                {"fm": str(a), "hybrid": str(b), "quality": str(c),
                 "target": str(target_file(a)), "qc_target": str(qc_target_path)}, resources.pop_all())


def task_r(task, prediction):
    summary, _ = evaluate_record_predictions(prediction, task.target, task.name, task.rows,
                                              target_quality_mask=task.quality)
    # Keep exactly the established target/lead/device macro aggregation.
    return float(summary["r_missing11"])


def score_betas(tasks, output, betas=(.75, 1., 1.25)):
    rows = []
    for beta in betas:
        scores = {}
        for name, task in tasks.items():
            if beta == 1:
                scores[name] = task_r(task, task.hybrid)
                continue
            path = output / ("_temporary_beta_" + name + ".npy")
            candidate = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=task.fm.shape)
            try:
                for ids in task.groups:
                    p, h = record(task.fm, ids), record(task.hybrid, ids)
                    fast = p[1:] - moving_average(p[1:])
                    # Recover the actual predicted slow curve from its saved hybrid.
                    s = h[1:] - fast
                    value = p.copy()
                    value[1:] = fast + beta * s
                    candidate[ids] = value.reshape(12, 12, 5000).transpose(1, 0, 2)
                candidate.flush()
                scores[name] = task_r(task, candidate)
            finally:
                candidate._mmap.close()
                path.unlink(missing_ok=True)
        rows.append({"beta": beta, "task1_r": scores["task1"], "task2_r": scores["task2"],
                     "mean_r": (scores["task1"] + scores["task2"]) / 2,
                     "status": "validation_probe_only_not_a_selected_deployment_model"})
    return rows


def summarize(rows, grouping, fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in grouping), []).append(row)
    result = []
    for key, values in groups.items():
        entry = dict(zip(grouping, key)) | {"n_record_leads": len(values)}
        for field in fields:
            x = [r[field] for r in values if isinstance(r.get(field), (float, int))
                 and np.isfinite(r[field])]
            entry[field + "_mean"] = float(np.mean(x)) if x else None
            entry[field + "_n_defined"] = len(x)
        result.append(entry)
    return result


def unique_records(task):
    selected = {}
    for ids in task.groups:
        first = task.rows[ids[0]]
        key = first["target_record_id"]
        if key in selected:
            old = selected[key]
            if not np.array_equal(record(task.target, ids), record(task.target, old), equal_nan=True):
                raise ValueError("Repeated target waveforms differ")
            if not np.array_equal(task.quality[ids[0]], task.quality[old[0]]):
                raise ValueError("Repeated target QC differs")
        if key not in selected or first["input_type"] == "ecg_machine_d6":
            selected[key] = ids
    return selected


def detail_metrics(tasks, leads):
    t = tasks["task2"]
    shared = {r["target_record_id"] for r in tasks["task1"].rows}
    components, beats, boundaries, anchor_rows = [], [], [], []
    for key, ids in unique_records(t).items():
        row = t.rows[ids[0]]
        y, p, h = record(t.target, ids), record(t.fm, ids), record(t.hybrid, ids)
        mask = t.quality[ids[0]]
        anchors = qrs_candidates(y[0]) if mask[0] else np.array([], dtype=int)
        anchor_rows.append({"target_record_id": key, "input_I_good": bool(mask[0]),
                            "n_approximate_I_qrs_anchors": len(anchors),
                            "status": "approximate_I_only_not_clinical_annotations"})
        for j, lead in enumerate(LEADS[1:], 1):
            if not mask[j]:
                continue
            if not all(np.isfinite(x[j]).all() for x in (y, p, h)):
                raise ValueError("Nonfinite QC-good signal: " + key + " " + lead)
            f = p[j] - moving_average(p[j])
            s = h[j] - f
            meta = {"target_record_id": key, "subject_id": row["subject_id"], "lead": lead,
                    "shared_with_task1": key in shared, "quality": "good",
                    "reference_pair_id": row["pair_id"], "reference_device": row["input_type"]}
            components.append(meta | component_metrics(p[j], s, y[j]))
            for method, wave in (("fm", p[j]), ("hybrid", h[j])):
                boundaries.append(meta | {"method": method} | boundary_metrics(wave, y[j]))
                if lead in leads:
                    beats.append(meta | {"method": method} |
                                 local_shape(wave - moving_average(wave),
                                             y[j] - moving_average(y[j]), anchors))
        print("diagnostic target: " + key, flush=True)
    return components, beats, boundaries, anchor_rows


def plot_examples(tasks, components, leads, output, max_records=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = tasks["task2"]
    roster = unique_records(t)
    ranking = []
    for key in roster:
        values = [r["hybrid_raw_r"] for r in components if r["target_record_id"] == key
                  and r["hybrid_raw_r"] is not None]
        if values:
            ranking.append((float(np.mean(values)), key))
    ranking.sort()
    indices = sorted(set(np.linspace(0, len(ranking) - 1, min(max_records, len(ranking)), dtype=int)))
    folder = output / "plots"
    folder.mkdir()
    manifest = []
    for index in indices:
        diagnostic_r, key = ranking[index]
        ids = roster[key]
        y, p, h = record(t.target, ids), record(t.fm, ids), record(t.hybrid, ids)
        anchors = qrs_candidates(y[0]) if t.quality[ids[0], 0] else np.array([], dtype=int)
        center = int(anchors[len(anchors) // 2]) if len(anchors) else 30000
        lo, hi = max(0, center - 1500), min(60000, center + 1500)
        time = np.arange(60000) / 500
        for lead in leads:
            j = LEADS.index(lead)
            if not t.quality[ids[0], j]:
                continue
            ys, ps = moving_average(y[j]), moving_average(p[j])
            f = p[j] - ps
            s = h[j] - f
            hf, yf = h[j] - moving_average(h[j]), y[j] - ys
            fig, ax = plt.subplots(5, 1, figsize=(15, 15))
            for z, label, color in ((ys, "Target slow", "black"), (ps, "FM slow", "gray"),
                                     (s, "S50-F slow", "tab:blue")):
                ax[0].plot(time[::10], (z - z.mean())[::10], label=label, lw=.8, color=color)
            ax[0].set_title("120s slow SHAPE: own means removed FOR DISPLAY ONLY; score unchanged")
            for z, label, color in ((y[j], "Target", "black"), (p[j], "FM", "gray"), (h[j], "Hybrid", "tab:blue")):
                ax[1].plot(time[lo:hi], z[lo:hi], label=label, lw=.9, color=color)
            ax[1].set_title("Raw-voltage detail, common middle heartbeat neighborhood")
            for z, label, color in ((yf, "Target fast", "black"), (f, "FM fast", "gray"), (hf, "Hybrid fast", "tab:blue")):
                ax[2].plot(time[lo:hi], z[lo:hi], label=label, lw=.9, color=color)
            ax[2].set_title("Fast residual detail: no shift or amplitude calibration")
            if len(anchors):
                for z, label, color in ((yf, "Target", "black"), (f, "FM fast", "gray"), (hf, "Hybrid fast", "tab:blue")):
                    stack = np.stack([z[q - 100:q + 200] for q in anchors])
                    ax[3].plot(np.arange(-100, 200) * 2, np.median(stack, axis=0), label=label, color=color)
                ax[3].set_title("Median I-anchored beat FOR DIAGNOSIS; not clinical delineation")
                ax[3].set_xlabel("Time relative to approximate I QRS (ms)")
            else:
                ax[3].text(.1, .5, "No reliable I-only QRS candidates; see qrs_anchor_audit.csv")
            for z, label, color in ((ys, "Target slow", "black"), (s, "Predicted slow", "tab:blue")):
                frequencies = np.fft.rfftfreq(len(z), 1 / 500)
                power = np.abs(np.fft.rfft(z - z.mean())) ** 2 / len(z) ** 2
                band = (frequencies >= .02) & (frequencies <= 15)
                ax[4].plot(frequencies[band], 10 * np.log10(power[band] + 1e-12), label=label, color=color, lw=.7)
            ax[4].set_title("Slow-curve spectral comparison; moving average is NOT an ideal cutoff")
            ax[4].set_xlabel("Frequency (Hz)")
            for a in ax:
                a.grid(alpha=.2)
                handles, labels = a.get_legend_handles_labels()
                if handles:
                    a.legend(loc="upper right")
            fig.suptitle(f"{key} | {lead} | representative diagnostic record r={diagnostic_r:.3f}")
            fig.tight_layout()
            path = folder / f"{key}_{lead}.png"
            fig.savefig(path, dpi=115)
            plt.close(fig)
            manifest.append({"path": str(path.relative_to(output)), "target_record_id": key, "lead": lead,
                             "selection": "low/median/high unique-record diagnostic r; label used ONLY for diagnosis"})
    return manifest


def integration_report(root, tasks):
    rows = []
    for group in ("control", "slow", "slow_f"):
        folder = Path(root) / group / "evaluations" / "best_r"
        if not (folder / "evaluation.json").is_file():
            rows.append({"group": group, "path": str(folder), "status": "not_available_not_generated"})
            continue
        marker = read(folder / "evaluation.json")
        if not marker.get("completed") or marker.get("fingerprint", {}).get("sampling") != SAMPLING:
            raise ValueError("Incomplete/different sampling integration evaluation: " + str(folder))
        missing = [str(folder / name / f) for name in tasks for f in
                   ("prediction_uV.npy", "target_quality_mask.npy", "window_metadata.csv")
                   if not (folder / name / f).is_file()]
        if missing:
            rows.append({"group": group, "path": str(folder),
                         "status": "optional_arrays_missing_not_regenerated", "missing": ";".join(missing)})
            continue
        values = {}
        with ExitStack() as resources:
            for name, task in tasks.items():
                d = folder / name
                order = align(task.rows, csv_read(d / "window_metadata.csv"))
                y = np.load(target_file(d), mmap_mode="r", allow_pickle=False)
                resources.callback(y._mmap.close)
                quality = np.load(d / "target_quality_mask.npy", allow_pickle=False)[order]
                if not np.array_equal(task.quality, quality):
                    raise ValueError("Integration QC differs")
                for i, j in enumerate(order):
                    if not np.array_equal(task.target[i], y[j], equal_nan=True):
                        raise ValueError("Integration targets differ")
                pred = np.load(d / "prediction_uV.npy", mmap_mode="r", allow_pickle=False)
                resources.callback(pred._mmap.close)
                if not np.array_equal(order, np.arange(len(order))):
                    pred = pred[order]
                values[name] = task_r(task, pred)
        rows.append({"group": group, "path": str(folder), "status": "recomputed_from_existing_arrays_same_QC",
                     "task1_r": values["task1"], "task2_r": values["task2"],
                     "mean_r": (values["task1"] + values["task2"]) / 2})
    return rows


def run(args):
    repo = Path(args.repository_root).resolve()
    prepared = Path(args.prepared_info or repo / "results/B5-exp/slow_probe_parallel_v1/prepared/prepared.json")
    info = read(prepared)
    profile = read(Path(__file__).resolve().parents[2] / "configs/b5_best.json")
    fm_dir = Path(args.fm_evaluation_dir or info["evaluation_dir"])
    slow_root = Path(args.slow_evaluation_dir or repo / "results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints/evaluation_best_score")
    hybrid_dir = slow_root / "learned_slow_hybrid"
    integration_root = Path(args.integration_root or repo / "results/B5-S50/fm_slow_qc_three_v2")
    qc_dir = Path(args.qc_evaluation_dir or integration_root / "control/evaluations/best_r")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory; original diagnostics are never overwritten")
    marker = read(slow_root / "evaluation.json")
    if not (marker.get("completed") and marker.get("group") == "F" and marker.get("context_seconds") == 50
            and marker.get("target_used_at_inference") is False
            and marker.get("prepared_sha256") == sha(prepared)
            and marker.get("B5_checkpoint_sha256") == info.get("checkpoint_sha256") == profile["fm_sha256"]
            and marker.get("probe_checkpoint_sha256") == profile["slow_sha256"]):
        raise ValueError("The S50-F report does not belong to the stable original B5/prepared pair")
    print("Original FM evaluation: " + str(fm_dir), flush=True)
    print("S50-F saved hybrid: " + str(hybrid_dir), flush=True)
    print("QC from B5-S50 evaluation: " + str(qc_dir), flush=True)
    print("Checking frozen source evaluation files (CPU disk reads, no training)", flush=True)
    checked = {}
    for name, digest in info["evaluation_files"].items():
        path = fm_dir / name
        if sha(path) != digest:
            raise ValueError("Original source evaluation changed: " + str(path))
        checked[str(path)] = digest
    with ExitStack() as task_resources:
        tasks = {}
        for name in ("task1", "task2"):
            tasks[name] = load_task(name, fm_dir, hybrid_dir, qc_dir, True)
            task_resources.callback(tasks[name].close)
        # Full scoring validates cross-task QC, pairs and repeated target consistency.
        baseline = {name: task_r(t, t.fm) for name, t in tasks.items()}
        reference = {name: task_r(t, t.hybrid) for name, t in tasks.items()}
        for name in tasks:
            expected = profile["reference_qc_r"][name + "_r_missing11"]
            if abs(reference[name] - expected) > profile["reference_tolerance"]:
                raise ValueError(f"Reference r mismatch for {name}: {reference[name]} versus {expected}; check paths/QC, not weights")
        output.mkdir(parents=True, exist_ok=True)
        write(output / "preflight.json", {"ready": True, "protocol": PROTOCOL,
              "training_started": False, "optimizer_updates": 0, "models_loaded": False,
              "prepared_info": str(prepared), "prepared_sha256": sha(prepared), "checked_source_sha256": checked,
              "tasks": {k: v.paths for k, v in tasks.items()}, "sampling_inherited": SAMPLING,
              "baseline_r": baseline, "hybrid_r": reference,
              "note": "Historical arrays read; no B5-best training/evaluation directory required"})
        if args.check_only:
            print("Preflight passed; diagnostic computation not requested", flush=True)
            return
        beta_rows = score_betas(tasks, output)
        csv_save(output / "beta_scores.csv", beta_rows)
        csv_save(output / "integration_scores.csv", integration_report(integration_root, tasks))
        components, beats, boundaries, anchors = detail_metrics(tasks, args.leads)
        csv_save(output / "component_record_lead.csv", components)
        csv_save(output / "morphology_record_lead.csv", beats)
        csv_save(output / "boundary_record_lead.csv", boundaries)
        csv_save(output / "qrs_anchor_audit.csv", anchors)
        metadata_fields = {"target_record_id", "subject_id", "lead", "shared_with_task1", "quality",
                           "reference_pair_id", "reference_device"}
        component_fields = [k for k in components[0] if k not in metadata_fields] if components else []
        morphology_fields = [k for k in beats[0] if k.endswith("_median") or k.endswith("_fraction")] if beats else []
        csv_save(output / "component_lead_summary.csv", summarize(components, ("lead",), component_fields))
        csv_save(output / "morphology_lead_summary.csv", summarize(beats, ("lead", "method"), morphology_fields))
        csv_save(output / "boundary_lead_summary.csv", summarize(boundaries, ("lead", "method"),
                 ("centered_error_rms_near_seam_uV", "centered_error_rms_window_middle_uV", "seam_error_one_sample_jump_mae_uV")))
        plots = [] if args.no_plots else plot_examples(tasks, components, args.leads, output, args.plot_records)
        write(output / "plot_manifest.json", plots)
        best = max(beta_rows, key=lambda r: r["mean_r"])
        base_r = (reference["task1"] + reference["task2"]) / 2
        report = f"""# B5-best现有预测诊断（未训练）

    基准组合R={base_r:.8f}，与固定参考通过核对。原始FM、S50-F与QC文件路径见preflight.json。
    未读取模型权重，未执行新推理、未训练、未修改原始预测。数据来自原B5和慢CNN/B5-S50评估目录，不需要B5-best训练目录。

    ## 快慢比例

    beta_scores.csv：beta=1是当前方案。0.75/1.25只改变相对慢起伏强度，不是整体电压校准。
    本次三个验证探针中最高R={best['mean_r']:.8f}（beta={best['beta']}），与基准差={best['mean_r']-base_r:+.8f}。
    这不是已选出的新部署模型，不给每条记录或每导联按真值挑beta。后续需训练集确定规则或独立复核。

    ## 振荡、形态与边界

    - component_lead_summary.csv：慢/快标准差比例、慢r、超过1Hz能量比例、目标匹配的快速残差误差。
    - component_record_lead.csv：逐唯一目标、正常导联明细；诊断选机器配对优先，不代替两个任务正式权重。
    - morphology_lead_summary.csv：相同I锚点下的峰谷幅度、峰值时间、半高宽与局部形态r；与FM单独快残差比较。
    - morphology_record_lead.csv：逐记录中位数及p90。QRS锚点只是I算法候选，不是临床标注，不能仅凭自动指标断言某个病理形态错误。
    - qrs_anchor_audit.csv：检查候选心搏数量；I标差时不做心搏诊断。重点核对极低/极高心率或明显漏检记录。
    - boundary_lead_summary.csv：边界与中部的去固定偏移误差强度。它不是正式评分，不能把真实QRS跳变当边界错误。
    - plots：低/中/高诊断r的代表记录，包含原始与快曲线局部、慢走势和频谱；均值仅在慢走势显示面板去除。
    - integration_scores.csv：已有B5-S50三组best_r的同QC复核，缺失时明确标记，不生成新预测。

    正式beta评分始终使用完整120秒、原目标等权与Task2设备等权、明确导联QC。
    所有目标标签只用于诊断/评分，不作为模型输入。best-lag局部r仅用于时间偏移诊断，不平移正式预测。
    下一步依据结果选择融合比例/局部模块/慢网络改动，不自动更换稳定模型。
    """
        (output / "report.md").write_text(report, encoding="utf-8")
        write(output / "diagnostics.json", {"completed": True, "protocol": PROTOCOL,
              "training_started": False, "optimizer_updates": 0, "models_loaded": False,
              "inference_started": False, "n_unique_diagnostic_targets": len(anchors),
              "morphology_leads": args.leads, "beta_probe_count": len(beta_rows),
              "beta1_mean_r": base_r, "deployment_model_changed": False,
              "sampling_inherited": SAMPLING, "target_used_as_model_input": False,
              "plots": len(plots)})
        print(json.dumps({"completed": True, "output": str(output), "beta1_mean_r": base_r,
                          "training_started": False}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", default="/home/qht/huawei", help="Original DATA/results checkout, not necessarily this code checkout")
    p.add_argument("--prepared-info")
    p.add_argument("--fm-evaluation-dir", help="Original B5 evaluation root containing task1/task2; default from prepared.json")
    p.add_argument("--slow-evaluation-dir", help="S50-F evaluation_best_score root, NOT its checkpoint or learned_slow_hybrid child")
    p.add_argument("--qc-evaluation-dir", help="Existing B5-S50 evaluation with QC masks/metadata; default control/evaluations/best_r")
    p.add_argument("--integration-root", help="Existing three-arm run root; default results/B5-S50/fm_slow_qc_three_v2")
    p.add_argument("--output", required=True)
    p.add_argument("--leads", nargs="+", choices=LEADS[1:], default=["V1", "V3", "V5", "V6"])
    p.add_argument("--plot-records", type=int, default=3)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.plot_records < 1:
        p.error("--plot-records must be positive")
    run(args)


if __name__ == "__main__":
    main()
