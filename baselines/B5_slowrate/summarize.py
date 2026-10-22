"""Compare actual checkpoints; patient contrasts are diagnostics, not fusion proof."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

from baselines.B5_best.common import read, write, csv_write
from . import RATES
from .common import load_study

FIELDS = ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")


def patient_scores(path):
    grouped = defaultdict(list)
    identities = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identity = tuple(row[k] for k in ("task_id", "subject_id", "target_record_id", "pair_id", "input_type"))
            if identity in identities:
                raise ValueError("Duplicate diagnostic pair")
            identities.add(identity)
            if row["r_missing11"]:
                grouped[tuple(row[k] for k in ("task_id", "subject_id", "input_type"))].append(float(row["r_missing11"]))
    return {k: float(np.mean(v)) for k, v in grouped.items()}, identities


def summarize(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    study = load_study(root / "shared", verify_records=False)
    baseline = {k: study["reference"][k] for k in FIELDS}
    rows = [{"candidate": "G0", "sampling_hz": None, "epoch": 0, **baseline, "delta_G0": 0.}]
    selected, patients, identities, updates = {}, {}, {}, set()
    for group, hz in RATES.items():
        completed = read(root / group / "completed.json")
        if not completed.get("completed") or completed["sampling_hz"] != hz:
            raise ValueError("All three matched arms must complete")
        updates.add(completed["updates"])
        for name in ("best", "last"):
            folder = root / "verification" / group / name
            report = read(folder / "evaluation.json")
            actual = report["summary"]
            if (not report.get("completed") or report.get("sampling_hz") != hz or actual["group"] != group
                    or report["population"] != study["population"]):
                raise ValueError("Actual checkpoint reevaluation mismatch")
            rows.append({"candidate": group + "_" + name, "sampling_hz": hz, "epoch": actual["epoch"],
                         **{k: actual[k] for k in FIELDS}, "delta_G0": actual["mean_r"] - baseline["mean_r"]})
            if name == "best":
                if abs(actual["mean_r"] - completed["best_r"]) > 1e-7:
                    raise ValueError("Best checkpoint did not reproduce selected score")
                selected[group] = actual
                patients[group], identities[group] = patient_scores(folder / "pair_metrics.csv")
    if len(updates) != 1:
        raise ValueError("Matched arms used different update budgets")
    ref_patients = patients["R5"]
    diagnostic = []
    for group in ("R2p5", "R10"):
        if identities[group] != identities["R5"] or patients[group].keys() != ref_patients.keys():
            raise ValueError("Patient diagnostic support differs across rates")
        for key in sorted(ref_patients):
            diagnostic.append({"group": group, "task_id": key[0], "subject_id": key[1], "input_type": key[2],
                "rate_r": patients[group][key], "R5_r": ref_patients[key],
                "delta_R5": patients[group][key] - ref_patients[key]})
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output / "summary.csv", rows)
    if diagnostic:
        csv_write(output / "patient_rate_contrasts.csv", diagnostic)
    winner = max(selected, key=lambda g: selected[g]["mean_r"])
    differences = {g: selected[g]["mean_r"] - selected["R5"]["mean_r"] for g in RATES}
    write(output / "summary.json", {"completed": True, "summary": rows, "winner_this_seed": winner,
        "delta_matched_R5": differences, "training_seed": study["training"]["seed"],
        "updates_per_arm": next(iter(updates)), "rates_hz": RATES,
        "numerical_tolerance": 1e-7, "not_a_statistical_significance_test": True,
        "rate_difference_does_not_establish_multirate_benefit": True,
        "patient_diagnostic_means_are_not_official_score_weights": True})
    lines = ["# G120：2.5 / 5 / 10 Hz 匹配对照", "",
        "三组均从原Final出发，新增支路同参数、同初始化、同更新预算；不是从已训练G120继续。",
        "5 Hz是本轮重训对照，历史最好G120只作背景，不能替代匹配对照。", "",
        "|候选|Hz|epoch|Task1 r|机器r|体脂秤r|Task2 r|平均r|ΔG0|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['candidate']}|{row['sampling_hz'] or '—'}|{row['epoch']}|" +
                     "|".join(f"{row[k]:.8f}" for k in (*FIELDS, "delta_G0")) + "|")
    lines += ["", f"本seed最佳：{winner}；R2p5-R5：{differences['R2p5']:+.8f}；R10-R5：{differences['R10']:+.8f}。",
        "", "采样率同时改变输入抗混叠带宽、token密度及输出插值分辨率；计算量不同，不是纯带宽因果实验。",
        "先复核训练seed及同预算下的当前最佳成绩，再决定是否替换采样率。",
        "不同Hz有差异不代表多Hz融合有效。需先有互补线索，再用实际组合超过最强单Hz验证。",
        "patient_rate_contrasts.csv按患者/任务/设备聚合，仅作诊断，不是官方加权分数或独立患者检验。",
        "分尺度趋势见各verification目录的slow_diagnostics.csv；1e-7仅排除数值误差。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    summarize(args.run, args.output)


if __name__ == "__main__":
    main()
