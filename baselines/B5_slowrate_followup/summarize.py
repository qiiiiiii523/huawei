"""Show follow-up gains against source and the relevant control, including epoch0."""
import argparse
from pathlib import Path
from baselines.B5_best.common import read, write, csv_write
from .common import load_study

FIELDS = ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")


def summarize(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    study = load_study(root / "shared", verify_records=False)
    ref = study["reference"]
    rows = [{"candidate": "source_" + study["reference_group"], "epoch": 0,
             **{k: ref[k] for k in FIELDS}, "delta_reference": 0.}]
    selected, updates = {}, set()
    for group in study["groups"]:
        completed = read(root / group / "completed.json")
        if not completed.get("completed"):
            raise ValueError("All requested follow-ups must complete")
        updates.add(completed["updates"])
        initial = read(root / group / "validation/epoch_0000/evaluation.json")["summary"]
        rows.append({"candidate": group + "_initial", "epoch": 0, **{k: initial[k] for k in FIELDS},
                     "delta_reference": initial["mean_r"] - ref["mean_r"]})
        for name in ("best", "last"):
            report = read(root / "verification" / group / name / "evaluation.json")
            row = report["summary"]
            if not report.get("completed") or row["group"] != group or report["population"] != study["population"]:
                raise ValueError("Actual follow-up checkpoint evaluation mismatch")
            rows.append({"candidate": group + "_" + name, "epoch": row["epoch"],
                         **{k: row[k] for k in FIELDS}, "delta_reference": row["mean_r"] - ref["mean_r"]})
            if name == "best":
                if abs(row["mean_r"] - completed["best_r"]) > 1e-7:
                    raise ValueError("Selected best score did not reproduce")
                selected[group] = row["mean_r"]
    if len(updates) > 1:
        raise ValueError("Follow-up arms used different update budgets")
    contrasts = {}
    for a, b in (("FUSION", "FUSION_SAME"), ("FUSION", "CONT"), ("PTAIL", "CONT"),
                 ("PCTX", "PTAIL"), ("PCTX_RF", "PCTX"), ("PCTX_RF", "PTAIL")):
        if a in selected and b in selected:
            contrasts[a + "_minus_" + b] = selected[a] - selected[b]
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output / "summary.csv", rows)
    write(output / "summary.json", {"completed": True, "summary": rows, "contrasts": contrasts,
                                   "reference_group": study["reference_group"], "no_auto_promotion": True})
    lines = ["# 采样率后续实验：融合与P-S50/S100适配", "",
             "|候选|新增epoch|Task1 r|机器r|体脂秤r|Task2 r|平均r|相对源单Hz|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['candidate']}|{row['epoch']}|" +
                     "|".join(f"{row[k]:.8f}" for k in (*FIELDS, "delta_reference")) + "|")
    lines += ["", *[f"{key}: {value:+.8f}" for key, value in contrasts.items()], "",
              f"PTAIL为S50；PCTX为S{study['p_context_seconds']}原感受野；PCTX_RF为同窗口、约{study['P_designs']['PCTX_RF']['rf_seconds']:.2f}秒感受野。",
              "PCTX改变原CNN窗口执行，初始成绩可能变化；同时报告initial，不能假装第0步仍等于源模型。",
              "融合需超过源单Hz和FUSION_SAME；扩大感受野需看PCTX_RF相对PCTX及PTAIL。",
              "所有分数均来自实际权重。1e-7只用于数值复现；需训练seed/独立患者复核，不自动替换当前最佳。"]
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
