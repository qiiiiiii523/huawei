"""Paired Final/L120/C120 score table and explicit go/no-go interpretation."""
import argparse
from pathlib import Path
from baselines.B5_best.common import read, write, csv_write
from .common import load_study


def summarize(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    study = load_study(root / "shared", verify_records=False)
    output.mkdir(parents=True, exist_ok=False)
    g0 = {"group": "G0", "epoch": 0, **{k: study["reference"][k] for k in
          ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")}, "delta_G0": 0.}
    rows = [{"candidate": "G0", **g0}]
    selected = {}
    for group in ("L120", "C120"):
        report = read(root / group / "completed.json")
        if report.get("completed") is not True:
            raise ValueError("Both arms must complete")
        # Actual checkpoint evaluations performed by the launcher, not a stale
        # historical score copied from the training ledger.
        for name in ("best", "last"):
            actual = read(root / "verification" / group / name / "evaluation.json")["summary"]
            if actual["group"] != group:
                raise ValueError("Mismatched checkpoint arm")
            rows.append({"candidate": group + "_" + name, **{k: actual[k] for k in g0}})
            if name == "best":
                if abs(actual["mean_r"] - report["best_r"]) > 1e-7:
                    raise ValueError("Best checkpoint did not reproduce selected score")
                selected[group] = actual
    delta0 = selected["C120"]["mean_r"] - g0["mean_r"]
    delta50 = selected["C120"]["mean_r"] - selected["L120"]["mean_r"]
    passed = delta0 > 1e-7 and delta50 > 1e-7
    csv_write(output / "summary.csv", rows)
    write(output / "summary.json", {"completed": True, "summary": rows,
        "C120_minus_Final": delta0, "C120_minus_L120": delta50, "numerical_comparison_tolerance": 1e-7,
        "strictly_beats_both": passed, "training_seed": study["training"]["seed"],
        "requires_training_seed_replication": passed, "not_a_statistical_significance_test": True})
    lines = ["# B5-final：整段CNN120感受野实验", "", "|候选|epoch|Task1 r|机器r|体脂秤r|Task2 r|平均r|ΔFinal|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['candidate']}|{r['epoch']}|" + "|".join(f"{r[k]:.8f}" for k in
            ("task1_r", "machine_r", "body_r", "task2_r", "mean_r", "delta_G0")) + "|")
    lines += ["", f"C120-best − Final：{delta0:+.8f}；C120-best − L120-best：{delta50:+.8f}。",
        "", "初筛通过，补训练seed并检查逐患者慢走势。" if passed else
        "当前未同时超过Final和L120，不作为长感受野升级；结合慢走势诊断决定是否停止本设计。",
        "", "以上1e-7只排除数值误差，不代表统计显著。best与last均为实际权重复评。",
        "各验证目录提供pair_metrics.csv和slow_diagnostics.csv；按subject_id汇总配对，不把窗口当独立患者。"]
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
