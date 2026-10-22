"""Rank actual ADD/MID checkpoints by the same complete-record mean r."""
import argparse
from pathlib import Path
from baselines.B5_best.common import read, write, csv_write, sha
from .common import load_study


def compare(a):
    root = Path(a.run).resolve()
    study = load_study(root / "shared", verify_records=False)
    fields = ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")
    rows = [dict(arm="source_G120", candidate="source_best", stage="source", epoch=study["source_g120_epoch"],
                 **{k: study["reference"][k] for k in fields})]
    for arm in a.arms:
        group, mode = arm.split("_")
        done = read(root / arm / "completed.json")
        expected_public = study["training"]["public_epochs"] if group == "P" else 0
        if (not done.get("completed") or done.get("group") != group or done.get("fusion_mode") != mode
                or done.get("huawei_epochs") != study["training"]["epochs"]
                or done.get("public_epochs") != expected_public):
            raise ValueError(f"{arm} has not completed the matching configured budget")
        for name in ("best_r", "last"):
            report_path = root / arm / "evaluations" / name / "evaluation.json"
            if not report_path.is_file():
                raise FileNotFoundError(f"Missing {report_path}; rerun this arm launcher to finish evaluation")
            r = read(report_path)
            if (not r.get("completed") or r.get("fusion_mode") != mode or r["population"] != study["population"]
                    or r.get("study_sha256") != sha(root / "shared/study.json")
                    or r.get("checkpoint_sha256") != sha(root / arm / f"{name}.pt")):
                raise ValueError(f"{arm}/{name}: checkpoint, score population or study mismatch")
            rows.append(dict(arm=arm, candidate=name, stage=r["checkpoint_stage"], epoch=r["summary"]["epoch"],
                             **{k: r["summary"][k] for k in fields}))
    out = Path(a.output or root / "comparison")
    out.mkdir(parents=True, exist_ok=False)
    csv_write(out / "summary.csv", rows)
    write(out / "summary.json", dict(completed=True, rows=rows, selection="(r1+r2)/2",
          source_g120_sha256=study["source_g120_sha256"], context_fixed=True, beta=.75))
    lines = ["# 中间特征融合：实际权重复评", "", "相同目标QC、设备等权、完整120秒、固定FM/context与Heun16/K16。", "",
             "|组|候选|阶段/轮次|Task1 r|Task2 r|平均r|相对原G120|", "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['arm']}|{r['candidate']}|{r['stage']}/{r['epoch']}|{r['task1_r']:.6f}|{r['task2_r']:.6f}|{r['mean_r']:.6f}|{r['mean_r']-rows[0]['mean_r']:+.6f}|")
    lines += ["", "MID需超过同训练路线的ADD，才是融合有额外收益；P额外使用公开数据及计算，不能归因为结构单独收益。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "summary.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--arms", nargs="+", choices=("H_add", "H_mid", "P_add", "P_mid"), default=["H_add", "H_mid"])
    p.add_argument("--output")
    compare(p.parse_args())


if __name__ == "__main__":
    main()
