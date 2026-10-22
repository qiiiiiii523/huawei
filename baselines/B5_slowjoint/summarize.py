"""Compare source, H and P using completed actual-checkpoint reevaluations."""
import argparse
from pathlib import Path
from baselines.B5_best.common import read, write, csv_write, sha
from .common import load_study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Experiment root containing shared/, H/ and P/")
    p.add_argument("--groups", nargs="+", choices=("H", "P"), default=["H", "P"])
    p.add_argument("--output")
    a = p.parse_args()
    root = Path(a.run).resolve()
    study = load_study(root / "shared", verify_records=False)
    rows = [dict(group="source_G120", candidate="source_best", stage="source", epoch=study["source_g120_epoch"],
                 **{k: study["reference"][k] for k in ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")})]
    for group in a.groups:
        done = read(root / group / "completed.json")
        if not done.get("completed") or done["huawei_epochs"] != study["training"]["epochs"]:
            raise ValueError(f"{group} has not completed the configured Huawei budget")
        for name in ("best_r", "last"):
            path = root / group / "evaluations" / name / "evaluation.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; run the arm launcher again to finish evaluation (no retraining)")
            report = read(path)
            if (not report.get("completed") or report["population"] != study["population"]
                    or report.get("checkpoint_sha256") != sha(root / group / f"{name}.pt")):
                raise ValueError("Checkpoint/evaluation/population changed")
            cp_stage = report["checkpoint_stage"]
            rows.append(dict(group=group, candidate=name, stage=cp_stage,
                **{k: report["summary"][k] for k in ("epoch", "task1_r", "machine_r", "body_r", "task2_r", "mean_r")}))
    output = Path(a.output or root / "comparison")
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output / "summary.csv", rows)
    write(output / "summary.json", dict(completed=True, selection="(r1+r2)/2", rows=rows,
          source_g120_sha256=study["source_g120_sha256"], public_stage_extra_compute=True))
    lines = ["# 局部慢CNN＋G120共同训练结果", "", "FM、context、β=.75、Heun16/K16及评分集合固定。RMSE不参与选模。", "",
             "|组|候选|阶段/轮次|Task1 r|Task2 r|平均r|相对原G120|", "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['group']}|{r['candidate']}|{r['stage']}/{r['epoch']}|{r['task1_r']:.6f}|{r['task2_r']:.6f}|{r['mean_r']:.6f}|{r['mean_r']-rows[0]['mean_r']:+.6f}|")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "summary.md")


if __name__ == "__main__":
    main()
