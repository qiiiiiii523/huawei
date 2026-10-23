"""Compare actual ABC best/last checkpoints with one matched source G120."""
import argparse
from pathlib import Path
from baselines.B5_best.common import read, write, csv_write, sha
from . import PROTOCOL, GROUPS
from .common import load_study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    p.add_argument("--output")
    a = p.parse_args()
    root = Path(a.run).resolve()
    study = load_study(root / "shared", verify_records=False)
    study_sha = sha(root / "shared/study.json")
    metrics = ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")
    rows = [dict(group="source_G120_original_context", candidate="source_best", epoch=study["source_g120_epoch"],
                 **{k: study["reference"][k] for k in metrics})]
    for group in a.groups:
        done = read(root / group / "completed.json")
        if (done.get("protocol") != PROTOCOL or done.get("group") != group
                or not done.get("completed") or done["huawei_epochs"] != study["training"]["epochs"]):
            raise ValueError(f"{group} has not completed the configured budget")
        for name in ("best_r", "last"):
            report = read(root / group / "evaluations" / name / "evaluation.json")
            if (report.get("protocol") != PROTOCOL or not report.get("completed")
                    or report["population"] != study["population"] or report.get("study_sha256") != study_sha
                    or report["summary"]["group"] != group
                    or report.get("checkpoint_sha256") != sha(root / group / f"{name}.pt")):
                raise ValueError("Checkpoint/evaluation/population changed")
            rows.append(dict(group=group, candidate=name, epoch=report["summary"]["epoch"],
                             **{k: report["summary"][k] for k in metrics}))
    output = Path(a.output or root / "comparison")
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output / "summary.csv", rows)
    write(output / "summary.json", dict(completed=True, protocol=PROTOCOL, rows=rows,
        selection="full120 mean_r", G120_fixed=True, context_policy=study["context_policy"],
        source_g120_sha256=study["source_g120_sha256"], study_sha256=study_sha))
    lines = ["# SlowLead ABC", "", "G120、FM、原context及评分人群固定；不使用C3替换历史路由。", "",
             "|组|候选|轮次|Task1 r|Task2 r|平均r|相对源G120|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['group']}|{r['candidate']}|{r['epoch']}|{r['task1_r']:.6f}|{r['task2_r']:.6f}|{r['mean_r']:.6f}|{r['mean_r']-rows[0]['mean_r']:+.6f}|")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "summary.md")


if __name__ == "__main__":
    main()
