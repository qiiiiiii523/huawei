"""Summarize finished C/M/P, verifying a shared initial score/study/QC scope."""
import argparse
from pathlib import Path

import torch

from baselines.B5_best.common import read, write, csv_write
from .common import empty_output
from . import GROUPS, PROTOCOL


def compare(root, output, groups=GROUPS, checkpoint="best.pt"):
    root = Path(root)
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError("Choose distinct C/M/P groups")
    if checkpoint not in ("best.pt", "dense_best.pt", "stage1_best.pt"):
        raise ValueError("Unknown candidate checkpoint")
    rows, identity, baseline, training_seed = [], None, None, None
    for group in groups:
        folder = root / group
        completed = read(folder / "completed.json")
        init = read(folder / "initialization.json")
        cp = torch.load(folder / checkpoint, map_location="cpu", weights_only=True)
        if not completed.get("completed") or cp.get("protocol") != PROTOCOL or cp.get("group") != group:
            raise ValueError("Missing/unfinished/different group: " + group)
        seed = int(init.get("training_seed", 42))
        key = (cp["study_sha256"], init["slow_state_sha256"], completed["score_protocol"], seed)
        if identity is not None and key != identity:
            raise ValueError("Groups differ in study/initializer/score scope")
        identity = key
        training_seed = seed
        if baseline is None:
            baseline = init["initial_score"]
        for field in ("task1_r_missing11", "task2_r_missing11"):
            if abs(init["initial_score"][field] - baseline[field]) > 2e-5:
                raise ValueError("Initial scores differ between arms")
        score = cp["validation_score"]
        rows.append({"group": group, "selected_stage": cp["stage"], "selected_step": cp["step"], **score,
                     "delta_vs_initial": score["main_correlation"] - baseline["main_correlation"]})
    output = empty_output(output)
    csv_write(output / "summary.csv", rows)
    write(output / "summary.json", {"baseline": baseline, "rows": rows,
          "best_group": max(rows, key=lambda r: r["main_correlation"])["group"],
          "training_seed": training_seed, "checkpoint": checkpoint,
          "selection_scope": "original_coarse" if checkpoint == "best.pt" else "extra_diagnostic_candidates",
          "independent_test_result": False})
    lines = ["# 慢CNN公开数据扩充结果", "", "同一初始化、固定control100、β=0.75；仅比较本轮验证结果。", "",
             "|组|选中阶段/更新|Task1 r|Task2 r|平均r|相对起点|", "|---|---|---:|---:|---:|---:|",
             f"|起点|0/0|{baseline['task1_r_missing11']:.6f}|{baseline['task2_r_missing11']:.6f}|{baseline['main_correlation']:.6f}|0|"]
    for r in rows:
        lines.append(f"|{r['group']}|{r['selected_stage']}/{r['selected_step']}|{r['task1_r_missing11']:.6f}|"
                     f"{r['task2_r_missing11']:.6f}|{r['main_correlation']:.6f}|{r['delta_vs_initial']:+.6f}|")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(rows, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    p.add_argument("--checkpoint", choices=("best.pt", "dense_best.pt", "stage1_best.pt"), default="best.pt")
    a = p.parse_args()
    compare(a.root, a.output, a.groups, a.checkpoint)


if __name__ == "__main__":
    main()
