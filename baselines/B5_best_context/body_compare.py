"""Read completed seed studies; reject changed base data/settings or unequal paired budgets."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baselines.B5_best.common import read, write, csv_write


def load_study(path):
    root = Path(path).resolve()
    report, weights = read(root / "study.json"), read(root / "body_checkpoints.json")
    if report.get("completed") is not True:
        raise ValueError("Use a completed body study")
    configs = {group: read(Path(weights[group]).parent / "config.json") for group in ("context", "no_context")}
    context, control = configs["context"], configs["no_context"]
    a, b = dict(context), dict(control)
    a.pop("group"); b.pop("group")
    if a != b or context["group"] != "context" or control["group"] != "no_context":
        raise ValueError("Within-seed body groups changed initialization/order/budget/cache/settings")
    if weights["initial_weights_sha256"] != context["initial_weights_sha256"]:
        raise ValueError("Study/training initial weights mismatch")
    with (root / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        summary = list(csv.DictReader(handle))
    rows = {r["group"]: {k: v if k == "group" else float(v) for k, v in r.items()} for r in summary}
    if len(rows) != 9 or len(summary) != 9:
        raise ValueError("Expected all nine body route/ablation groups")
    return report, weights, configs, rows


def compare(old, new, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty comparison directory")
    before, after = load_study(old), load_study(new)
    left, right = before[2]["context"], after[2]["context"]
    if left["seed"] == right["seed"]:
        raise ValueError("This comparison requires different body training seeds")
    for key in ("base_checkpoint_sha256", "population"):
        if before[0][key] != after[0][key]:
            raise ValueError("Studies changed base predictions/target population/QC: " + key)
    aa, bb = dict(left), dict(right)
    for cfg in (aa, bb):
        cfg.pop("seed"); cfg.pop("initial_weights_sha256")
    if aa != bb:
        raise ValueError("Across-seed training changed more than seed/initial weights")
    for group in ("baseline", "d6_only"):
        if before[3][group] != after[3][group]:
            raise ValueError("Base/d6 predictions must stay identical between body training seeds")
    rows, all_groups = [], []
    for source, loaded in ((old, before), (new, after)):
        configs, values = loaded[2], loaded[3]
        seed = configs["context"]["seed"]
        context, control, base, d6, both = (values[k] for k in ("body_only", "body_no_context", "baseline", "d6_only", "both"))
        rows.append({"body_training_seed": seed, "baseline_mean_r": base["mean_r"],
            "d6_only_mean_r": d6["mean_r"], "both_mean_r": both["mean_r"],
            "body_gain_vs_base_mean_r": context["mean_r"] - base["mean_r"],
            "body_gain_over_d6_mean_r": both["mean_r"] - d6["mean_r"],
            "correct_vs_no_context_mean_r": context["mean_r"] - control["mean_r"],
            "correct_vs_disabled_mean_r": context["mean_r"] - values["body_disabled"]["mean_r"],
            **{f"correct_vs_shuffled_{s}_mean_r": context["mean_r"] - values[f"body_shuffled_{s}"]["mean_r"] for s in (101, 202, 303)}})
        for group, row in values.items():
            all_groups.append({"body_training_seed": seed, **row})
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "seed_comparison.csv", rows)
    csv_write(output / "all_groups.csv", all_groups)
    write(output / "comparison.json", {"completed": True, "training_started": False,
        "before_study": str(Path(old).resolve()), "after_study": str(Path(new).resolve()),
        "only_body_training_seed_changed": True, "same_base_cache_and_QC": True,
        "rows": rows, "training_configs": {"before": before[2], "after": after[2]}})
    lines = ["# Body context 训练seed复验", "", "仅body训练seed变化；FM采样仍为seed42、Heun16/K16、β=0.75。", "",
        "|训练seed|仅d6平均r|d6＋body平均r|body额外增量|正确相对无context增量|",
        "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['body_training_seed']}|{row['d6_only_mean_r']:.9f}|{row['both_mean_r']:.9f}|{row['body_gain_over_d6_mean_r']:+.9f}|{row['correct_vs_no_context_mean_r']:+.9f}|")
    lines += ["", "同时检查两次训练的正确/关闭/三个错配结果、逐记录升降分布。",
        "两个训练seed不能代替患者独立留出验证；不要只挑更高的seed来证明稳定性。"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"completed": True, "seeds": [left["seed"], right["seed"]], "output": str(output)}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before-study", required=True)
    p.add_argument("--after-study", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    compare(a.before_study, a.after_study, a.output)


if __name__ == "__main__":
    main()
