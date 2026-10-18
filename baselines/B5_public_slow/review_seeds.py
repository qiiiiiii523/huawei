"""Compare M/P training-seed results using the ORIGINAL coarse checkpoint grid only."""
import argparse
from pathlib import Path

from baselines.B5_best.common import read, write, csv_write
from .common import empty_output


def review(reference_root, current_root, output):
    roots = (Path(reference_root), Path(current_root))
    studies = [read(root / "shared/study.json") for root in roots]
    reports = [read(roots[0] / "comparison/summary.json"), read(roots[1] / "comparison_coarse/summary.json")]
    for key in ("bundle_sha256", "score_protocol", "sampling", "beta", "residual_weight", "F_width_native", "combo_weight",
                "mean_weight", "selection", "stage1_handoff", "public_patient_balanced", "M_batch"):
        if studies[0][key] != studies[1][key]:
            raise ValueError("Seed comparison changes another setting: " + key)
    for source in ("huawei", "public"):
        if studies[0][source]["sha256"] != studies[1][source]["sha256"]:
            raise ValueError("Different frozen data/cache: " + source)
    for key in ("stage1_updates", "stage2_updates", "validate_every", "batch_records", "learning_rate", "weight_decay", "gradient_clip"):
        if studies[0]["training"][key] != studies[1]["training"][key]:
            raise ValueError("Seed comparison changes training: " + key)
    seeds = [study["training"]["seed"] for study in studies]
    if seeds[0] == seeds[1]:
        raise ValueError("Training seeds must differ")
    for report, seed in zip(reports, seeds):
        if report.get("checkpoint", "best.pt") != "best.pt" or report["training_seed"] != seed:
            raise ValueError("Do not mix dense/stage1 candidates into original-grid seed verification")
    for key in ("task1_r_missing11", "task2_r_missing11"):
        if abs(reports[0]["baseline"][key] - reports[1]["baseline"][key]) > 2e-5:
            raise ValueError("Initial predictions/score scope differ")
    rows = []
    for report, seed in zip(reports, seeds):
        available = {r["group"]: r for r in report["rows"]}
        for group in ("M", "P"):
            row = available[group]
            rows.append({"training_seed": seed, **row})
    means = {group: sum(r["main_correlation"] for r in rows if r["group"] == group) / 2 for group in ("M", "P")}
    differences = [{"training_seed": seed,
                    "P_minus_M": next(r["main_correlation"] for r in rows if r["training_seed"] == seed and r["group"] == "P") -
                                 next(r["main_correlation"] for r in rows if r["training_seed"] == seed and r["group"] == "M")}
                   for seed in seeds]
    output = empty_output(output)
    csv_write(output / "seed_comparison.csv", rows)
    write(output / "seed_comparison.json", {"rows": rows, "mean_R_by_group": means, "paired_differences": differences,
          "selection_scope": "original_coarse_only", "fixed_FM_noise_seed": studies[0]["sampling"]["seed"],
          "independent_test_result": False, "note": "Two training seeds, no formal significance claim; dense candidates reported separately"})
    lines = ["# M/P训练seed复核", "", "仅使用原390步选模的best.pt；加密验证与阶段一候选不混入。", "",
             "|训练seed|组|Task1 r|Task2 r|平均r|", "|---:|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['training_seed']}|{r['group']}|{r['task1_r_missing11']:.6f}|{r['task2_r_missing11']:.6f}|{r['main_correlation']:.6f}|")
    lines += ["", f"两seed平均：M={means['M']:.6f}，P={means['P']:.6f}。",
              "", "这是训练随机性复核，不是独立测试，也不是新增第二seed的C对照。"]
    (output / "seed_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"mean_R_by_group": means, "paired_differences": differences}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    review(a.reference_root, a.root, a.output)


if __name__ == "__main__":
    main()
