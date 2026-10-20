"""Compare actual best/last/package re-evaluations against F0 and matched F1."""
import argparse
from pathlib import Path
from baselines.B5_best.common import read, write, csv_write
from . import PROTOCOL, ROUTES
from .common import load_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Parent containing shared, F0, F1, D1 and verification")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.run), Path(args.output)
    study = load_study(root / "shared", verify_records=False)
    if output.exists():
        raise FileExistsError("Use a new summary directory")
    f0 = read(root / "F0/evaluation.json")
    results, selected = [("F0", f0)], {}
    selection = study["experiment"]["training"]["selection_route"]
    for group in ("F1", "D1"):
        training = read(root / group / "completed.json")
        if training.get("completed") is not True or training["group"] != group:
            raise ValueError("Both training arms must complete")
        for candidate in ("best", "last"):
            report = read(root / "verification" / group / candidate / "evaluation.json")
            if report["group"] != group or report["population"] != study["population"]:
                raise ValueError("Wrong candidate arm or validation population")
            results.append((group + "_" + candidate, report))
            if candidate == "best":
                if abs(report["summary"][selection]["mean_r"] - training["best_score"]) > 1e-7:
                    raise ValueError("Best weight did not reproduce the training selection")
                selected[group] = report
        packaged = read(root / "verification" / group / "package" / "evaluation.json")
        if (packaged["inference_path"] != "portable_package_visible_inputs_only"
                or packaged["group"] != group or packaged["population"] != study["population"]
                or packaged["fast_state_sha256"] != selected[group]["fast_state_sha256"]):
            raise ValueError("Package did not exercise the selected model's real deployment path")
        for route in ROUTES:
            for key in ("task1_r", "task2_r", "machine_r", "body_r", "mean_r"):
                if abs(packaged["summary"][route][key] - selected[group]["summary"][route][key]) > 1e-7:
                    raise ValueError("Portable inference does not reproduce cached-component evaluation")
        results.append((group + "_package", packaged))
    rows = []
    for name, report in results:
        for route in ROUTES:
            row = report["summary"][route]
            rows.append({"candidate": name, **row,
                         "delta_F0_same_route": row["mean_r"] - f0["summary"][route]["mean_r"]})
    delta = {route: {"D1_minus_F0": selected["D1"]["summary"][route]["mean_r"] - f0["summary"][route]["mean_r"],
                     "D1_minus_F1": selected["D1"]["summary"][route]["mean_r"] - selected["F1"]["summary"][route]["mean_r"]}
             for route in ROUTES}
    passed = all(value > 1e-7 for value in delta[selection].values())
    output.mkdir(parents=True)
    csv_write(output / "summary.csv", rows)
    write(output / "summary.json", {"protocol": PROTOCOL, "completed": True, "summary": rows,
        "deltas": delta, "selection_route": selection, "D1_beats_F0_and_F1": passed,
        "package_reproduced": True, "training_seed": study["experiment"]["training"]["seed"],
        "numerical_tolerance": 1e-7, "not_statistical_significance": True})
    lines = ["# B5-final：直接回归快分量实验", "", f"预先固定选模路由：{selection}。", "",
        "|候选|路由|epoch|Task1 r|机器r|体脂秤r|Task2 r|平均r|Δ同路由F0|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['candidate']}|{row['route']}|{row['epoch']}|" + "|".join(f"{row[k]:.8f}" for k in
            ("task1_r", "machine_r", "body_r", "task2_r", "mean_r", "delta_F0_same_route")) + "|")
    lines.extend(["", "D1同时超过F0与F1，值得补训练seed复核。" if passed else
                  "D1未同时超过F0与F1，本轮不作为已确认升级。", "",
        "baseline判断快预测本身；d6_only判断机器修正；both包含冻结body迁移。不要混用不同路由得分。",
        "package列重新从原始可见I预测P/I，未把缓存慢真值或目标波形传给预测器。",
        "每个验证目录提供逐导联报告及pair_metrics.csv；后续按患者汇总配对不确定性。",
        "1e-7仅用于排除数值误差；单个训练seed和反复使用的验证集不能证明稳定泛化，更不保证+0.1。"])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
