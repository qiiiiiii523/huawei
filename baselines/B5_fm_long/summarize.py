"""Join paired result identities and verify independent checkpoint/package evaluations."""
import argparse
from pathlib import Path

from baselines.B5_best.common import read, write, csv_write
from . import PROTOCOL, MODES


def collect(root):
    root = Path(root)
    paths = list(sorted((root / "screen").glob("FM*/evaluation.json")))
    paths += list(sorted((root / "verification").glob("FM*/*/evaluation.json")))
    rows, study_sha, population = [], None, None
    for path in paths:
        report = read(path)
        if report.get("protocol") != PROTOCOL or report.get("completed") is not True or report.get("context") != "disabled":
            raise ValueError("Incomplete or incompatible evaluation: " + str(path))
        if study_sha is None:
            study_sha, population = report["study_sha256"], report["population"]
        if report["study_sha256"] != study_sha or report["population"] != population:
            raise ValueError("Cannot compare different patient/QC/study populations")
        for mode in MODES:
            rows.append({"evaluation": path.parent.relative_to(root).as_posix(), **report["summary"][mode]})
    if not rows:
        raise ValueError("No completed screening/evaluation reports")
    screening = {r["window_seconds"]: r for r in rows if r["mode"] == "hybrid" and r["label"] == "initial"}
    if set(screening) != {10, 40}:
        raise ValueError("Screening must include BOTH FM10 and FM40")
    trained = [(root / f"FM{seconds}").exists() for seconds in (10, 40)]
    if any(trained) and not all(trained):
        raise ValueError("Paired comparison requires BOTH completed training groups")
    for seconds in (10, 40):
        folder = root / f"FM{seconds}"
        if not folder.exists():
            continue
        completed = read(folder / "completed.json")
        if completed.get("completed") is not True:
            raise ValueError("Do not summarize interrupted training as completed")
        if completed["updates"] != read(root / "shared/study.json")["experiment"]["training"]["updates"]:
            raise ValueError("Training budget changed")
        for mode in MODES:
            expected = completed["best"][mode]
            for kind in ("best_" + mode, "package_" + mode):
                name = f"verification/FM{seconds}/{kind}"
                found = [r for r in rows if r["evaluation"] == name and r["mode"] == mode]
                if len(found) != 1 or abs(found[0]["mean_r"] - expected["mean_r"]) > 1e-7:
                    raise ValueError("Candidate/package does not reproduce selected best: " + name)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = collect(args.run)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a new comparison directory")
    out.mkdir(parents=True)
    csv_write(out / "summary.csv", rows)
    write(out / "comparison.json", {"protocol": PROTOCOL, "completed": True, "rows": rows,
          "context": "disabled", "independent_candidate_and_package_checks_passed":
          all((Path(args.run) / f"FM{s}").exists() for s in (10, 40))})
    lines = ["# FM10 与 FM40 完整记录结果", "", "设备 context 关闭；慢 CNN 固定。raw 与 hybrid 来自同一次 FM 生成。", "",
             "| 评估 | 窗口 | 更新 | 输出 | Task1 r | Task2 r | 平均 r |", "|---|---:|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['evaluation']} | {row['window_seconds']} | {row['update']} | {row['mode']} | "
                     f"{row['task1_r']:.8f} | {row['task2_r']:.8f} | {row['mean_r']:.8f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
