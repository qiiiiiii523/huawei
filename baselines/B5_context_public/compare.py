"""Completed-arm verification and fixed-route competition summary."""
import argparse
from pathlib import Path
from baselines.B5_best.common import read, write, csv_write, sha
from .sampling import SAMPLING, validate_machine_references
from .data import load_study


def compare(root):
    root = Path(root); study = load_study(root / "shared")
    refs = read(root / "references/references.json")
    if refs["study_sha256"] != sha(root / "shared/study.json"): raise ValueError("Reference study mismatch")
    validate_machine_references(refs["reports"])
    stable = refs["reports"]["stable_machine"]
    rows = [dict(arm="stable_paired", candidate="reference", machine_r=stable["machine_r"], mean_r=stable["mean_r"])]
    updates = None
    for arm in ("control", "context_public"):
        done = read(root / arm / "completed.json")
        if (not done.get("completed") or done["sampling"] != SAMPLING or done["study_sha256"] != sha(root / "shared/study.json")
            or done["public_epochs"] != study["training"]["public_epochs"] or done["huawei_epochs"] != study["training"]["huawei_epochs"]):
            raise ValueError("Both arms must complete matched budgets and study")
        if updates is not None and done["updates"] != updates:
            raise ValueError("Arms completed unequal optimizer update budgets")
        updates = done["updates"]
        for candidate in ("best_r", "last"):
            d = read(root / arm / "evaluations" / candidate / "evaluation.json")
            if (d.get("sampling") != SAMPLING or not d.get("completed")
                    or d.get("population_sha256") != stable["population_sha256"]
                    or d.get("record_keys_sha256") != stable["record_keys_sha256"]):
                raise ValueError("Candidate sampling/population/record keys differ from paired reference")
            rows.append(dict(arm=arm, candidate=candidate, machine_r=d["machine_r"], mean_r=d["mean_r"]))
    out = root / "comparison"; out.mkdir(exist_ok=True)
    csv_write(out / "summary.csv", rows); write(out / "summary.json", rows)
    text = "# ContextPublic完整组合比较\n\n|组|权重|机器r|最终平均r|\n|---|---|---:|---:|\n"
    for r in rows: text += f"|{r['arm']}|{r['candidate']}|{r['machine_r']:.8f}|{r['mean_r']:.8f}|\n"
    text += "\nTask1和体脂秤固定，机器r提高δ对应最终平均r提高δ/4。历史诊断见context_public/diagnostics。\n"
    (out / "summary.md").write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); a = p.parse_args(); compare(a.run)


if __name__ == "__main__": main()
