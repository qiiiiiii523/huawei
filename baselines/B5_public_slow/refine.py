"""Reuse verified historical P handoffs; paired stage2 LR trials with a common dense grid."""
import argparse
import copy
from pathlib import Path
import shutil

import torch

from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import read, write, sha, state_sha, csv_write
from . import PROTOCOL
from .common import empty_output, model_from_bundle, load_study
from .prepare import prepare

ARMS = {"lr1e4": 1e-4, "lr3e5": 3e-5}


def historical_handoff(root, seed):
    """Read-only import: old code hash is recorded, never rewritten to pass load_study."""
    root = Path(root).resolve()
    path = root / "shared/study.json"
    study = read(path)
    if (study.get("protocol") != PROTOCOL or study.get("score_protocol") != SCORE_PROTOCOL or
            study["training"]["seed"] != seed or study["training"]["stage1_updates"] != 1560 or
            study["training"]["stage2_updates"] < 390 or study["training"]["learning_rate"] != 1e-4 or
            study.get("stage2_restart") or not study.get("code_sha256")):
        raise ValueError("Need the completed original P seed42/43 two-stage study")
    bundle_path = path.parent / "components.pt"
    if sha(bundle_path) != study["bundle_sha256"]:
        raise ValueError("Historical initializer changed")
    bundle = load_bundle(bundle_path)
    for item in [study["huawei"], study["public"], *study["external_checks"]]:
        filename = item.get("manifest", item.get("path"))
        if sha(filename) != item["sha256"]:
            raise ValueError("Historical source manifest changed: " + filename)
    cp_path = root / "P/stage1_last.pt"
    cp = torch.load(cp_path, map_location="cpu", weights_only=True)
    if (cp.get("protocol") != PROTOCOL or cp.get("study_sha256") != sha(path) or
            cp.get("group") != "P" or cp.get("stage") != 1 or cp.get("step") != 1560 or
            cp.get("training_seed", 42) != seed or
            cp.get("FM_state_sha256") != bundle["fm_state_sha256"] or
            cp.get("initial_slow_state_sha256") != bundle["slow_state_sha256"]):
        raise ValueError("Not the matching P stage1_last.pt; do not substitute best.pt or last.pt")
    if not read(root / "P/completed.json").get("completed"):
        raise ValueError("Original P run did not finish")
    model_from_bundle(bundle, torch.device("cpu")).load_state_dict(cp["model"], strict=True)
    score_path = root / "P/validation/stage2_step000390/correlation_score.json"
    score = read(score_path)
    if score["score_protocol"] != SCORE_PROTOCOL:
        raise ValueError("Different historical score scope")
    return study, bundle, {
        "checkpoint": str(cp_path), "sha256": sha(cp_path), "source_study": str(path),
        "source_study_sha256": sha(path), "source_code_sha256": study["code_sha256"],
        "source_step": 1560, "slow_state_sha256": state_sha(cp["model"]),
        "reference_step390": str(score_path), "reference_step390_sha256": sha(score_path)}


def prepare_trials(reference42, reference43, output):
    sources = {42: historical_handoff(reference42, 42), 43: historical_handoff(reference43, 43)}
    a, b = sources[42][0], sources[43][0]
    for key in ("bundle_sha256", "huawei", "public", "sampling", "beta", "residual_weight",
                "F_width_native", "combo_weight", "mean_weight", "initial_slow_state_sha256"):
        if a[key] != b[key]:
            raise ValueError("Historical seeds changed a frozen setting: " + key)
    # Both reference studies must describe the same original training budget/loss/optimizer.
    for key in ("stage1_updates", "stage2_updates", "learning_rate", "batch_records", "weight_decay", "gradient_clip"):
        if a["training"][key] != b["training"][key]:
            raise ValueError("Historical training settings differ: " + key)
    output = empty_output(output)
    rows = []
    for seed, (source, bundle, handoff) in sources.items():
        shared = output / f"seed{seed}_lr1e4/shared"
        # Normal current-code preparation verifies every data file and the training/deployment forward.
        prepare(Path(handoff["source_study"]).parent / "components.pt", source["huawei"]["root"],
                source["public"]["root"], shared, cache_study=source["source_cache_study"],
                u1=1560, u2=780, validate_every=78, train_seed=seed)
        template = read(shared / "study.json")
        for key in ("sampling", "beta", "residual_weight", "F_width_native", "combo_weight", "mean_weight"):
            if template[key] != source[key]:
                raise ValueError("Current preparation differs from the original experiment: " + key)
        for arm, lr in ARMS.items():
            folder = output / f"seed{seed}_{arm}/shared"
            if arm != "lr1e4":
                folder.mkdir(parents=True)
                shutil.copyfile(shared / "components.pt", folder / "components.pt")
            study = copy.deepcopy(template)
            study["training"].update(stage1_updates=0, stage2_updates=780, validate_every=78,
                                     learning_rate=lr, dense_validate_every=0, dense_stage2_until=0)
            study["stage2_restart"] = handoff
            study["selection"] = "common_initial_bundle_or_stage2_initial_or_every78_steps_through780"
            study["stage1_handoff"] = "reuse_own_seed_P_stage1_last_fresh_stage2_optimizer_and_sampler"
            study["external_checks"] += [{"path": handoff["checkpoint"], "sha256": handoff["sha256"]},
                {"path": handoff["source_study"], "sha256": handoff["source_study_sha256"]},
                {"path": handoff["reference_step390"], "sha256": handoff["reference_step390_sha256"]}]
            write(folder / "study.json", study)
            write(folder / "preflight.json", {"ready": True, "training_started": False,
                "FM_optimizer_updates": 0, "training_seed": seed, "learning_rate": lr,
                "new_stage1_updates": 0, "stage2_updates": 780, "validate_every": 78,
                "source_handoff_sha256": handoff["sha256"], "score_protocol": SCORE_PROTOCOL})
            rows.append({"seed": seed, "arm": arm, "study": str(folder / "study.json"),
                         "learning_rate": lr, "source_handoff_sha256": handoff["sha256"]})
    write(output / "trials.json", {"ready": True, "trials": rows, "training_started": False})
    print("Prepared four stage2-only trials: " + str(output), flush=True)


def verify_reproduction(trial):
    trial = Path(trial)
    study, _ = load_study(trial / "shared/study.json")
    if study["training"]["learning_rate"] != 1e-4:
        raise ValueError("Only the unchanged learning-rate arm can reproduce the original step390")
    current = read(trial / "P/validation/stage2_step000390/correlation_score.json")
    expected = read(study["stage2_restart"]["reference_step390"])
    keys = ("task1_r_missing11", "task2_r_missing11", "main_correlation")
    deltas = {k: current[k] - expected[k] for k in keys}
    matched = current["score_protocol"] == expected["score_protocol"] and all(abs(v) <= 2e-5 for v in deltas.values())
    write(trial / "reproduction.json", {"reference_matched": matched, "delta": deltas,
          "reference": expected, "actual": current, "tolerance": 2e-5})
    if not matched:
        raise ValueError("Original step390 not reproduced; inspect reproduction.json before comparing LR arms")
    return deltas


def summarize(root, output):
    root = Path(root)
    rows, grid_rows = [], []
    shared_identity, seed_handoffs = None, {}
    for seed in (42, 43):
        for arm, lr in ARMS.items():
            trial = root / f"seed{seed}_{arm}"
            path = trial / "shared/study.json"
            study, _ = load_study(path)
            cfg = study["training"]
            handoff = study["stage2_restart"]
            identity = {k: study[k] for k in ("bundle_sha256", "huawei", "public", "sampling", "beta",
                        "residual_weight", "combo_weight", "mean_weight", "score_protocol")}
            if shared_identity is not None and identity != shared_identity:
                raise ValueError("Refinement arms changed data/model/loss/score settings")
            shared_identity = identity
            if seed in seed_handoffs and seed_handoffs[seed] != handoff["sha256"]:
                raise ValueError("Learning-rate arms must use the same handoff for each seed")
            seed_handoffs[seed] = handoff["sha256"]
            if (cfg["seed"] != seed or cfg["learning_rate"] != lr or cfg["stage1_updates"] != 0 or
                    cfg["stage2_updates"] != 780 or cfg["validate_every"] != 78):
                raise ValueError("Unexpected trial budget/seed/LR/grid")
            if not read(trial / "P/completed.json").get("completed"):
                raise ValueError("An arm is unfinished")
            cp = torch.load(trial / "P/best.pt", map_location="cpu", weights_only=True)
            if cp["study_sha256"] != sha(path):
                raise ValueError("Selected checkpoint belongs to another study")
            if arm == "lr1e4":
                verify_reproduction(trial)
            score = cp["validation_score"]
            rows.append({"training_seed": seed, "arm": arm, "learning_rate": lr,
                         "stage": cp["stage"], "step": cp["step"], **score})
            for step in range(0, 781, 78):
                folder = trial / "P/validation" / f"stage2_step{step:06d}"
                score = read(folder / "correlation_score.json")
                grid_rows.append({"training_seed": seed, "arm": arm, "step": step, **score})
    output = empty_output(output)
    csv_write(output / "summary.csv", rows)
    csv_write(output / "curves.csv", grid_rows)
    write(output / "summary.json", {"rows": rows, "selection_scope": "matched_78_step_grid_through780_plus_common_initial",
          "independent_test_result": False, "FM_optimizer_updates": 0,
          "mean_R_by_arm": {arm: sum(r["main_correlation"] for r in rows if r["arm"] == arm) / 2 for arm in ARMS}})
    lines = ["# P慢CNN：加密验证与阶段二学习率比较", "", "同seed内同公开阶段权重、同数据顺序、相同780次更新；仅学习率不同。", "",
             "|训练seed|学习率|最佳阶段/更新|Task1 r|Task2 r|平均r|", "|---|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['training_seed']}|{r['learning_rate']}|{r['stage']}/{r['step']}|{r['task1_r_missing11']:.6f}|"
                     f"{r['task2_r_missing11']:.6f}|{r['main_correlation']:.6f}|")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(rows, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--reference42", required=True); a.add_argument("--reference43", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("verify-reproduction"); a.add_argument("--trial", required=True)
    a = sub.add_parser("summarize"); a.add_argument("--root", required=True); a.add_argument("--output", required=True)
    a = p.parse_args()
    if a.action == "prepare": prepare_trials(a.reference42, a.reference43, a.output)
    elif a.action == "verify-reproduction": print(verify_reproduction(a.trial), flush=True)
    else: summarize(a.root, a.output)


if __name__ == "__main__":
    main()
