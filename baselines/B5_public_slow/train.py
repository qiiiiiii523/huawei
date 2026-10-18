"""Fixed-update C/M/P, automatic two-stage handoff and exact checkpoint resume."""
import argparse
import json
from pathlib import Path
import time

import torch

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best.common import read, write, sha, state_sha
from . import PROTOCOL, GROUPS
from .common import load_study, empty_output, model_from_bundle, read_record, SourceSampler
from .core import batch_record, generate, objective, FIELDS
from .evaluate import evaluate, public_diagnostic


def sources_for(group, stage):
    if stage == 2 or group == "C":
        return ("huawei", "huawei")
    return ("huawei", "public") if group == "M" else ("public", "public")


def validation_schedule(stage, step, budget, cfg):
    coarse = step % cfg["validate_every"] == 0 or step == budget
    every = cfg.get("dense_validate_every", 0)
    dense = bool(stage == 2 and every and step <= cfg.get("dense_stage2_until", 390) and step % every == 0)
    return coarse, dense


def run(study_path, group, output, device_name, resume=False):
    study_path, output = Path(study_path).resolve(), Path(output).resolve()
    study, bundle = load_study(study_path, deep=True)
    cfg = study["training"]
    device = device_from_name(device_name)
    seed_all(cfg["seed"], True)
    model = model_from_bundle(bundle, device)
    scales = torch.tensor(bundle["fm"]["scales"]["d12"], device=device)
    entries = {name: read(study[name]["manifest"])["train"] for name in ("huawei", "public")}
    study_sha = sha(study_path)
    cp = None
    if resume:
        if not (output / "last.pt").is_file():
            raise FileNotFoundError("No last.pt to resume")
        cp = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if (cp.get("protocol") != PROTOCOL or cp.get("study_sha256") != study_sha or
                cp.get("group") != group or cp.get("output") != str(output)):
            raise ValueError("Resume must retain study/group/output")
        model.load_state_dict(cp["model"], strict=True)
        best = cp["best"]
        existing = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        if existing.get("study_sha256") != study_sha or existing.get("group") != group:
            raise ValueError("Existing best has another identity")
        if existing["validation_score"]["main_correlation"] > best["score"]["main_correlation"]:
            best = existing["best"]
    else:
        output = empty_output(output)
        best = None
    write(output / "completed.json", {"completed": False, "group": group, "resuming": resume})
    optimizer, samplers = None, None
    stage, step = 0, 0

    def payload(score=None):
        return {"protocol": PROTOCOL, "study_sha256": study_sha, "group": group, "output": str(output),
            "stage": stage, "step": step, "model": model.state_dict(),
            "training_seed": cfg["seed"],
            "optimizer": None if optimizer is None else optimizer.state_dict(), "rng": pack_rng(),
            "samplers": None if samplers is None else {k: s.state() for k, s in samplers.items()},
            "best": best, "validation_score": score, "FM_state_sha256": bundle["fm_state_sha256"],
            "initial_slow_state_sha256": bundle["slow_state_sha256"]}

    def validation(select):
        nonlocal best
        rng = pack_rng()
        try:
            path = output / "validation" / f"stage{stage}_step{step:06d}"
            if path.exists():
                path = path.with_name(path.name + "_attempt_" + str(time.time_ns()))
            score = evaluate(model, study, bundle, device, path)
            if stage == 1:
                write(path / "public_diagnostic.json", public_diagnostic(model, study, bundle, device))
            if select and (best is None or score["main_correlation"] > best["score"]["main_correlation"]):
                best = {"stage": stage, "step": step, "score": score}
                atomic_save(output / "best.pt", payload(score))
            # Keep diagnostic candidates separate from the predeclared 390-step seed comparison.
            if stage == 1:
                save_candidate("stage1_best.pt", score, "stage1_diagnostic_only")
            if stage == 0 or stage == 2:
                save_candidate("dense_best.pt", score, "stage2_all_observed_checks_plus_initial")
            write(path / "selection.json", {"eligible_for_best": select, "current_best": best})
            print(json.dumps({"phase": "validation", "stage": stage, "step": step, **score,
                              "best": best}, ensure_ascii=False), flush=True)
            return score
        finally:
            restore_rng(rng)

    def save_candidate(filename, score, role):
        destination = output / filename
        if destination.is_file():
            previous = torch.load(destination, map_location="cpu", weights_only=True)
            if previous.get("study_sha256") != study_sha or previous.get("group") != group:
                raise ValueError("Diagnostic candidate identity mismatch")
            if previous["validation_score"]["main_correlation"] >= score["main_correlation"]:
                return
        atomic_save(destination, payload(score) | {"candidate_role": role})

    if not resume:
        initial = validation(True)
        atomic_save(output / "last.pt", payload(initial))
        write(output / "initialization.json", {"group": group, "study_sha256": study_sha,
              "training_seed": cfg["seed"],
              "slow_state_sha256": state_sha(model.state_dict()), "FM_frozen": True,
              "score_protocol": study["score_protocol"], "initial_score": initial})

    for next_stage in (1, 2):
        if cp is not None and cp["stage"] > next_stage:
            continue
        stage = next_stage
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        samplers = {name: SourceSampler(entries[name], cfg["seed"] + 100 * stage + i,
                                       patient_balanced=(name == "public"))
                    for i, name in enumerate(("huawei", "public"))}
        step = 0
        same_stage_resume = cp is not None and cp["stage"] == stage
        if same_stage_resume:
            optimizer.load_state_dict(cp["optimizer"])
            for name, sampler in samplers.items():
                sampler.restore(cp["samplers"][name])
            restore_rng(cp["rng"])
            step = cp["step"]
        else:
            seed_all(cfg["seed"] + stage, True)
        budget = cfg[f"stage{stage}_updates"]
        if stage == 2 and not same_stage_resume:
            score = validation(True)
            atomic_save(output / "last.pt", payload(score))
        for step in range(step + 1, budget + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals = {}
            # Two record-wise microbatches allow real 50..120s lengths without padding labels.
            # No BatchNorm/dropout in this S50 CNN: equal-record gradient averaging matches batch2.
            for source in sources_for(group, stage):
                entry = samplers[source].draw()
                record = read_record(study[source]["root"], entry)
                batch = batch_record(record, device)
                prediction = generate(model, {key: batch[key] for key in FIELDS})
                loss, parts = objective(prediction, batch, scales, study["residual_weight"],
                                        study["combo_weight"] if stage == 2 else 0.)
                (loss / 2).backward()
                for name, value in {"loss": loss, **parts}.items():
                    totals[name] = totals.get(name, 0.) + float(value.detach()) / 2
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            score = None
            coarse_check, dense_check = validation_schedule(stage, step, budget, cfg)
            if coarse_check or dense_check:
                score = validation(stage == 2 and coarse_check)
                if state_sha(bundle["fm"]["ema"]) != bundle["fm_state_sha256"]:
                    raise ValueError("Frozen FM changed")
                write(output / f"stage{stage}_coverage.json", {k: s.coverage() for k, s in samplers.items()})
            atomic_save(output / "last.pt", payload(score))
            record = {"stage": stage, "step": step, "source_batch": sources_for(group, stage),
                      "train": totals, "best": best}
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if step == 1 or step % 10 == 0 or score is not None:
                print(json.dumps(record, ensure_ascii=False), flush=True)
        atomic_save(output / f"stage{stage}_last.pt", payload())
        cp = None
    write(output / "completed.json", {"completed": True, "group": group, "best": best,
          "stage1_updates": cfg["stage1_updates"], "stage2_updates": cfg["stage2_updates"],
          "FM_optimizer_updates": 0, "score_protocol": study["score_protocol"], "training_seed": cfg["seed"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute-training", action="store_true")
    a = p.parse_args()
    if not a.execute_training:
        p.error("No training started. Pass --execute-training on the server")
    run(a.study, a.group, a.output, a.device, a.resume)


if __name__ == "__main__":
    main()
