"""Four device-route groups plus no-context/disabled/shuffled body controls."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, csv_write, sha
from baselines.B5.scoring import SCORE_PROTOCOL, scores_from_summaries, write_score_report
from ecg12gen.evaluate import write_report
from .adapters import ContextRouter, load_settings, _visible_digest
from .body_prepare import load_cache
from .evaluate import load_baseline, HistoryProvider, _score, _record_metrics


def shuffled_donors(body_rows, seed):
    """A fixed subject-level derangement; never swap windows or keep the same patient."""
    by_subject = {}
    for row in sorted(body_rows, key=lambda r: r["pair_id"]):
        by_subject.setdefault(row["subject_id"], row)
    subjects = sorted(by_subject)
    if len(subjects) < 2:
        raise ValueError("Shuffled control requires at least two body subjects")
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        order = rng.permutation(len(subjects))
        if np.all(order != np.arange(len(subjects))):
            return {subject: by_subject[subjects[int(order[i])]] for i, subject in enumerate(subjects)}
    raise ValueError("Could not build a subject derangement")


def groups():
    return [("baseline", "none", None, None), ("d6_only", "fir25", None, None),
        ("body_no_context", "none", "no_context", None),
        ("body_only", "none", "correct", None), ("both", "fir25", "correct", None),
        ("body_disabled", "none", "disabled", None),
        *[(f"body_shuffled_{seed}", "none", "correct", seed) for seed in (101, 202, 303)]]


def run_study(baseline_dir, output, provider, context_adapter, no_context_adapter, *, loaded=None):
    """Models see only visible I/prediction/history. Targets remain in the scoring code."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty study directory; keep tee logs adjacent")
    baseline, data, fingerprints = loaded or load_baseline(baseline_dir)
    output.mkdir(parents=True, exist_ok=True)
    body_rows = [item["rows"][ids[0]] for item in [data["task2"]] for ids in item["pairs"]
                 if item["rows"][ids[0]]["input_type"] == "body_scale_d6"]
    summary, base_score, reference = [], None, {}
    for name, machine_mode, body_mode, shuffle in groups():
        adapter = no_context_adapter if body_mode == "no_context" else context_adapter if body_mode else None
        if adapter is not None:
            adapter.mode = "disabled" if body_mode in ("disabled", "no_context") else "correct"
        router = ContextRouter(load_settings(), machine_mode=machine_mode, body_adapter=adapter)
        donor = shuffled_donors(body_rows, shuffle) if shuffle is not None else None
        if donor:
            csv_write(output / name / "shuffled_subjects.csv", [{"subject_id": s,
                "donor_subject_id": r["subject_id"], "donor_pair_id": r["pair_id"], "seed": shuffle}
                for s, r in donor.items()])
        tasks, statuses, record_metrics, changes = {}, [], [], {}
        group = output / name
        for task, item in data.items():
            folder = group / task
            folder.mkdir(parents=True, exist_ok=True)
            prediction = np.lib.format.open_memmap(folder / "prediction_uV.npy", mode="w+",
                dtype=np.float32, shape=item["prediction"].shape)
            max_forbidden = 0.
            for indices in item["pairs"]:
                row = item["rows"][indices[0]]
                base = np.concatenate([item["prediction"][i] for i in indices], axis=-1)
                anchor = np.concatenate([item["target"][i, :1] for i in indices], axis=-1)
                device = row["input_type"]
                history = None
                donor_row = row
                if task == "task2" and ((device == "ecg_machine_d6" and machine_mode == "fir25")
                        or (device == "body_scale_d6" and adapter is not None)):
                    donor_row = donor[row["subject_id"]] if donor and device == "body_scale_d6" else row
                    history = provider(donor_row)
                enhanced, status = router.apply(base, anchor, task_id=task, input_type=device, history=history)
                allowed = (list(range(1, 6)) if device == "ecg_machine_d6" and machine_mode == "fir25"
                           else [1] if device == "body_scale_d6" and adapter is not None else [])
                forbidden = [i for i in range(12) if i not in allowed]
                difference = float(np.max(np.abs(enhanced[forbidden] - base[forbidden])))
                max_forbidden = max(max_forbidden, difference)
                if difference != 0:
                    raise AssertionError("Context changed a forbidden lead/route")
                for j, index in enumerate(indices):
                    prediction[index] = enhanced[:, j * 5000:(j + 1) * 5000]
                truth = np.concatenate([item["target"][i] for i in indices], axis=-1)
                record_metrics.append(_record_metrics(enhanced, truth, item["quality"][indices[0]], row, task))
                statuses.append({"task_id": task, "pair_id": row["pair_id"], "subject_id": row["subject_id"],
                    "target_record_id": row["target_record_id"], "input_type": device,
                    "history_subject_id": donor_row["subject_id"], **status})
            prediction.flush()
            overall, details = _score(prediction, item, task)
            overall.update(model="B5-best-context-body", context_group=name,
                sampling=baseline["sampling"], slow_beta=.75, target_used_at_inference=False)
            write_report(folder, overall, details, title=f"B5-best-context {name} {task}")
            csv_write(folder / "window_metadata.csv", item["rows"])
            write(folder / "source_arrays.json", {"baseline_evaluation": str(Path(baseline_dir).resolve()),
                "population": fingerprints[task], "target_and_QC": str(Path(baseline_dir).resolve() / task)})
            tasks[task], changes[task] = overall, {"forbidden_lead_or_route_max_abs_uV": max_forbidden}
            del prediction
        score = scores_from_summaries(tasks)
        if base_score is None:
            base_score = score
            for key in ("task1_r_missing11", "task2_r_missing11", "main_correlation"):
                if abs(score[key] - baseline["score"][key]) > 1e-8:
                    raise ValueError("Baseline arrays failed to reproduce the recorded score")
        write_score_report(group, score)
        # Status fields differ between routes; make a rectangular sidecar explicitly.
        fields = list(dict.fromkeys(k for s in statuses for k in s))
        csv_write(group / "postprocess_records.csv", [{k: s.get(k) for k in fields} for s in statuses])
        csv_write(group / "pair_metrics.csv", record_metrics)
        write(group / "evaluation.json", {"completed": True, "score_protocol": SCORE_PROTOCOL,
            "score": score, "sampling": baseline["sampling"], "slow_beta": .75,
            "checkpoint_sha256": baseline["checkpoint_sha256"], "population": fingerprints,
            "group": name, "invariance": changes, "target_used_at_inference": False})
        row = {"group": name, "task1_r": score["task1_r_missing11"],
            "machine_r": tasks["task2"]["ecg_machine_d6_r_missing11"],
            "body_r": tasks["task2"]["body_scale_d6_r_missing11"],
            "task2_r": score["task2_r_missing11"], "mean_r": score["main_correlation"],
            "delta_mean_r": score["main_correlation"] - base_score["main_correlation"]}
        reference[name] = row; summary.append(row)
        print(row, flush=True)
    if abs((reference["both"]["mean_r"] - reference["d6_only"]["mean_r"]) -
           (reference["body_only"]["mean_r"] - reference["baseline"]["mean_r"])) > 1e-10:
        raise AssertionError("Independent device-route effects must add exactly in mean r")
    for a, b, field in (("both", "d6_only", "machine_r"), ("both", "body_only", "body_r")):
        if abs(reference[a][field] - reference[b][field]) > 1e-10:
            raise AssertionError("Cross-device route contamination")
    csv_write(output / "summary.csv", summary)
    lines = ["# B5-best：d6/body 配对结果", "", "同一B5-best预测；Heun16/K16，seed42，β=0.75。", "",
        "|组|Task1 r|机器r|体脂秤r|Task2 r|平均r|Δ平均r|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append("|" + r["group"] + "|" + "|".join(f"{r[k]:.8f}" for k in (
            "task1_r", "machine_r", "body_r", "task2_r", "mean_r", "delta_mean_r")) + "|")
    lines += ["", "body_only 对比 body_no_context 区分个人信息与通用修正收益；",
        "再看 body_disabled 和全部预设 shuffled，不挑最好的打乱种子。",
        "both 对比 d6_only 是 body 对现有方案的额外贡献。两个设备分别路由，不串联处理同一条记录。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(output / "study.json", {"completed": True, "training_started": False, "summary": summary,
        "base_checkpoint_sha256": baseline["checkpoint_sha256"], "population": fingerprints,
        "device_additivity_checked": True})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--context-checkpoint", required=True)
    p.add_argument("--no-context-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    from baselines.B5.runtime import device_from_name
    from .body_model import BodyAdapter
    cache = load_cache(args.cache)
    frozen = cache["frozen"]
    loaded = load_baseline(frozen["baseline_evaluation"])
    if frozen["baseline_population"] != loaded[2]:
        raise ValueError("Validation population/base predictions changed since preparation")
    base_sha = loaded[0]["checkpoint_sha256"]
    device = device_from_name(args.device)
    context = BodyAdapter(args.context_checkpoint, base_sha, device, expected_group="context")
    control = BodyAdapter(args.no_context_checkpoint, base_sha, device, expected_group="no_context")
    for adapter in (context, control):
        if adapter.payload["cache_sha256"] != sha(Path(args.cache) / "cache.json"):
            raise ValueError("Body checkpoint was trained on a different cache")
    if context.payload["initial_weights_sha256"] != control.payload["initial_weights_sha256"]:
        raise ValueError("Body pair did not start from identical weights")
    provider = HistoryProvider(args.config, loaded[1]["task2"])
    try:
        validation_rows = {r["pair_id"]: r for r in cache["records"] if r["split"] == "validation"}
        for ids in loaded[1]["task2"]["pairs"]:
            row = loaded[1]["task2"]["rows"][ids[0]]
            if row["input_type"] == "body_scale_d6":
                if _visible_digest(provider(row)) != validation_rows[row["pair_id"]]["history_visible_sha256"]:
                    raise ValueError("Body validation history changed since training")
        summary = run_study(frozen["baseline_evaluation"], args.output, provider, context, control, loaded=loaded)
        write(Path(args.output) / "body_checkpoints.json", {"context": str(Path(args.context_checkpoint).resolve()),
            "context_sha256": sha(args.context_checkpoint), "no_context": str(Path(args.no_context_checkpoint).resolve()),
            "no_context_sha256": sha(args.no_context_checkpoint), "cache": str(Path(args.cache).resolve()),
            "initial_weights_sha256": context.payload["initial_weights_sha256"], "summary": summary})
    finally:
        provider.close()


if __name__ == "__main__":
    main()
