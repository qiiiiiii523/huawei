"""Paired baseline/projection/FIR25 evaluation using exactly one frozen prediction cache."""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, csv_write, sha
from baselines.B5.scoring import SCORE_PROTOCOL, scores_from_summaries, diagnostic_summary, write_score_report
from ecg12gen.evaluate import evaluate_record_predictions, write_report
from . import PROTOCOL
from .adapters import ContextRouter, history_from_sample, load_settings


def pairs(rows):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["pair_id"]].append(index)
    result = []
    for indices in grouped.values():
        indices.sort(key=lambda i: int(rows[i]["start_sample_500hz"]))
        if [int(rows[i]["start_sample_500hz"]) for i in indices] != list(range(0, 60000, 5000)):
            raise ValueError("Every pair requires 12 real chronological windows")
        for field in ("target_record_id", "subject_id", "input_type"):
            if len({rows[i][field] for i in indices}) != 1:
                raise ValueError("Pair mixes physical identities")
        result.append(indices)
    return result


def load_baseline(directory):
    directory = Path(directory).resolve()
    report = read(directory / "evaluation.json")
    sampling = report.get("sampling", {})
    if (report.get("completed") is not True or report.get("score_protocol") != SCORE_PROTOCOL
            or report.get("require_reliable_input") is not True):
        raise ValueError("Use a completed current reliable-I B5-best validation, not an old QC/60-point report")
    if sampling.get("solver") != "heun" or sampling.get("steps") != 16 or sampling.get("samples") != 16:
        raise ValueError("Paired context study requires unchanged Heun16/K16 base predictions")
    if report.get("slow_beta") != .75:
        raise ValueError("The current context candidate is pinned to B5-best beta=0.75")
    data, fingerprints = {}, {}
    for task in ("task1", "task2"):
        folder = directory / task
        with (folder / "window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        arrays = {key: np.load(folder / filename, mmap_mode="r", allow_pickle=False) for key, filename in (
            ("prediction", "prediction_uV.npy"), ("target", "target_uV.npy"), ("quality", "target_quality_mask.npy"))}
        if (arrays["prediction"].shape != (len(rows), 12, 5000)
                or arrays["target"].shape != arrays["prediction"].shape
                or arrays["quality"].shape != (len(rows), 12)
                or arrays["prediction"].dtype != np.float32):
            raise ValueError("Expected parent B5-best float32 window prediction/target and quality arrays")
        fingerprints[task] = {"metadata_sha256": sha(folder / "window_metadata.csv"),
                              "target_sha256": sha(folder / "target_uV.npy"),
                              "QC_sha256": sha(folder / "target_quality_mask.npy")}
        if report.get("population", {}).get(task) != fingerprints[task]:
            raise ValueError("Baseline metadata/target/QC changed after validation: " + task)
        fingerprints[task]["prediction_sha256"] = sha(folder / "prediction_uV.npy")
        data[task] = {**arrays, "rows": rows, "pairs": pairs(rows)}
    return report, data, fingerprints


class HistoryProvider:
    """Use current main's raw history+input QC. Check the live task2 target population first."""

    def __init__(self, config, baseline_task2):
        from baselines.B5.config import load_config
        from baselines.B5.data import common_config
        from ecg12gen.dataset import JointAnchorDataset
        cfg = load_config(config, data_only=True)
        self.dataset = JointAnchorDataset(common_config(cfg), "task2", "validation")
        live_rows = [self.dataset._rows[i] for i in self.dataset._indices]
        self.indices = {(r["pair_id"], int(r["start_sample_500hz"])): i for i, r in enumerate(live_rows)}
        if len(self.indices) != len(live_rows) or len(live_rows) != len(baseline_task2["rows"]):
            raise ValueError("History data and baseline have different target populations")
        for i, row in enumerate(baseline_task2["rows"]):
            key = row["pair_id"], int(row["start_sample_500hz"])
            if key not in self.indices:
                raise ValueError("Baseline pair/window absent from live history data")
            sample = self.dataset[self.indices[key]]
            if (sample.subject_id != row["subject_id"] or sample.target_record_id != row["target_record_id"]
                    or sample.context_source_type != row["input_type"]
                    or not np.array_equal(sample.Y_12lead, baseline_task2["target"][i])
                    or not np.array_equal(sample.target_quality_mask, baseline_task2["quality"][i])):
                raise ValueError("Live target identity/waveform/QC differs from the frozen baseline")
        self.cache = {}

    def __call__(self, row):
        key = row["pair_id"], int(row["start_sample_500hz"])
        if row["pair_id"] not in self.cache:
            # Only visible history fields cross the adapter boundary; never sample.Y_12lead.
            self.cache[row["pair_id"]] = history_from_sample(self.dataset[self.indices[key]])
        return self.cache[row["pair_id"]]

    def close(self):
        self.dataset.close()


def _score(prediction, item, task):
    overall, details = evaluate_record_predictions(prediction, item["target"], task, item["rows"],
        target_quality_mask=item["quality"], input_quality_mask=item["quality"][:, 0],
        require_reliable_input=True, input_quality_source="same_record_target_I")
    return diagnostic_summary(overall), details


def _record_metrics(prediction, target, quality, row, task):
    """Post-hoc diagnostics only, never used to fit/select a patient's FIR strength."""
    from ecg12gen.contracts import D12_LEADS
    result = {"task_id": task, "pair_id": row["pair_id"], "target_record_id": row["target_record_id"],
              "subject_id": row["subject_id"], "input_type": row["input_type"], "input_I_reliable": bool(quality[0])}
    values = []
    for lead in range(1, 12):
        value = None
        if quality[0] and quality[lead]:
            x, y = prediction[lead].astype(float), target[lead].astype(float)
            x, y = x - x.mean(), y - y.mean()
            den = np.sqrt(np.sum(x*x) * np.sum(y*y))
            if den > 0:
                value = float(np.sum(x*y) / den)
                values.append(value)
        result["r_" + D12_LEADS[lead]] = value
    result["r_missing11"] = float(np.mean(values)) if values else None
    return result


def run_study(baseline_dir, output_dir, history_provider, settings=None, *, loaded=None):
    """history_provider(row) returns only HistoryRecord. No model/optimizer is instantiated."""
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty study directory; keep tee logs in the parent")
    baseline, data, fingerprints = load_baseline(baseline_dir) if loaded is None else loaded
    settings = load_settings() if settings is None else settings
    output.mkdir(parents=True, exist_ok=True)
    summaries, base_score = {}, None
    summary_rows, pair_comparison = [], {}
    for mode, name in (("none", "baseline"), ("projection", "projection"), ("fir25", "fir25")):
        router = ContextRouter(settings, machine_mode=mode)
        group = output / name
        tasks, statuses, changes, pair_metrics = {}, [], {}, []
        for task, item in data.items():
            folder = group / task
            folder.mkdir(parents=True, exist_ok=True)
            # Base data stay immutable; enhanced results are per pair, never cached by target ID.
            prediction = (item["prediction"] if mode == "none" else np.lib.format.open_memmap(
                folder / "prediction_uV.npy", mode="w+", dtype=np.float32, shape=item["prediction"].shape))
            chest_max = other_route_max = 0.
            for indices in item["pairs"]:
                row = item["rows"][indices[0]]
                base = np.concatenate([item["prediction"][i] for i in indices], axis=-1)
                anchor = np.concatenate([item["target"][i, :1] for i in indices], axis=-1)
                # Current I is the public observed same-record I; the other target leads are never passed.
                history = (history_provider(row) if mode == "fir25" and task == "task2"
                           and row["input_type"] == "ecg_machine_d6" else None)
                enhanced, status = router.apply(base, anchor, task_id=task, input_type=row["input_type"], history=history)
                truth = np.concatenate([item["target"][i] for i in indices], axis=-1)
                diagnostic = _record_metrics(enhanced, truth, item["quality"][indices[0]], row, task)
                pair_metrics.append(diagnostic)
                pair_key = (task, row["pair_id"])
                if mode == "none":
                    pair_comparison[pair_key] = {k: diagnostic[k] for k in (
                        "task_id", "pair_id", "target_record_id", "subject_id", "input_type", "input_I_reliable")}
                pair_comparison[pair_key][name + "_r"] = diagnostic["r_missing11"]
                chest_max = max(chest_max, float(np.max(np.abs(enhanced[6:] - base[6:]))))
                if task != "task2" or row["input_type"] != "ecg_machine_d6":
                    other_route_max = max(other_route_max, float(np.max(np.abs(enhanced - base))))
                if mode != "none":
                    for ordinal, index in enumerate(indices):
                        prediction[index] = enhanced[:, ordinal * 5000:(ordinal + 1) * 5000]
                statuses.append({"task_id": task, "pair_id": row["pair_id"],
                    "target_record_id": row["target_record_id"], "input_type": row["input_type"], **status})
            if mode != "none":
                prediction.flush()
            overall, details = _score(prediction, item, task)
            overall.update(model="B5-best-context", context_mode=mode, sampling=baseline["sampling"],
                           slow_beta=baseline["slow_beta"], target_used_at_inference=False)
            write_report(folder, overall, details, title=f"B5-best-context {name} {task}")
            # Keep population metadata immutable; actual processing/usage is explicit in the sidecar.
            csv_write(folder / "window_metadata.csv", item["rows"])
            write(folder / "source_arrays.json", {"baseline_dir": str(Path(baseline_dir).resolve()),
                "target_quality_from": str(Path(baseline_dir).resolve() / task),
                "prediction_from": str((Path(baseline_dir).resolve() / task) if mode == "none" else folder),
                "context_usage_from": "../postprocess_records.csv", "population": fingerprints[task]})
            tasks[task] = overall
            changes[task] = {"chest_max_abs_difference_uV": chest_max,
                             "task1_or_body_max_abs_difference_uV": other_route_max}
            if mode != "none":
                del prediction
        if any(v != 0 for change in changes.values() for v in change.values()):
            raise AssertionError("Machine FIR changed a chest lead or a non-machine route")
        score = scores_from_summaries(tasks)
        if mode == "none":
            base_score = score
            for key in ("task1_r_missing11", "task2_r_missing11", "main_correlation"):
                if abs(score[key] - baseline["score"][key]) > 1e-8:
                    raise ValueError("Frozen baseline arrays do not reproduce their recorded score")
        write_score_report(group, score)
        csv_write(group / "postprocess_records.csv", statuses)
        csv_write(group / "pair_metrics.csv", pair_metrics)
        machine = [s for s in statuses if s["task_id"] == "task2" and s["input_type"] == "ecg_machine_d6"]
        report = {"completed": True, "protocol": PROTOCOL, "score_protocol": SCORE_PROTOCOL,
                  "mode": mode, "settings": settings, "score": score,
                  "baseline_evaluation": str(Path(baseline_dir).resolve()), "population": fingerprints,
                  "checkpoint_sha256": baseline["checkpoint_sha256"], "sampling": baseline["sampling"],
                  "slow_beta": baseline["slow_beta"], "invariance": changes,
                  "training_started": False, "optimizer_updates": 0, "target_used_at_inference": False,
                  "machine_pairs_processed_including_input_qc_exclusions": len(machine),
                  "personal_fir_pairs_including_input_qc_exclusions": sum(s["personal_fir_used"] for s in machine),
                  "fallback_pairs_including_input_qc_exclusions": sum(s["fallback"] == "projected_base" for s in machine)}
        write(group / "evaluation.json", report)
        row = {"group": name, "task1_r": score["task1_r_missing11"],
               "machine_r": tasks["task2"]["ecg_machine_d6_r_missing11"],
               "body_r": tasks["task2"]["body_scale_d6_r_missing11"],
               "task2_r": score["task2_r_missing11"], "mean_r": score["main_correlation"],
               "delta_mean_r": score["main_correlation"] - base_score["main_correlation"],
               "fir_used_pairs_before_input_qc": report["personal_fir_pairs_including_input_qc_exclusions"]}
        summary_rows.append(row)
        summaries[name] = report
        print(row, flush=True)
    csv_write(output / "summary.csv", summary_rows)
    comparison = []
    for record in pair_comparison.values():
        for name in ("projection", "fir25"):
            record[name + "_delta_vs_baseline"] = (None if record["baseline_r"] is None or record[name + "_r"] is None
                else record[name + "_r"] - record["baseline_r"])
        record["fir25_delta_vs_projection"] = (None if record["projection_r"] is None or record["fir25_r"] is None
            else record["fir25_r"] - record["projection_r"])
        comparison.append(record)
    csv_write(output / "pair_comparison.csv", comparison)
    write(output / "study.json", {"completed": True, "protocol": PROTOCOL, "settings": settings,
        "baseline": str(Path(baseline_dir).resolve()), "summary": summary_rows, "population": fingerprints,
        "scoring": "current reliable-I + target-lead QC; targets equal, Task2 devices equal; mean r only",
        "training_started": False, "optimizer_updates": 0})
    lines = ["# B5-best-context 三组配对结果", "", "同一基础预测，Heun16/K16、beta=0.75；仅Task2机器后处理。", "",
             "|组|Task1 r|机器 r|体脂秤 r|Task2 r|平均r|平均r变化|", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append("|" + row["group"] + "|" + "|".join(f"{row[k]:.8f}" for k in (
            "task1_r", "machine_r", "body_r", "task2_r", "mean_r", "delta_mean_r")) + "|")
    lines += ["", "FIR额外增益应比较fir25与projection；机器分项增益不能直接当作最终平均r增益。",
              "原始人口/输入mask维持不变；实际FIR使用及回退见各组postprocess_records.csv。",
              "派生组target/QC数组只引用baseline_evaluation，不复制；它们不是独立B5-best模型包。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_rows
