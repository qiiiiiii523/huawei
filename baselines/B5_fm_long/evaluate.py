"""Actual full120s raw/hybrid scores from the SAME FM sample; no device context."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch

from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5.scoring import diagnostic_summary, scores_from_summaries, write_score_report
from baselines.B5_best.bundle import load_bundle
from baselines.B5.checkpoint import model_from_checkpoint
from baselines.B5_best.common import write, csv_write, state_sha, sha
from baselines.B5_best.combine import slow_curve
from baselines.B5_best_context.evaluate import _record_metrics
from ecg12gen.evaluate import evaluate_record_predictions, write_report
from . import PROTOCOL, MODES
from .common import load_study, load_candidate, read_record, metadata
from .inference import generate_record, outputs


def component_diagnostics(prediction, target, quality, row, mode, task):
    """Hidden targets are used here ONLY after generation for offline diagnostics."""
    if not quality[0]:
        return []
    from ecg12gen.contracts import D12_LEADS
    p, y = prediction[1:].astype(np.float64), target[1:].astype(np.float64)
    ps, ys = slow_curve(p), slow_curve(y)
    pf, yf = p - ps, y - ys
    seams = np.arange(5000, 60000, 5000)
    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / den) if den > 0 else None
    result = []
    for j in range(11):
        if not quality[j + 1]:
            continue
        dy = float(np.std(ys[j]))
        jump_error = ((p[j, seams] - p[j, seams - 1]) - (y[j, seams] - y[j, seams - 1]))
        result.append({"task_id": task, "pair_id": row["pair_id"], "subject_id": row["subject_id"],
            "target_record_id": row["target_record_id"], "input_type": row["input_type"],
            "mode": mode, "lead": D12_LEADS[j + 1], "full_r": corr(p[j], y[j]),
            "slow_r": corr(ps[j], ys[j]), "fast_r": corr(pf[j], yf[j]),
            "slow_std_ratio": float(np.std(ps[j]) / dy) if dy > 0 else None,
            "same_10s_boundary_jump_error_rms_uV": float(np.sqrt(np.mean(jump_error ** 2)))})
    return result


class Evaluator:
    def __init__(self, run, study, device, batch_size=1):
        self.root, self.study, self.device = Path(run), study, device
        self.batch_size = batch_size

    @torch.no_grad()
    def evaluate(self, model, window_seconds, output, label="initial", update=0, package=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        model.eval()
        scale = np.asarray(self.study["scale_uV"], np.float32)
        summaries = {mode: {} for mode in MODES}
        pair_rows, components, generated_cache = [], [], {}
        started = time.perf_counter()
        for task, entries in self.study["validation"].items():
            rows = self.study["validation_rows"][task]
            n = len(rows)
            task_dir = output / "arrays" / task
            task_dir.mkdir(parents=True)
            prediction = {}
            for mode in MODES:
                prediction[mode] = np.lib.format.open_memmap(task_dir / (mode + "_prediction_uV.npy"),
                    mode="w+", dtype=np.float32, shape=(n, 12, 5000))
            target = np.lib.format.open_memmap(task_dir / "target_uV.npy", mode="w+", dtype=np.float32, shape=(n, 12, 5000))
            quality = np.lib.format.open_memmap(task_dir / "quality.npy", mode="w+", dtype=bool, shape=(n, 12))
            assigned = set()
            try:
                for entry in entries:
                    data = read_record(self.root, entry)
                    key = entry["target_record_id"]
                    if key not in generated_cache:
                        if package is None:
                            raw = generate_record(model, data["anchor_uV"], metadata(data), key, scale,
                                                  self.device, window_seconds, self.batch_size)
                            generated_cache[key] = outputs(raw, data["slow_uV"])
                        else:
                            # Portable verification recomputes slow prediction from visible I.
                            generated_cache[key] = package.predict_encoded(data["anchor_uV"], metadata(data), key,
                                                                           self.batch_size)
                    values = generated_cache[key]
                    ids = entry["window_indices"]
                    if assigned.intersection(ids) or len(ids) != 12:
                        raise ValueError("Duplicate/incomplete validation assignment")
                    row = rows[ids[0]]
                    for j, i in enumerate(ids):
                        span = slice(j * 5000, (j + 1) * 5000)
                        target[i] = data["target_uV"][:, span]
                        quality[i] = data["mask"]
                        for mode in MODES:
                            prediction[mode][i] = values[mode][:, span]
                    for mode in MODES:
                        pair_rows.append({"mode": mode, **_record_metrics(values[mode], data["target_uV"],
                                                                 data["mask"], row, task)})
                        components.extend(component_diagnostics(values[mode], data["target_uV"], data["mask"], row, mode, task))
                    assigned.update(ids)
                    print({"task": task, "target": key, "window_seconds": window_seconds, "update": update}, flush=True)
                if assigned != set(range(n)):
                    raise ValueError("Evaluation must retain ALL original validation windows")
                target.flush(); quality.flush()
                for mode in MODES:
                    prediction[mode].flush()
                    overall, details = evaluate_record_predictions(prediction[mode], target, task, rows,
                        target_quality_mask=quality, input_quality_mask=quality[:, 0],
                        require_reliable_input=True, input_quality_source="same_record_target_I")
                    overall = diagnostic_summary(overall)
                    overall.update(context_used=False, fm_window_seconds=window_seconds,
                                   output_mode=mode, sampling=self.study["sampling"])
                    write_report(output / mode / task, overall, details, title=f"FM{window_seconds} {mode} {task}")
                    csv_write(output / mode / task / "window_metadata.csv", rows)
                    summaries[mode][task] = overall
            finally:
                for array in list(prediction.values()) + [target, quality]:
                    array._mmap.close()
        result = {}
        for mode in MODES:
            score = scores_from_summaries(summaries[mode])
            write_score_report(output / mode, score)
            result[mode] = {"label": label, "window_seconds": window_seconds, "update": update, "mode": mode,
                "task1_r": score["task1_r_missing11"], "task2_r": score["task2_r_missing11"],
                "machine_r": summaries[mode]["task2"]["ecg_machine_d6_r_missing11"],
                "body_r": summaries[mode]["task2"]["body_scale_d6_r_missing11"],
                "mean_r": score["main_correlation"]}
        csv_write(output / "summary.csv", list(result.values()))
        csv_write(output / "pair_metrics.csv", pair_rows)
        csv_write(output / "component_diagnostics.csv", components)
        report = {"protocol": PROTOCOL, "completed": True, "summary": result,
            "model_state_sha256": state_sha(model.state_dict()), "context": "disabled",
            "sampling": self.study["sampling"], "slow_state_sha256": self.study["slow_state_sha256"],
            "elapsed_seconds": time.perf_counter() - started,
            "training_started": update > 0, "target_used_at_inference": False,
            "inference_path": "portable_visible_inputs" if package else "frozen_slow_cache",
            "noise_protocol": "same_record_10s_keyed_noise_concatenated_then_sliced",
            "population": self.study["sources"], "study_sha256": sha(self.root / "study.json")}
        write(output / "evaluation.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-seconds", type=int, choices=(10, 40))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--candidate")
    source.add_argument("--package")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Positive batch size required")
    study = load_study(args.run)
    device = device_from_name(args.device)
    seed_all(42, True)
    package = None
    if args.candidate:
        model, payload = load_candidate(args.candidate, study, Path(args.run) / "study.json", device)
        seconds, label, update = payload["window_seconds"], "trained", payload["update"]
    elif args.package:
        from .predict import Predictor
        package = Predictor(args.package, device)
        model = package.fm
        seconds, label, update = package.seconds, "packaged", package.payload["update"]
        if package.payload["study_sha256"] != sha(Path(args.run) / "study.json"):
            raise ValueError("Package belongs to another prepared study")
    else:
        if args.window_seconds is None:
            parser.error("Initial screening requires --window-seconds10 or40")
        model = model_from_checkpoint(load_bundle(study["base_checkpoint"])["fm"], device)
        seconds, label, update = args.window_seconds, "initial", 0
    if args.window_seconds is not None and args.window_seconds != seconds:
        parser.error("Window override disagrees with candidate/package")
    result = Evaluator(args.run, study, device, args.batch_size).evaluate(
        model, seconds, args.output, label, update, package)
    print(result["summary"], flush=True)


if __name__ == "__main__":
    main()
