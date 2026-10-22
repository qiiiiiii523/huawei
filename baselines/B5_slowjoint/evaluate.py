"""Score actual full120 predictions after the fixed Final context routes."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch

from baselines.B5.runtime import device_from_name
from baselines.B5.scoring import scores_from_summaries, write_score_report
from baselines.B5_best.common import write, csv_write, sha
from baselines.B5_best_context.evaluate import load_baseline, HistoryProvider, _score, _record_metrics
from baselines.B5_best_context.body_model import BodyAdapter
from baselines.B5_best_context.adapters import ContextRouter, load_settings
from baselines.B5_final.profile import BASE_SHA, BODY_SHA
from . import PROTOCOL
from .common import load_study, load_model
from baselines.B5_slow120.model import correct_base
from .model import visible_batch


def slow_diagnostics(entry, data, delta, scales):
    from baselines.B5_best.diagnostics_core import moving_average, corr
    p = (data["base_slow"].astype(float) + delta) * scales[1:, None]
    y = data["slow_target"].astype(float) * scales[1:, None]
    rows = []
    for lead in range(11):
        if not data["mask"][0] or not data["mask"][lead + 1]:
            continue
        a, b = p[lead], y[lead]
        five_a, five_b = moving_average(a, 251), moving_average(b, 251)
        long_a, long_b = moving_average(a, 751), moving_average(b, 751)
        rows.append({k: entry[k] for k in ("task", "subject_id", "target_record_id", "pair_id", "input_type")} |
            {"lead": ["II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"][lead],
             "slow_r": corr(a, b), "short_lt5s_r": corr(a - five_a, b - five_b),
             "middle_5to15s_r": corr(five_a - long_a, five_b - long_b), "long_gt15s_r": corr(long_a, long_b),
             "slow_std_ratio": float(a.std() / b.std()) if b.std() > 1e-10 else None,
             "slow_rmse_uV": float(np.sqrt(np.mean((a - b) ** 2)))})
    return rows


class Evaluator:
    def __init__(self, run, study, device):
        self.run, self.study, self.device = Path(run), study, device
        self.cache_root = Path(study["source_run"])
        for key, digest in (("base_checkpoint", BASE_SHA), ("body_checkpoint", BODY_SHA)):
            if sha(study[key]) != digest:
                raise ValueError("Selected Final weight changed: " + key)
        if sha(study["config"]) != study["config_sha256"]:
            raise ValueError("Live data configuration changed")
        self.loaded = load_baseline(study["baseline_evaluation"])
        if self.loaded[2] != study["population"]:
            raise ValueError("Baseline data/prediction fingerprints changed")
        self.provider = HistoryProvider(study["config"], self.loaded[1]["task2"])
        self.body = BodyAdapter(study["body_checkpoint"], BASE_SHA, device, expected_group="context")
        self.body.model.requires_grad_(False)
        self.router = ContextRouter(load_settings(), machine_mode="fir25", body_adapter=self.body)

    def close(self):
        self.provider.close()
        for item in self.loaded[1].values():
            for k in ("prediction", "target", "quality"):
                item[k]._mmap.close()

    @torch.no_grad()
    def evaluate(self, model, output, group, epoch, *, keep_predictions=False, verify_zero=False):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        if model is not None:
            model.eval()
        scale = np.asarray(self.study["scale_uV"], float)
        tasks, pairs, diagnostic, statuses = {}, [], [], []
        started = time.perf_counter()
        for task, item in self.loaded[1].items():
            folder = output / task
            folder.mkdir()
            path = folder / "prediction_uV.npy"
            pred = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=item["prediction"].shape)
            assigned = set()
            for entry in (e for e in self.study["validation"] if e["task"] == task):
                ids = entry["window_indices"]
                row = item["rows"][ids[0]]
                with np.load(self.cache_root / entry["file"], allow_pickle=False) as data:
                    delta = (np.zeros((11, 6000), np.float32) if model is None else
                             (model(**visible_batch(data, self.device, scale[0]))[0].cpu().numpy() - data["base_slow"]))
                    if not np.isfinite(delta).all():
                        raise FloatingPointError("Nonfinite correction")
                    base = correct_base(data["base_uV"], delta, scale)
                    if verify_zero and (np.any(delta != 0) or not np.array_equal(base, data["base_uV"])):
                        raise AssertionError("Epoch0 must reproduce base bit-for-bit")
                    history = self.provider(row) if task == "task2" else None
                    enhanced, status = self.router.apply(base, data["anchor_uV"], task_id=task,
                        input_type=row["input_type"], history=history)
                    if not np.array_equal(enhanced[0], data["base_uV"][0]):
                        raise AssertionError("Slow residual/context changed the original I head")
                    for j, index in enumerate(ids):
                        if index in assigned:
                            raise ValueError("Duplicate validation window")
                        assigned.add(index)
                        pred[index] = enhanced[:, j * 5000:(j + 1) * 5000]
                    pairs.append(_record_metrics(enhanced, data["target_uV"], data["mask"], row, task))
                    diagnostic.extend(slow_diagnostics(entry, data, delta, scale))
                    statuses.append({"task_id": task, "pair_id": row["pair_id"], **status})
            if assigned != set(range(len(item["rows"]))):
                raise ValueError("Incomplete validation record coverage")
            pred.flush()
            overall, details = _score(pred, item, task)
            from ecg12gen.evaluate import write_report
            write_report(folder, overall, details, title=f"slowjoint {group} epoch{epoch} {task}")
            csv_write(folder / "window_metadata.csv", item["rows"])
            write(folder / "source_arrays.json", {"baseline_evaluation": self.study["baseline_evaluation"],
                "target_and_QC": str(Path(self.study["baseline_evaluation"]) / task),
                "population": self.study["population"][task]})
            tasks[task] = overall
            pred._mmap.close()
            if not keep_predictions:
                path.unlink()
        score = scores_from_summaries(tasks)
        row = {"group": group, "epoch": epoch, "task1_r": score["task1_r_missing11"],
            "machine_r": tasks["task2"]["ecg_machine_d6_r_missing11"],
            "body_r": tasks["task2"]["body_scale_d6_r_missing11"],
            "task2_r": score["task2_r_missing11"], "mean_r": score["main_correlation"],
            "delta_reference": score["main_correlation"] - self.study["reference"]["mean_r"],
            "elapsed_seconds": time.perf_counter() - started}
        differences = {k: row[k] - self.study["reference"][k]
                       for k in ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")}
        write_score_report(output, score)
        csv_write(output / "pair_metrics.csv", pairs)
        csv_write(output / "slow_diagnostics.csv", diagnostic)
        fields = list(dict.fromkeys(k for s in statuses for k in s))
        csv_write(output / "context_records.csv", [{k: s.get(k) for k in fields} for s in statuses])
        write(output / "evaluation.json", {"protocol": PROTOCOL, "completed": True, "summary": row,
            "score": score, "sampling": self.study["sampling"], "slow_beta": .75,
            "population": self.study["population"], "target_used_at_inference": False,
            "context_weights_fixed": True, "local_and_global_joint": True, "predictions_retained": keep_predictions,
            "epoch_zero_deltas": differences if verify_zero else None})
        if verify_zero and any(abs(d) > 1e-7 for d in differences.values()):
            raise ValueError("Epoch0 failed to reproduce Final reference; report retained")
        print(row, flush=True)
        return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", required=True, help="Joint best_r.pt/last.pt")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--keep-predictions", action="store_true")
    args = p.parse_args()
    device = device_from_name(args.device)
    study = load_study(args.run)
    model, cp = load_model(args.checkpoint, study, Path(args.run) / "study.json", device)
    evaluator = Evaluator(args.run, study, device)
    try:
        evaluator.evaluate(model, args.output, cp["group"], cp["epoch"],
            keep_predictions=args.keep_predictions, verify_zero=False)
        report = Path(args.output) / "evaluation.json"
        from baselines.B5_best.common import read
        value = read(report)
        value.update(checkpoint_sha256=sha(args.checkpoint), checkpoint_stage=cp["stage"],
                     study_sha256=sha(Path(args.run) / "study.json"))
        write(report, value)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
