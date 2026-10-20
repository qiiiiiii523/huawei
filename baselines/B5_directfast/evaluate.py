"""Evaluate F0/F1/D1 with baseline, machine-only and explicit frozen-body routes."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch

from baselines.B5.runtime import device_from_name
from baselines.B5.scoring import scores_from_summaries, write_score_report
from baselines.B5_best.common import write, csv_write, sha, state_sha
from baselines.B5_best_context.evaluate import load_baseline, HistoryProvider, _score, _record_metrics
from baselines.B5_final.profile import BASE_SHA, BODY_SHA
from . import PROTOCOL, ROUTES
from .common import load_study, load_candidate, array_fields
from .context_transfer import routers
from .inference import predict_base


class Evaluator:
    def __init__(self, run, study, device, batch_size=4):
        self.run, self.study, self.device, self.batch_size = Path(run), study, device, batch_size
        for key, expected in (("base_checkpoint", BASE_SHA), ("body_checkpoint", BODY_SHA)):
            if sha(study[key]) != expected:
                raise ValueError("Original Final assets changed")
        if sha(study["config"]) != study["config_sha256"]:
            raise ValueError("Live Huawei configuration changed")
        self.loaded = load_baseline(study["baseline_evaluation"])
        if self.loaded[2] != study["population"]:
            raise ValueError("Baseline population or predictions changed")
        self.provider = HistoryProvider(study["config"], self.loaded[1]["task2"])
        self.routes = routers(study["body_checkpoint"], device, group="F0", predictor_sha=BASE_SHA)

    def close(self):
        self.provider.close()
        for item in self.loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()

    @torch.no_grad()
    def evaluate(self, model, output, group, epoch, keep_predictions=False, packaged=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        if model is not None:
            model.eval()
        identity = BASE_SHA if model is None else state_sha(model.state_dict())
        transfer = self.routes["both"].body_adapter
        transfer.group, transfer.predictor_sha = group, identity
        scale = np.asarray(self.study["scale_uV"], np.float32)
        tasks, pair_rows, statuses = {route: {} for route in ROUTES}, [], []
        begun = time.perf_counter()
        for task, item in self.loaded[1].items():
            predictions = {}
            for route in ROUTES:
                folder = output / route / task
                folder.mkdir(parents=True)
                predictions[route] = np.lib.format.open_memmap(folder / "prediction_uV.npy", mode="w+",
                    dtype=np.float32, shape=item["prediction"].shape)
            assigned = set()
            for entry in (e for e in self.study["validation"] if e["task"] == task):
                ids = entry["window_indices"]
                row = item["rows"][ids[0]]
                with np.load(self.run / entry["file"], allow_pickle=False) as data:
                    base = (packaged.predict_base_encoded(data["anchor_uV"], array_fields(data),
                                                          entry["target_record_id"], self.batch_size)
                            if packaged is not None else data["base_uV"].copy() if model is None else
                            predict_base(model, group, data["anchor_uV"], array_fields(data),
                                entry["target_record_id"], scale, data["slow_uV"], data["original_i_uV"],
                                self.device, self.batch_size))
                    history = self.provider(row) if task == "task2" else None
                    for route in ROUTES:
                        router = self.routes[route] if packaged is None else packaged.routes[route]
                        prediction, status = router.apply(base, data["anchor_uV"], task_id=task,
                            input_type=row["input_type"], history=history)
                        same_i = (np.array_equal(prediction[0], data["original_i_uV"][0]) if packaged is None else
                                  np.allclose(prediction[0], data["original_i_uV"][0], rtol=2e-5, atol=2e-3))
                        if not same_i:
                            raise ValueError("Experiment changed original I head")
                        for j, i in enumerate(ids):
                            predictions[route][i] = prediction[:, j * 5000:(j + 1) * 5000]
                        pair_rows.append({"route": route, **_record_metrics(prediction, data["target_uV"],
                                                                         data["mask"], row, task)})
                        statuses.append({"route": route, "task_id": task, "pair_id": row["pair_id"], **status})
                if assigned.intersection(ids):
                    raise ValueError("Duplicated validation window")
                assigned.update(ids)
            if assigned != set(range(len(item["rows"]))):
                raise ValueError("Incomplete validation coverage")
            for route, prediction in predictions.items():
                folder = output / route / task
                prediction.flush()
                overall, details = _score(prediction, item, task)
                from ecg12gen.evaluate import write_report
                write_report(folder, overall, details, title=f"directfast {group} {route} epoch{epoch}")
                csv_write(folder / "window_metadata.csv", item["rows"])
                write(folder / "source_arrays.json", {"baseline_evaluation": self.study["baseline_evaluation"],
                                                       "population": self.study["population"][task]})
                tasks[route][task] = overall
                prediction._mmap.close()
                if not keep_predictions:
                    (folder / "prediction_uV.npy").unlink()
        summaries = {}
        for route in ROUTES:
            score = scores_from_summaries(tasks[route])
            summaries[route] = {"group": group, "epoch": epoch, "route": route,
                "task1_r": score["task1_r_missing11"], "task2_r": score["task2_r_missing11"],
                "machine_r": tasks[route]["task2"]["ecg_machine_d6_r_missing11"],
                "body_r": tasks[route]["task2"]["body_scale_d6_r_missing11"],
                "mean_r": score["main_correlation"]}
            write_score_report(output / route, score)
        csv_write(output / "summary.csv", list(summaries.values()))
        csv_write(output / "pair_metrics.csv", pair_rows)
        keys = list(dict.fromkeys(k for status in statuses for k in status))
        csv_write(output / "context_records.csv", [{k: status.get(k) for k in keys} for status in statuses])
        if group == "F0":
            for key in ("task1_r", "task2_r", "machine_r", "body_r", "mean_r"):
                if abs(summaries["both"][key] - self.study["reference"][key]) > 1e-7:
                    raise ValueError("F0 does not reproduce pinned Final; do not train")
        selection = self.study["experiment"]["training"]["selection_route"]
        report = {"protocol": PROTOCOL, "completed": True, "group": group, "epoch": epoch,
            "summary": summaries, "selection_route": selection, "selected_score": summaries[selection]["mean_r"],
            "base_sha256": BASE_SHA, "body_sha256": BODY_SHA, "fast_state_sha256": identity,
            "population": self.study["population"], "target_used_at_inference": False,
            "body_transfer": group != "F0", "elapsed_seconds": time.perf_counter() - begun,
            "predictions_retained": keep_predictions,
            "inference_path": "portable_package_visible_inputs_only" if packaged is not None else "cached_frozen_components"}
        write(output / "evaluation.json", report)
        print(report, flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--candidate", help="Omit candidate and package to evaluate original F0")
    source.add_argument("--package", help="Recompute P/I from raw visible inputs using the deployable package")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--keep-predictions", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Positive batch size required")
    device = device_from_name(args.device)
    study = load_study(args.run)
    packaged = None
    model, value = (load_candidate(args.candidate, device, study, Path(args.run) / "study.json")
                    if args.candidate else (None, {"group": "F0", "epoch": 0}))
    if args.package:
        from .predict import Predictor
        packaged = Predictor(args.package, device)
        model, value = packaged.model, packaged.payload["candidate"]
        if value["scale_uV"] != study["scale_uV"] or value["model_config"] != study["model_config"]:
            raise ValueError("Package does not match prepared study")
    evaluator = Evaluator(args.run, study, device, args.batch_size)
    try:
        evaluator.evaluate(model, args.output, value["group"], value["epoch"], args.keep_predictions, packaged)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
