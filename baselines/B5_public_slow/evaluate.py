"""Recombine fixed control100 predictions; score using latest reliable-I/target-QC rules."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ecg12gen.evaluate import evaluate_record_predictions, write_report
from baselines.B5.scoring import scores_from_summaries, diagnostic_summary, write_score_report
from baselines.B5_best.combine import combine, upsample_native
from baselines.B5_best.common import read, write, csv_write, sha
from baselines.B5.runtime import device_from_name
from .common import read_record, load_study, model_from_bundle, empty_output
from .core import batch_record, generate, objective
from . import PROTOCOL


@torch.no_grad()
def public_diagnostic(model, study, bundle, device):
    item = study["public"]
    entries = read(item["manifest"])["validation"]
    model.eval()
    scales = torch.tensor(bundle["fm"]["scales"]["d12"], device=device)
    total, correlations, rmses = [], [], []
    for entry in entries:
        record = read_record(item["root"], entry)
        batch = batch_record(record, device)
        prediction = generate(model, {k: batch[k] for k in ("anchor", "numeric", "sex", "field_mask", "age_topcoded")})
        loss, _ = objective(prediction, batch, scales, study["residual_weight"], 0.)
        total.append(float(loss))
        predicted = prediction[0].cpu().numpy()
        for lead in np.flatnonzero(record["mask"][1:]):
            p, y = predicted[lead].astype(float), record["slow_target"][lead].astype(float)
            a, b = p - p.mean(), y - y.mean()
            if np.sum(b * b) > 1e-12:
                r = np.sum(a * b) / max(np.sqrt(np.sum(a * a) * np.sum(b * b)), 1e-12)
                correlations.append(float(r))
            rmses.append(float(np.sqrt(np.mean((p - y) ** 2)) * float(scales[lead + 1])))
    return {"records": len(entries), "record_mean_loss": float(np.mean(total)),
            "diagnostic_slow_r": float(np.mean(correlations)) if correlations else None,
            "diagnostic_slow_rmse_uV": float(np.mean(rmses)) if rmses else None,
            "selection_metric": False}


@torch.no_grad()
def evaluate(model, study, bundle, device, output, save_arrays=False):
    output = empty_output(output)
    model.eval()
    item = study["huawei"]
    manifest = read(item["manifest"])
    scales = np.asarray(bundle["fm"]["scales"]["d12"], np.float32)
    learned = {}
    lookup = {e["cache_key"]: e for e in manifest["validation"]}
    for entry in manifest["validation"]:
        key = entry["target_record_id"]
        if key not in learned:
            record = read_record(item["root"], entry)
            inputs = batch_record(record, device, labels=False)
            native = generate(model, inputs)[0].cpu().numpy()
            learned[key] = upsample_native(native.astype(float) * scales[1:, None].astype(float))
    summaries = {}
    for task, rows in manifest["validation_rows"].items():
        folder = output / task
        folder.mkdir()
        shape = (len(rows), 12, 5000)
        prediction, target = np.empty(shape, np.float32), np.empty(shape, np.float32)
        quality = np.empty((len(rows), 12), bool)
        grouped = {}
        for i, row in enumerate(rows):
            grouped.setdefault(task + ":" + str(row["pair_id"]), []).append(i)
        for pair, indices in grouped.items():
            entry = lookup[pair]
            record = read_record(item["root"], entry)
            generated = combine(record["fm_uV"], learned[entry["target_record_id"]], .75)
            for i in indices:
                start = int(rows[i]["start_sample_500hz"])
                prediction[i] = generated[:, start:start + 5000]
                target[i] = record["target_uV"][:, start:start + 5000]
                quality[i] = record["mask"]
        overall, details = evaluate_record_predictions(prediction, target, task, rows,
            target_quality_mask=quality, input_quality_mask=quality[:, 0], require_reliable_input=True,
            input_quality_source="same_record_target_I")
        overall = diagnostic_summary(overall)
        write_report(folder, overall, details, title="PublicSlow full120s reliable-I target-lead QC")
        csv_write(folder / "window_metadata.csv", rows)
        if save_arrays:
            for name, array in (("prediction_uV", prediction), ("target_uV", target), ("target_quality_mask", quality)):
                np.save(folder / (name + ".npy"), array, allow_pickle=False)
        summaries[task] = overall
        del prediction, target, quality
    score = scores_from_summaries(summaries)
    write_score_report(output, score)
    write(output / "evaluation.json", {"completed": True, "score": score, "sampling": study["sampling"],
          "beta": .75, "FM_frozen": True, "target_used_at_inference": False,
          "input_quality_source": "same_record_target_I", "require_reliable_input": True})
    return score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-arrays", action="store_true")
    a = p.parse_args()
    study, bundle = load_study(a.study, deep=True)
    cp = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if cp.get("protocol") != PROTOCOL or cp.get("study_sha256") != sha(a.study):
        raise ValueError("Checkpoint belongs to another study")
    device = device_from_name(a.device)
    model = model_from_bundle(bundle, device)
    model.load_state_dict(cp["model"], strict=True)
    print(evaluate(model, study, bundle, device, a.output, a.save_arrays), flush=True)


if __name__ == "__main__":
    main()
