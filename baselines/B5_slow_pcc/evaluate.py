"""Full-record original-uV hybrid scoring with unchanged QC and device weights."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg12gen.evaluate import evaluate_record_predictions, write_report
from baselines.B5.runtime import device_from_name
from baselines.B5_best.common import csv_write, write, sha
from baselines.B5_best.combine import combine, upsample_native
from baselines.B5_best.scoring import diagnostic_summary, scores_from_summaries, write_score_report
from baselines.B5_best.diagnostics_core import corr
from .common import Records, load_cache, load_checkpoint, validation_key
from .core import generate, model_inputs, device_batch


@torch.no_grad()
def evaluate(model, bundle, study, cache_root, manifest, device, output, group, epoch, initial=False):
    output, cache_root = Path(output), Path(cache_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty evaluation output; keep tee log beside it")
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    unique = {}
    for entry in manifest["validation"]:
        unique.setdefault(entry["target_record_id"], entry)
    unique_entries = list(unique.values())
    dataset = Records(cache_root, unique_entries)
    loader = DataLoader(dataset, batch_size=study["training"]["batch_records"], shuffle=False, num_workers=0)
    learned, native_rows = {}, []
    scales = np.asarray(bundle["fm"]["scales"]["d12"], np.float32)
    offset = 0
    for batch in loader:
        inputs = device_batch(model_inputs(batch), device)
        native = generate(model, inputs).cpu().numpy()
        for j in range(len(native)):
            entry = unique_entries[offset + j]
            restored = upsample_native(native[j].astype(np.float64) * scales[1:, None].astype(np.float64))
            learned[entry["target_record_id"]] = restored
            for lead in range(11):
                if bool(batch["mask"][j, lead + 1]):
                    y, p = batch["slow_target"][j, lead].numpy(), native[j, lead]
                    native_rows.append({"lead_index": lead, "r": corr(p, y),
                          "rmse_uV": float(np.sqrt(np.mean((p.astype(float) - y) ** 2)) * scales[lead + 1])})
        offset += len(native)
    lookup = {r["cache_key"]: r for r in manifest["validation"]}
    summaries = {}
    for task, rows in manifest["validation_rows"].items():
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            arrays = {}
            for name, shape, dtype in (("prediction_uV", (len(rows), 12, 5000), np.float32),
                                      ("target_uV", (len(rows), 12, 5000), np.float32),
                                      ("target_quality_mask", (len(rows), 12), bool)):
                value = np.lib.format.open_memmap(folder / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
                stack.callback(value._mmap.close)
                arrays[name] = value
            opened = {}
            for i, row in enumerate(rows):
                key = validation_key(task, row["pair_id"])
                start = int(row["start_sample_500hz"])
                if key not in opened:
                    archive = stack.enter_context(np.load(cache_root / lookup[key]["file"], allow_pickle=False))
                    opened[key] = {"target": archive["target_uV"], "mask": archive["mask"],
                                   "prediction": combine(archive["fm_uV"], learned[row["target_record_id"]], study["beta"])}
                record = opened[key]
                arrays["prediction_uV"][i] = record["prediction"][:, start:start + 5000]
                arrays["target_uV"][i] = record["target"][:, start:start + 5000]
                arrays["target_quality_mask"][i] = record["mask"]
            for value in arrays.values():
                value.flush()
            overall, details = evaluate_record_predictions(arrays["prediction_uV"], arrays["target_uV"], task,
                              rows, target_quality_mask=arrays["target_quality_mask"])
            overall = diagnostic_summary(overall)
            overall.update(sampling=study["sampling"], model="B5-best-SlowPCC-output-hybrid",
                           FM_frozen=True, beta=study["beta"], target_used_at_inference=False)
            write_report(folder, overall, details, title="SlowPCC full-record QC r")
            csv_write(folder / "window_metadata.csv", rows)
            summaries[task] = overall
    score = scores_from_summaries(summaries)
    write_score_report(output, score)
    native_summary = {}
    for field in ("r", "rmse_uV"):
        per_lead = []
        for lead in range(11):
            values = [r[field] for r in native_rows if r["lead_index"] == lead and r[field] is not None]
            if values:
                per_lead.append(float(np.mean(values)))
        native_summary["slow_" + field] = float(np.mean(per_lead)) if per_lead else None
    reference = bundle["reference_qc_r"]
    matched = all(abs(score[k] - reference[k]) <= bundle["reference_tolerance"] for k in reference) if initial else None
    write(output / "experiment.json", {"group": group, "epoch": epoch, "score": score,
          "slow_diagnostic": native_summary, "FM_frozen": True, "slow_CNN_trainable": True,
          "beta": study["beta"], "reference_matched": matched, "sampling": study["sampling"],
          "selection_is_final_hybrid_r": True, "training_validation_cache_separated": True,
          "validation_FM_handling": "preserve_historical_predictions_per_task_pair"})
    write(output / "evaluation.json", {"completed": True, "group": group, "epoch": epoch,
          "score": score, "sampling": study["sampling"], "FM_frozen": True,
          "target_used_at_inference": False, "reference_matched": matched})
    if initial and not matched:
        raise ValueError("Initial hybrid does not reproduce B5-best; keep report and inspect data/cache before training")
    print({"group": group, "epoch": epoch, **score, **native_summary}, flush=True)
    return score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = device_from_name(args.device)
    checkpoint, study, bundle, model = load_checkpoint(args.checkpoint, args.study, device)
    manifest = load_cache(args.cache, args.study)
    if checkpoint["cache_sha256"] != sha(Path(args.cache) / "cache.json"):
        raise ValueError("Checkpoint/cache mismatch")
    evaluate(model, bundle, study, args.cache, manifest, device, args.output,
             checkpoint["group"], checkpoint["epoch"])


if __name__ == "__main__":
    main()
