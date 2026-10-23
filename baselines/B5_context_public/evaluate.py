"""Machine-only paired generation, fixed slow/context, unchanged other routes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from baselines.B5.runtime import device_from_name
from baselines.B5_best.common import write, csv_write, sha
from baselines.B5_best.combine import combine, upsample_native
from baselines.B5_best_context.adapters import ContextRouter, HistoryRecord
from baselines.B5_best_context.evaluate import _record_metrics
from .sampling import SAMPLING, sample
from .data import load_study, read_record, tensor_batch, model_inputs
from .common import load_candidate
from . import PROTOCOL


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@torch.no_grad()
def generate(model, record, entry, scales, device, context=False, internal_history=None):
    parts = []
    for start in range(0, 60000, 5000):
        batch = tensor_batch([record], [start], scales, device, context, [internal_history] if internal_history is not None else None)
        inputs = model_inputs(batch)
        parts.append(sample(model, inputs, [entry["target_record_id"] + ":" + str(start)],
                            context_enabled=context)[0].cpu().numpy() * scales[:, None])
    fm = np.concatenate(parts, -1).astype(np.float32)
    base = combine(fm, upsample_native(record["fixed_slow"]), slow_beta=.75)
    # Terminal history is always the correct one, irrespective of the internal-history intervention.
    history = HistoryRecord("ecg_machine_d6", entry["input_record_id"], record["history_uV"], record["history_mask"])
    output, status = ContextRouter(machine_mode="fir25").apply(base, record["anchor_uV"],
        task_id="task2", input_type="ecg_machine_d6", history=history)
    return output, status


@torch.no_grad()
def evaluate(model, root, study, output, context=False, variant="correct", checkpoint_sha=None, keep=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    model.eval(); device = next(model.parameters()).device
    scale = np.asarray(study["scale_uV"], np.float32)
    entries = study["records"]["huawei_validation"]
    all_records = [read_record(root, e) for e in entries]
    # Prefer equally available donors from another patient; fixed mapping, no outcome selection.
    donors = []
    for i, e in enumerate(entries):
        own_available = bool(all_records[i]["history_mask"].any())
        candidates = [j for j in range(len(entries)) if entries[j]["subject_id"] != e["subject_id"]
                      and bool(all_records[j]["history_mask"].any()) == own_available]
        candidates.sort(key=lambda j: (abs(int(all_records[j]["history_mask"].sum()) - int(all_records[i]["history_mask"].sum())), entries[j]["target_record_id"]))
        if variant == "shuffled" and not candidates:
            raise ValueError("Cannot match a cross-patient donor with the same history availability")
        donors.append(candidates[0] if candidates else i)
    metrics, statuses = [], []
    for i, (entry, record) in enumerate(zip(entries, all_records)):
        hist = all_records[donors[i]] if variant == "shuffled" else record
        if variant == "disabled":
            hist = dict(record, history_uV=np.zeros((6, 1), np.float32), history_mask=np.zeros((6, 1), bool))
        prediction, status = generate(model, record, entry, scale, device, context, hist)
        row = _record_metrics(prediction, record["target_uV"], record["mask"], entry, "task2")
        metrics.append(row)
        statuses.append(dict(pair_id=entry["pair_id"], internal_variant=variant,
            internal_donor=entries[donors[i]]["target_record_id"] if variant == "shuffled" else entry["target_record_id"],
            terminal_history_id=entry["input_record_id"], **status))
        if keep: np.save(output / f"prediction_{i:04d}.npy", prediction)
    grouped = {}
    for row in metrics:
        if row["r_missing11"] is not None: grouped.setdefault(row["target_record_id"], []).append(row["r_missing11"])
    if not grouped: raise ValueError("No reliable machine targets scored")
    machine = float(np.mean([np.mean(v) for v in grouped.values()]))
    task1, body = study["fixed_routes"]["task1_r"], study["fixed_routes"]["body_r"]
    result = dict(protocol=PROTOCOL, completed=True, route="ecg_machine_d6", full_record_seconds=120,
        slow_beta=.75, sampling=dict(SAMPLING), machine_r=machine, task1_r=task1, body_r=body,
        task2_r=(machine + body) / 2, mean_r=.5 * task1 + .25 * machine + .25 * body,
        unchanged_routes_source=study["stable_source_report"], machine_targets=len(grouped),
        internal_context=context, history_variant=variant, terminal_history="correct_unchanged",
        checkpoint_sha256=checkpoint_sha,
        population_sha256=fingerprint([(e["subject_id"], e["target_record_id"], e["sha256"]) for e in entries]),
        record_keys_sha256=fingerprint([e["target_record_id"] + ":" + str(s) for e in entries for s in range(0, 60000, 5000)]),
        scales_sha256=fingerprint(study["scale_uV"]),
        slow_assets_sha256=fingerprint(study["assets"]), terminal_rules_sha256=fingerprint("machine_fir25_v1"))
    csv_write(output / "record_metrics.csv", metrics); csv_write(output / "context_records.csv", statuses)
    write(output / "evaluation.json", result)
    print({k: result[k] for k in ("machine_r", "mean_r", "history_variant")}, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "checkpoint", "output"): p.add_argument("--" + k, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--history", choices=("correct", "shuffled", "disabled"), default="correct")
    p.add_argument("--keep-predictions", action="store_true")
    a = p.parse_args(); study = load_study(a.run)
    model, cp = load_candidate(a.checkpoint, study, a.run, device_from_name(a.device))
    evaluate(model, a.run, study, a.output, cp["arm"] == "context_public", a.history, sha(a.checkpoint), a.keep_predictions)


if __name__ == "__main__": main()
