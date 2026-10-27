"""Whole-record missing11 correlation, pair/target and group equal weighting."""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import GROUPS, load_record
from .inference import predict_records

SCORE_PROTOCOL = "B7-full-record-lead-target-pair-macro-v2"


def pearson(prediction, target):
    p, y = np.asarray(prediction, dtype="float64"), np.asarray(target, dtype="float64")
    p, y = p - p.mean(), y - y.mean()
    den = np.linalg.norm(p) * np.linalg.norm(y)
    # Preserve the existing scorer's undefined-Pearson handling (do not invent a zero r).
    return float(np.dot(p, y) / den) if den > 0 else float("nan")


def limb_diagnostics(anchor, prediction, target, mask):
    """Descriptive only; never changes best-checkpoint selection or lead formulas."""
    result = {}
    for j, name in ((1, 'II'), (2, 'III'), (5, 'aVF')):
        if mask[j]:
            y = target[j].astype('float64'); p = prediction[j].astype('float64')
            std = float(y.std())
            result[name] = {'target_std_uV': std, 'pred_std_uV': float(p.std()),
                            'std_ratio': float(p.std()/std) if std > 0 else None,
                            'normalized_mae': float(np.mean(abs(p-y))/max(std,20.))}
    if mask[1]:
        a = anchor[0].astype('float64'); y = target[1].astype('float64'); p = prediction[1].astype('float64')
        a -= a.mean(); y -= y.mean(); p -= p.mean()
        aa = float(a @ a)
        if aa / len(a) >= 20.**2 and y.std() > 0:
            actual, predicted = float(a@y/aa), float(a@p/aa)
            true_r, pred_r = pearson(a,y), pearson(a,p)
            result['I_II'] = {'true_r': true_r if np.isfinite(true_r) else None,
                              'pred_r': pred_r if np.isfinite(pred_r) else None,
                              'true_projection': actual, 'pred_projection': predicted,
                              'projection_component_error': abs(actual-predicted)*float(a.std())/max(float(y.std()),20.)}
    return result


def aggregate(rows, stage):
    # QC differs by lead: lead means must be computed before the final 11-lead mean.
    # Averaging each pair's available leads first would over-weight sparsely eligible targets.
    by_lead_target = defaultdict(list)
    group_names, targets = set(), set()
    for row in rows:
        group_names.add(row["group"])
        targets.add((row["group"], row["target_record_id"]))
        for lead, metrics in row["leads"].items():
            value = metrics["r"]
            if value is None:
                continue  # Undefined pair r, as in the established scorer.
            if not np.isfinite(value):
                raise ValueError("Nonfinite correlation cannot enter model selection")
            by_lead_target[(row["group"], lead, row["target_record_id"])].append(value)
    groups = defaultdict(lambda: defaultdict(list))
    for (group, lead, _), values in by_lead_target.items():
        groups[group][lead].append(float(np.mean(values)))
    lead_means, means = {}, {}
    for group in sorted(group_names):
        if set(groups[group]) != {str(j) for j in range(1, 12)}:
            raise ValueError(f"No defined QC-eligible targets for every missing lead in group {group}")
        lead_means[group] = {lead: float(np.mean(values)) for lead, values in groups[group].items()}
        means[group] = float(np.mean(list(lead_means[group].values())))
    if stage in ("huawei40", "huawei10"):
        if set(means) != set(GROUPS):
            raise ValueError("Huawei best selection requires task1, machine and body validation groups")
        score = .5 * means["task1"] + .25 * means["machine"] + .25 * means["body"]
    else:
        if not means:
            raise ValueError("No valid validation groups")
        score = float(np.mean(list(means.values())))
    return {"r": score, "groups": means, "lead_groups": lead_means,
            "targets": len(targets), "pairs": len(rows), "score_protocol": SCORE_PROTOCOL}


def _evaluate_records(model, manifest, config, stage):
    rows = []
    pending = []
    scanned = 0
    total_records = sum(r["split"] == "validation" for r in manifest["records"])
    print(f"[validation] START records={total_records} record_batch={config.validation_batch_size} "
          f"ode_batch={config.sample_batch_size} K={config.samples}", flush=True)
    def flush():
        if not pending:
            return
        from .position import view_metadata
        predictions = predict_records(model, [x[1] for x in pending], manifest["scales_uV"],
            [x[0]["record_id"] for x in pending], [view_metadata(x[4], x[0]) for x in pending], config).cpu().numpy()
        for (item, anchor, target, mask, _), prediction in zip(pending, predictions):
            correlations, by_lead = [], {}
            for j in range(1, 12):
                if mask[j]:
                    value = pearson(prediction[j], target[j])
                    if np.isfinite(value):
                        correlations.append(value)
                    by_lead[str(j)] = {"r": value if np.isfinite(value) else None,
                        "rmse_uV": float(np.sqrt(np.mean((prediction[j].astype("float64") - target[j]) ** 2)))}
            if by_lead:
                rows.append({"record_id": item["record_id"], "group": item["group"],
                    "target_record_id": item["target_record_id"],
                    "r": float(np.mean(correlations)) if correlations else None, "leads": by_lead,
                    "limb_diagnostics": limb_diagnostics(anchor, prediction, target, mask)})
        pending.clear()
        print(f"[validation] scored={len(rows)} scanned={scanned}/{total_records}", flush=True)
    target_masks, target_identity = {}, {}
    for row in manifest["records"]:
        if row["split"] != "validation":
            continue
        scanned += 1
        anchor, target, mask, metadata = load_record(row)
        key = row["target_record_id"]
        if key in target_masks and not np.array_equal(target_masks[key], mask):
            raise ValueError("QC differs for the same physical target")
        target_masks[key] = mask.copy()
        fingerprint = (row["subject_id"], target.shape, hashlib.sha256(
            np.where(mask[:, None], target, 0.).tobytes()).hexdigest())
        if key in target_identity and target_identity[key] != fingerprint:
            raise ValueError("Repeated physical target has different subject or waveform")
        target_identity[key] = fingerprint
        if not row["input_reliable"]:
            continue
        if stage == "huawei40" and target.shape[-1] != 60000:
            raise ValueError("Huawei validation must contain real full120s, not padded targets")
        if stage == "huawei10" and target.shape[-1] != 5000:
            raise ValueError("Huawei10 validation requires indexed real10s records")
        if stage != "huawei40":
            real_length = target.shape[-1] // 5000 * 5000
            if real_length < config.window_seconds * 500:
                continue
            anchor, target = anchor[:, :real_length], target[:, :real_length]
        if target.shape[-1] % 5000:
            raise ValueError("Prepared validation record must have complete real10s cores")
        if pending and pending[0][1].shape[-1] != anchor.shape[-1]:
            flush()
        pending.append((row, anchor, target, mask, metadata))
        if len(pending) >= config.validation_batch_size:
            flush()
    flush()
    return aggregate(rows, stage), rows



def evaluate(model, manifest, config, stage):
    """Public scores retain their protocol; every Huawei selection scores full120."""
    if stage not in ("huawei10", "huawei40"):
        return _evaluate_records(model, manifest, config, stage)
    from .huawei120 import full120_validation_manifest, full120_precision, HUAWEI_SELECTION_PROTOCOL
    full = full120_validation_manifest(manifest) if stage == "huawei10" else manifest
    with full120_precision():
        metrics, rows = _evaluate_records(model, full, config, "huawei40")
    metrics.update(scored_seconds=120, selection_protocol=HUAWEI_SELECTION_PROTOCOL,
                   prediction_window_seconds=config.window_seconds)
    return metrics, rows


def main():
    from .checkpoint import load_checkpoint, model_from_checkpoint
    from .config import SamplingConfig
    from .data import read_manifest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError("Use a new evaluation output")
    value, manifest = load_checkpoint(a.checkpoint), read_manifest(a.manifest)
    if list(value["scales_uV"]) != list(manifest["scales_uV"]):
        raise ValueError("Evaluation scales do not match checkpoint")
    metrics, rows = evaluate(model_from_checkpoint(value, a.device), manifest,
                             SamplingConfig(**value["sampling"]), value["stage"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
