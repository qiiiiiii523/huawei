"""Whole-record missing11 correlation, pair/target and group equal weighting."""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import time
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
    if stage in ("huawei40", "huawei120"):
        if set(means) != set(GROUPS):
            raise ValueError("Huawei best selection requires task1, machine and body validation groups")
        score = .5 * means["task1"] + .25 * means["machine"] + .25 * means["body"]
    else:
        if not means:
            raise ValueError("No valid validation groups")
        score = float(np.mean(list(means.values())))
    return {"r": score, "groups": means, "lead_groups": lead_means,
            "targets": len(targets), "pairs": len(rows), "score_protocol": SCORE_PROTOCOL}


def evaluate(model, manifest, config, stage, *, batch_size=1):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive integer validation batch_size required")
    rows = []
    total = sum(r['split'] == 'validation' for r in manifest['records'])
    scanned = 0
    started = time.monotonic()
    print(f'[validation] records={total} window={config.window_seconds}s core={config.core_seconds}s K={config.samples}', flush=True)
    target_masks, target_identity = {}, {}
    pending = []

    def flush():
        predictions = predict_records(model, [item[1] for item in pending], manifest["scales_uV"],
            [item[0]["record_id"] for item in pending], [item[4] for item in pending], config,
            batch_size=batch_size)
        for (row, _, target, mask, _), output in zip(pending, predictions):
            prediction = output.cpu().numpy()
            correlations, by_lead = [], {}
            for j in range(1, 12):
                if mask[j]:
                    value = pearson(prediction[j], target[j])
                    if np.isfinite(value):
                        correlations.append(value)
                    by_lead[str(j)] = {"r": value if np.isfinite(value) else None,
                        "rmse_uV": float(np.sqrt(np.mean(
                        (prediction[j].astype("float64") - target[j]) ** 2)))}
            if not by_lead:
                continue
            rows.append({"record_id": row["record_id"], "group": row["group"],
                         "target_record_id": row["target_record_id"],
                         "r": float(np.mean(correlations)) if correlations else None, "leads": by_lead})
        pending.clear()
        elapsed = time.monotonic()-started
        eta = elapsed * (total-scanned)/scanned if scanned else 0.
        print(f'[validation] scanned={scanned}/{total} scored_pairs={len(rows)} '
              f'elapsed_seconds={elapsed:.1f} estimated_remaining_seconds={eta:.1f}', flush=True)

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
        if stage in ("huawei40", "huawei120") and target.shape[-1] != 60000:
            raise ValueError("Huawei validation must contain real full120s, not padded targets")
        if stage not in ("huawei40", "huawei120"):
            real_length = target.shape[-1] // 5000 * 5000
            if real_length < config.window_seconds * 500:
                continue
            anchor, target = anchor[:, :real_length], target[:, :real_length]
        if target.shape[-1] % 5000:
            raise ValueError("Prepared validation record must have complete real10s cores")
        pending.append((row, anchor, target, mask, metadata))
        if len(pending) == batch_size:
            flush()
    if pending:
        flush()
    return aggregate(rows, stage), rows


def main():
    from .checkpoint import load_checkpoint, model_from_checkpoint
    from .config import SamplingConfig
    from .data import read_manifest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32, help="Maximum concurrent ODE trajectories")
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError("Use a new evaluation output")
    value, manifest = load_checkpoint(a.checkpoint), read_manifest(a.manifest)
    if list(value["scales_uV"]) != list(manifest["scales_uV"]):
        raise ValueError("Evaluation scales do not match checkpoint")
    metrics, rows = evaluate(model_from_checkpoint(value, a.device), manifest,
                             SamplingConfig(**value["sampling"]), value["stage"], batch_size=a.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "records": rows, "batch_size": a.batch_size},
                                 ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
