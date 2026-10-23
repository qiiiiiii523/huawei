"""Whole-record missing11 correlation, pair/target and group equal weighting."""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import GROUPS, load_record
from .inference import predict_record

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
    if stage == "huawei40":
        if set(means) != set(GROUPS):
            raise ValueError("Huawei best selection requires task1, machine and body validation groups")
        score = .5 * means["task1"] + .25 * means["machine"] + .25 * means["body"]
    else:
        if not means:
            raise ValueError("No valid validation groups")
        score = float(np.mean(list(means.values())))
    return {"r": score, "groups": means, "lead_groups": lead_means,
            "targets": len(targets), "pairs": len(rows), "score_protocol": SCORE_PROTOCOL}


def evaluate(model, manifest, config, stage):
    rows = []
    target_masks, target_identity = {}, {}
    for row in manifest["records"]:
        if row["split"] != "validation":
            continue
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
        if stage != "huawei40":
            real_length = target.shape[-1] // 5000 * 5000
            if real_length < config.window_seconds * 500:
                continue
            anchor, target = anchor[:, :real_length], target[:, :real_length]
        if target.shape[-1] % 5000:
            raise ValueError("Prepared validation record must have complete real10s cores")
        prediction = predict_record(model, anchor, manifest["scales_uV"], row["record_id"], metadata, config).cpu().numpy()
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
        rows.append({"record_id": row["record_id"], "group": row["group"], "target_record_id": key,
                     "r": float(np.mean(correlations)) if correlations else None, "leads": by_lead})
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
