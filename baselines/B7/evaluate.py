"""Whole-record missing11 correlation, pair/target and group equal weighting."""
from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import numpy as np
from .data import GROUPS, load_record
from .inference import predict_record


def pearson(prediction, target):
    p, y = np.asarray(prediction, dtype="float64"), np.asarray(target, dtype="float64")
    p, y = p - p.mean(), y - y.mean()
    den = np.linalg.norm(p) * np.linalg.norm(y)
    # Constant failed predictions score zero, never disappear from the evaluation.
    return float(np.dot(p, y) / den) if den > 0 else 0.0


def aggregate(rows, stage):
    by_target = defaultdict(list)
    for row in rows:
        by_target[(row["group"], row["target_record_id"])].append(row["r"])
    groups = defaultdict(list)
    for (group, _), values in by_target.items():
        groups[group].append(float(np.mean(values)))
    means = {k: float(np.mean(v)) for k, v in groups.items()}
    if stage == "huawei40":
        if set(means) != set(GROUPS):
            raise ValueError("Huawei best selection requires task1, machine and body validation groups")
        score = .5 * means["task1"] + .25 * means["machine"] + .25 * means["body"]
    else:
        if not means:
            raise ValueError("No valid validation groups")
        score = float(np.mean(list(means.values())))
    return {"r": score, "groups": means, "targets": len(by_target), "pairs": len(rows)}


def evaluate(model, manifest, config, stage):
    rows = []
    target_masks = {}
    for row in manifest["records"]:
        if row["split"] != "validation":
            continue
        anchor, target, mask, metadata = load_record(row)
        key = row["target_record_id"]
        if key in target_masks and not np.array_equal(target_masks[key], mask):
            raise ValueError("QC differs for the same physical target")
        target_masks[key] = mask.copy()
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
            if mask[j] and np.var(target[j].astype("float64")) > 0:
                value = pearson(prediction[j], target[j])
                correlations.append(value)
                by_lead[str(j)] = {"r": value, "rmse_uV": float(np.sqrt(np.mean(
                    (prediction[j].astype("float64") - target[j]) ** 2)))}
        if not correlations:
            continue
        rows.append({"record_id": row["record_id"], "group": row["group"], "target_record_id": key,
                     "r": float(np.mean(correlations)), "leads": by_lead})
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
