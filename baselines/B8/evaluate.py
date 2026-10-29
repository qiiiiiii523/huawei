"""B8 evaluation uses B7's complete-120, reliable-input, target/lead macro scorer."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np
from .inference import predict_records
from baselines.B7.data import GROUPS, load_record
from baselines.B7.evaluate import limb_diagnostics, pearson

SCORE_PROTOCOL = "B7-full-record-lead-target-pair-macro-v2"

def aggregate(rows, stage):
    by_lead_target, groups, targets = defaultdict(list), set(), set()
    for row in rows:
        groups.add(row["group"]); targets.add((row["group"], row["target_record_id"]))
        for lead, metrics in row["leads"].items():
            if metrics["r"] is not None: by_lead_target[(row["group"], lead, row["target_record_id"])].append(metrics["r"])
    grouped = defaultdict(lambda: defaultdict(list))
    for (group, lead, _), values in by_lead_target.items(): grouped[group][lead].append(float(np.mean(values)))
    means, lead_means = {}, {}
    for group in groups:
        if set(grouped[group]) != {str(i) for i in range(1, 12)}: raise ValueError("every group needs each QC-eligible missing lead")
        lead_means[group] = {lead: float(np.mean(values)) for lead, values in grouped[group].items()}
        means[group] = float(np.mean(list(lead_means[group].values())))
    if stage in ("huawei10", "huawei40"):
        if set(means) != set(GROUPS): raise ValueError("Huawei selection requires task1, machine, body")
        score = .5 * means["task1"] + .25 * means["machine"] + .25 * means["body"]
    else: score = float(np.mean(list(means.values())))
    return {"r": score, "groups": means, "lead_groups": lead_means, "targets": len(targets), "pairs": len(rows), "score_protocol": SCORE_PROTOCOL}

def evaluate(model, manifest, stage):
    if stage == "huawei10":
        # The 10s view is for training only. Selection stays on each complete 120s target.
        from baselines.B7.huawei120 import full120_validation_manifest
        manifest = full120_validation_manifest(manifest)
        stage = "huawei40"
    rows, pending, seen_targets = [], [], {}
    def flush():
        if not pending: return
        predictions = predict_records(model, [item[1] for item in pending], manifest["scales_uV"]).cpu().numpy()
        for (row, anchor, target, mask), prediction in zip(pending, predictions):
            leads = {}
            for lead in range(1, 12):
                if mask[lead]:
                    r = pearson(prediction[lead], target[lead])
                    leads[str(lead)] = {"r": r if np.isfinite(r) else None, "rmse_uV": float(np.sqrt(np.mean((prediction[lead] - target[lead]) ** 2)))}
            if leads: rows.append({"record_id": row["record_id"], "group": row["group"], "target_record_id": row["target_record_id"], "leads": leads, "limb_diagnostics": limb_diagnostics(anchor, prediction, target, mask)})
        pending.clear()
    for row in manifest["records"]:
        if row["split"] != "validation" or not row["input_reliable"]: continue
        anchor, target, mask, _ = load_record(row)
        if stage.startswith("huawei") and target.shape[-1] != 60000: raise ValueError("B8 Huawei selection must score complete real 120s")
        target_id = row["target_record_id"]; fingerprint = (row["subject_id"], mask.tobytes(), hashlib.sha256(np.where(mask[:, None], target, 0).tobytes()).hexdigest())
        if target_id in seen_targets and seen_targets[target_id] != fingerprint: raise ValueError("inconsistent repeated physical target")
        seen_targets[target_id] = fingerprint
        if pending and pending[0][1].shape[-1] != anchor.shape[-1]: flush()
        pending.append((row, anchor, target, mask))
        if len(pending) >= 4: flush()
    flush()
    return aggregate(rows, stage), rows
