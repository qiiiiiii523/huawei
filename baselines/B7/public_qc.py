"""Versioned public ECG quality policy; raw files are never changed."""
from __future__ import annotations

import hashlib
import numpy as np

PUBLIC_QC_VERSION = "B7-public-QC-v2"
PUBLIC_SOURCES = {"ptbxl", "chapman"}


def flat_samples(x):
    """Longest exactly identical run, in samples (including both endpoints)."""
    boundaries = np.r_[0, np.flatnonzero(np.diff(x) != 0) + 1, len(x)]
    return int(np.diff(boundaries).max())


def inspect_public(anchor, target, mask, fs):
    if fs != 500 or target.shape != (12, 5000):
        raise ValueError("Public10 QC requires real 12x5000 samples at 500Hz")
    finite = np.isfinite(target).all(1)
    flat = [flat_samples(x) if ok else 0 for x, ok in zip(target, finite)]
    jumps = np.array([float(np.abs(np.diff(x.astype('float64'))).max())
                      if ok else float('inf') for x, ok in zip(target, finite)])
    # Exact long holds are acquisition defects, not a low-amplitude ECG threshold.
    # A 50mV one-sample discontinuity is rejected; lower jumps remain review flags.
    valid = np.asarray(mask, dtype=bool) & finite & (np.array(flat) < 500) & (jumps <= 50000)
    anchor_ok = bool(np.isfinite(anchor).all() and flat_samples(anchor[0]) < 500
                     and np.abs(np.diff(anchor[0].astype('float64'))).max() <= 50000)
    flags = {}
    for name, selector in (
        ("exact_flat_ge_1s", np.array(flat) >= 500),
        ("jump_gt_5mV_review", jumps > 5000),
        ("jump_gt_50mV_rejected", jumps > 50000),
        ("abs_gt_10mV_review", np.max(np.abs(target), axis=1) > 10000),
        ("nonfinite", ~finite),
    ):
        indices = np.flatnonzero(selector).tolist()
        if indices:
            flags[name] = indices
    reconstructed = (target[1] - target[0], -(target[0] + target[1]) / 2,
                     target[0] - target[1] / 2, target[1] - target[0] / 2)
    residual = [float(np.sqrt(np.mean((x.astype('float64') - target[j]) ** 2)))
                for j, x in zip(range(2, 6), reconstructed)]
    inconsistent = [j for j, value in zip(range(2, 6), residual) if value > 10]
    if inconsistent:
        flags['limb_formula_rmse_gt_10uV'] = inconsistent
        valid[inconsistent] = False
    # Reconstructed limb leads may inherit defects in either input lead.
    if not valid[0] or not valid[1]:
        valid[2:6] = False
    fingerprint = hashlib.sha256(np.asarray(target, dtype='<f4').tobytes()).hexdigest()
    return valid, anchor_ok, {"flags": flags, "max_flat_samples": flat,
                              "limb_formula_rmse_uV": residual,
                              "waveform_sha256": fingerprint}


def deduplicate(records):
    """Keep held-out copies before training copies; do not move patients/splits."""
    groups = {}
    for row in records:
        if row["source"] in PUBLIC_SOURCES:
            groups.setdefault(row["public_qc"]["waveform_sha256"], []).append(row)
    removed, audit = set(), []
    priority = {"test": 0, "validation": 1, "train": 2}
    for digest, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda r: (priority[r["split"]], r["record_id"]))
        removed.update(r["record_id"] for r in ordered[1:])
        audit.append({"waveform_sha256": digest,
                      "retained": ordered[0]["record_id"],
                      "members": [{"record_id": r["record_id"], "split": r["split"]} for r in ordered]})
    return [r for r in records if r["record_id"] not in removed], audit
