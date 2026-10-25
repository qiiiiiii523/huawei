"""Prepare independent B7 caches from raw records or an explicit QC/split index."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

from . import D12_LEADS
from .config import read_json
from .data import DATA_PROTOCOL, fit_scales, read_manifest
from .public_qc import PUBLIC_QC_VERSION, PUBLIC_SOURCES, inspect_public, deduplicate


def _gain(text, zero):
    match = re.fullmatch(r"([+\-\deE.]+)(?:\(([+\-\deE.]+)\))?(?:/(.+))?", text)
    if not match:
        raise ValueError("Unrecognized WFDB gain")
    gain = float(match[1])
    baseline = float(match[2]) if match[2] is not None else float(zero)
    unit = (match[3] or "mV").lower().replace("μ", "u").replace("µ", "u")
    if not math.isfinite(gain) or gain <= 0 or unit not in ("mv", "uv", "v"):
        raise ValueError("Invalid gain/unit")
    return gain, baseline, {"uv": 1., "mv": 1000., "v": 1e6}[unit]


def read_wfdb(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    head = lines[0].split()
    # Known local Chapman defect: MAT4 still supplies the complete 12x5000 raw data.
    if path.stem == "JS01052" and len(head) > 3 and "/mV" in head[3]:
        with path.with_suffix(".mat").open("rb") as h:
            typ, nr, nc, imag, name_len = struct.unpack("<5i", h.read(20))
            h.read(name_len)
            if (typ, nr, nc, imag) != (30, 12, 5000, 0):
                raise ValueError("Unexpected malformed-header MAT layout")
            raw = np.frombuffer(h.read(), dtype="<i2").reshape(nr, nc, order="F").astype("float32")
        return raw, 500, "Chapman_JS01052_MAT4_verified_layout"
    declared, fs, n = int(head[1]), float(head[2].split("/")[0]), int(head[3])
    signal_lines = lines[1:]
    joined = False
    # Some Chapman headers concatenate the first signal line after the record
    # fields (e.g. S23074 ... 5000 JS23074.mat 16+24 ... I).
    if len(head) > 4 and len(head[4:]) == 9 and head[4].endswith((".mat", ".dat")):
        signal_lines = [" ".join(head[4:]), *signal_lines]
        joined = True
    rows = [line.split() for line in signal_lines[:declared]]
    # WFDB may store different leads in different files (PTB: 12 in .dat, 3 in .xyz).
    standard = {k.lower() for k in D12_LEADS}
    requested = [r for r in rows if r[-1].lower() in standard]
    if (len(requested) != 12 or {r[-1].lower() for r in requested} != standard
            or len({r[0] for r in requested}) != 1):
        raise ValueError(f"{path}: all standard12 leads must be present in one format16 payload; "
                         f"found {[r[-1] for r in rows if r]}")
    filename = requested[0][0]
    payload_rows = [r for r in rows if r[0] == filename]
    fmt = payload_rows[0][1]
    if fmt.split("+")[0] != "16" or any(r[1] != fmt for r in payload_rows):
        raise ValueError("Reader requires an interleaved format16 payload")
    offset = int(fmt.split("+")[1]) if "+" in fmt else 0
    payload_path = path.parent / filename
    size = payload_path.stat().st_size - offset
    actual = len(payload_rows)
    rows = payload_rows
    repair = "WFDB_multifile_standard12_selected" if actual != declared else "none"
    if joined:
        repair = "WFDB_joined_first_signal_line_verified"
    if size != n * actual * 2:
        raise ValueError("WFDB size/channel/sample count mismatch")
    with payload_path.open("rb") as h:
        h.seek(offset)
        raw = np.frombuffer(h.read(), dtype="<i2").reshape(n, actual).T
    expected = [int(r[5]) for r in rows]
    # WFDB headers use both signed and unsigned spellings of the same 16-bit checksum.
    checksums = [int(v) % 65536 for v in raw.sum(1, dtype="int64")]
    if raw[:, 0].tolist() != expected or checksums != [int(r[6]) % 65536 for r in rows]:
        raise ValueError("WFDB initial values/checksums disagree")
    mapped = {}
    for j, row in enumerate(rows):
        gain, baseline, factor = _gain(row[2], row[4])
        mapped[row[-1].lower()] = ((raw[j].astype("float32") - baseline) / gain) * factor
    return np.stack([mapped[k.lower()] for k in D12_LEADS]), fs, repair


def read_xml(path):
    root = ET.parse(path).getroot()
    ns = {"h": root.tag[1:].split("}", 1)[0]}
    candidates = []
    for ss in root.findall(".//h:sequenceSet", ns):
        mapped, fs = {}, None
        for seq in ss.findall("h:component/h:sequence", ns):
            code = seq.find("h:code", ns)
            if code is None:
                continue
            name = code.get("code", "")
            if name == "TIME_ABSOLUTE":
                inc = seq.find(".//h:increment", ns)
                dt = float(inc.get("value")) * (0.001 if inc.get("unit") == "ms" else 1.)
                fs = 1 / dt
            elif name.startswith("MDC_ECG_LEAD_"):
                val = seq.find("h:value", ns)
                digits, scale, origin = val.find("h:digits", ns), val.find("h:scale", ns), val.find("h:origin", ns)
                if digits is None or scale is None:
                    continue
                unit = scale.get("unit", "uV").lower().replace("μ", "u").replace("µ", "u")
                multiplier = {"uv": 1., "mv": 1000., "v": 1e6}[unit]
                a = np.fromstring(digits.text, sep=" ", dtype="float32") * float(scale.get("value"))
                if origin is not None:
                    a += float(origin.get("value", 0))
                mapped[name.replace("MDC_ECG_LEAD_", "").lower()] = a * multiplier
        if fs and all(k.lower() in mapped for k in D12_LEADS):
            arrays = [mapped[k.lower()] for k in D12_LEADS]
            if len({len(a) for a in arrays}) != 1:
                raise ValueError("XML lead lengths differ")
            candidates.append((np.stack(arrays), fs))
    if not candidates:
        raise ValueError("No complete standard12 XML sequence")
    target, fs = max(candidates, key=lambda x: x[0].shape[-1] / x[1])
    return target, fs, "longest_complete_XML_sequence"


def read_raw(path):
    if path.suffix.lower() == ".hea":
        target, fs, repair = read_wfdb(path)
        return target[:1].copy(), target, fs, repair
    if path.suffix.lower() == ".xml":
        target, fs, repair = read_xml(path)
        return target[:1].copy(), target, fs, repair
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as a:
            target = a["target_uV"].astype("float32")
            anchor = a["anchor_uV"].astype("float32") if "anchor_uV" in a else target[:1].copy()
            fs = float(a["sample_rate"].item()) if "sample_rate" in a else 500.
        return anchor, target, fs, "raw_uV_NPZ"
    raise ValueError("Expected .hea, .xml or raw-uV .npz")


def public_index(source, root):
    if source == "public10":
        return public_index("ptbxl", root / "PTB-XL") + public_index("chapman", root / "Chapman_Shaoxing_Ningbo")
    rows = []
    if source == "ptbxl":
        with (root / "ptbxl_database.csv").open(encoding="utf-8-sig", newline="") as h:
            entries = list(csv.DictReader(h))
        records = [(root / (r["filename_hr"] + ".hea"), r) for r in entries]
    else:
        records = [(p, {}) for p in sorted(root.rglob("*.hea"))]
    if source == "ptbdb":
        canonical = []
        for path, info in records:
            relative = path.relative_to(root)
            if len(relative.parts) == 1:
                print(f"[index] exclude root-level PTB header: {relative}", flush=True)
                continue
            if len(relative.parts) != 2 or not re.fullmatch(r"patient\d+", relative.parts[0]):
                raise ValueError(f"Unexpected PTB header location; review required: {path}")
            canonical.append((path, info))
        records = canonical
    for path, info in records:
        patient = str(info.get("patient_id", path.parent.name if source == "ptbdb" else path.stem))
        if source == "ptbxl":
            fold = int(info["strat_fold"])
            split = "test" if fold == 10 else "validation" if fold == 9 else "train"
        else:
            num = int(hashlib.sha256(f"B7-split|{source}|{patient}".encode()).hexdigest()[:8], 16) % 10
            split = "validation" if num == 8 else "test" if num == 9 else "train"
        age, sex, height, weight = info.get("age"), None, info.get("height"), info.get("weight")
        if source == "ptbxl":
            sex = {"0": 0, "1": 1}.get(info.get("sex"))  # PTB-XL and B7: 0=male, 1=female.
        else:
            comments = path.read_text(encoding="utf-8-sig").splitlines()
            for line in comments:
                comment = line.lstrip("#").strip()
                if line.startswith("#") and comment.lower().startswith("age:"):
                    age = comment.split(":", 1)[1].strip()
                if line.startswith("#") and comment.lower().startswith("sex:"):
                    sex = {"male": 0, "female": 1}.get(comment.split(":", 1)[1].strip().lower())
        numeric, mask = [0., 0., 0.], [False] * 4
        topcoded = 0.
        for j, raw, divisor, maximum in ((0, age, 100., 120.), (1, height, 200., 260.), (2, weight, 150., 500.)):
            try:
                val = float(raw)
                if j == 0 and source == "ptbxl" and 200 <= val <= 400:
                    val = 90.
                if math.isfinite(val) and 0 < val <= maximum:
                    if j == 0:
                        topcoded = float(val >= 90)
                        val = min(val, 90)
                    numeric[j] = val / divisor
                    mask[(0, 2, 3)[j]] = True
            except (ValueError, TypeError):
                pass
        mask[1] = sex is not None
        identity = f"{patient}/{path.stem}" if source == "ptbdb" else path.stem
        rows.append({"record_id": f"{source}:{identity}", "subject_id": f"{source}:{patient}",
                     "metadata_encoding": "male0-female1-v2",
                     "target_record_id": f"{source}:{identity}", "split": split, "source": source,
                     "group": source, "path": str(path.resolve()), "input_reliable": True,
                     "metadata": {"numeric": numeric, "sex": sex if sex is not None else 2,
                                  "field_mask": mask, "age_topcoded": [topcoded]}})
    return rows


def prepare(args):
    output = Path(args.output).resolve()
    resume = getattr(args, "resume", False)
    if output.exists() and not resume:
        raise FileExistsError("Use a new cache directory; raw data are never modified")
    if resume and (output / "manifest.json").exists():
        raise FileExistsError("Cache is complete; reuse its manifest instead of resuming preparation")
    if args.index:
        rows = read_json(args.index)
        if not isinstance(rows, list):
            raise ValueError("Index must be a list of record metadata")
        base = Path(args.index).resolve().parent
    else:
        rows = public_index(args.public_source, Path(args.root).resolve())
        base = Path(args.root).resolve()
    if not rows:
        raise ValueError("No input records")
    scale_data = read_manifest(args.scale_manifest) if args.scale_manifest else None
    if scale_data is not None and scale_data.get("public_qc_version") != PUBLIC_QC_VERSION:
        raise ValueError("QC-v2 preparation must reuse QC-v2 train-fitted scales; rebuild public10 first")
    # Validate metadata and patient isolation before creating any output.
    splits, ids, targets = {}, set(), {}
    for row in rows:
        required = ("record_id", "target_record_id", "subject_id", "source", "group", "path", "split", "input_reliable")
        if any(k not in row for k in required) or row["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid index row")
        if row["source"] == "ptbxl" and row.get("metadata_encoding") != "male0-female1-v2":
            raise ValueError("PTB-XL index needs corrected metadata_encoding; rebuild via --public-source public10")
        if row["record_id"] in ids or not isinstance(row["input_reliable"], bool):
            raise ValueError("Duplicate ID or nonboolean reliability")
        ids.add(row["record_id"])
        subject = row["subject_id"]
        if subject in splits and splits[subject] != row["split"]:
            raise ValueError("Patient leakage")
        splits[subject] = row["split"]
        target = row["target_record_id"]
        owner = (subject, row["split"])
        if target in targets and targets[target] != owner:
            raise ValueError("Physical target leakage or inconsistent patient identity")
        targets[target] = owner
        if row["group"] in ("task1", "machine", "body") and "lead_mask" not in row:
            raise ValueError("Huawei index requires the established explicit12 QC mask")
        if "lead_mask" in row:
            qc = np.asarray(row["lead_mask"])
            if qc.shape != (12,) or not np.isin(qc, [0, 1]).all():
                raise ValueError("lead_mask must contain exactly12 boolean/0-1 values")
    output.mkdir(parents=True, exist_ok=resume)
    policy = {"public_qc_version": PUBLIC_QC_VERSION, "index_sha256": hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "scale_manifest_sha256": hashlib.sha256(Path(args.scale_manifest).read_bytes()).hexdigest()
        if args.scale_manifest else None}
    ledger = output / "prepare_identity.json"
    if resume:
        if not ledger.is_file() or read_json(ledger) != policy:
            raise ValueError("Preparation identity/QC changed; use a new cache directory")
    else:
        ledger.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (output / "records").mkdir(exist_ok=resume)
    expected_names = {f"{i:07d}.npz" for i in range(len(rows))}
    # A killed np.savez may leave its own temporary file; it is rewritten below.
    expected_names.update(f"{i:07d}.tmp" for i in range(len(rows)))
    if any(p.name not in expected_names for p in (output / "records").iterdir()):
        raise ValueError("Partial cache contains unexpected files; inspect before resuming")
    prepared = []
    qc_audit = []
    reused = 0
    for i, original in enumerate(rows):
        row = dict(original)
        raw_path = Path(row["path"])
        raw_path = (base / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
        if i % 250 == 0:
            print(f"[prepare] {i}/{len(rows)} verified_reused={reused} next={raw_path}", flush=True)
        try:
            anchor, target, fs, repair = read_raw(raw_path)
        except Exception as exc:
            raise ValueError(f"Failed record {i}, {row['record_id']}, {raw_path}: {exc}") from exc
        if not math.isfinite(fs) or fs <= 0:
            raise ValueError("Sampling rate must be finite and positive")
        if row["source"] in PUBLIC_SOURCES and fs != 500:
            raise ValueError("Public10 sources must use their original 500Hz records")
        if target.ndim != 2 or target.shape[0] != 12 or anchor.shape != (1, target.shape[-1]):
            raise ValueError("Invalid raw signal shapes")
        mask = np.asarray(row.pop("lead_mask", np.isfinite(target).all(1)), dtype=bool)
        if mask.shape != (12,) or not np.isfinite(target[mask]).all() or not np.isfinite(anchor).all():
            raise ValueError("Invalid raw target/anchor/QC")
        if fs != 500:
            from fractions import Fraction
            from scipy.signal import resample_poly
            ratio = Fraction(500 / fs).limit_denominator(10000)
            anchor = resample_poly(anchor, ratio.numerator, ratio.denominator, axis=-1).astype("float32")
            # Invalid leads are never supervised, prevent their NaNs entering filtering.
            target = resample_poly(np.where(mask[:, None], target, 0.), ratio.numerator, ratio.denominator, axis=-1).astype("float32")
        if row["group"] in ("task1", "machine", "body"):
            if target.shape[-1] < 60000:
                raise ValueError("Huawei record needs at least120 real seconds")
            anchor, target = anchor[:, :60000], target[:, :60000]
        if target.shape[-1] < 5000:
            raise ValueError("Less than10 real seconds")
        if row["source"] in PUBLIC_SOURCES:
            mask, anchor_ok, quality = inspect_public(anchor, target, mask, 500)
            row["public_qc"] = quality
            row["input_reliable"] = row["input_reliable"] and anchor_ok
            qc_audit.append({"record_id": row["record_id"], "split": row["split"],
                             "input_reliable": row["input_reliable"], "lead_mask": mask.tolist(), **quality})
        metadata = row.pop("metadata", {})
        destination = output / "records" / f"{i:07d}.npz"
        arrays = {"anchor_uV": anchor, "target_uV": target, "lead_mask": mask,
                  **{k: np.asarray(v) for k, v in metadata.items()}}
        if destination.exists():
            # Old caches lack a provenance ledger: independently decode each raw
            # record and compare all stored arrays before trusting the prefix.
            with np.load(destination, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(
                    saved[k].shape != v.shape or not np.array_equal(saved[k], v, equal_nan=True)
                    for k, v in arrays.items()
                ):
                    raise ValueError(f"Existing cache differs from raw record {i}: {destination}")
            reused += 1
        else:
            with destination.with_suffix(".tmp").open("wb") as handle:
                np.savez(handle, **arrays)
            destination.with_suffix(".tmp").replace(destination)
        row.update(path=str(destination.relative_to(output)), resolved_path=str(destination),
                   raw_path=str(raw_path), reader=repair, real_samples=target.shape[-1])
        prepared.append(row)
    prepared, duplicates = deduplicate(prepared)
    quality_by_id = {q["record_id"]: q for q in qc_audit}
    # Exclude unusable inputs entirely, including evaluation and scale fitting.
    excluded = [r["record_id"] for r in prepared if r["source"] in PUBLIC_SOURCES
                and (not r["input_reliable"] or not any(
                    quality_by_id[r["record_id"]]["lead_mask"][j]
                    for j in (1, 6, 7, 8, 9, 10, 11)))]
    excluded_set = set(excluded)
    prepared = [r for r in prepared if r["record_id"] not in excluded_set]
    audit = {"version": PUBLIC_QC_VERSION, "input_records": len(rows), "retained_records": len(prepared),
             "excluded_unusable": excluded, "duplicate_groups": duplicates, "records": qc_audit,
             "policy": "heldout-first exact dedup; >=1s exact flat and >50mV jumps masked; other flags review only"}
    (output / "public_qc_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    scales = scale_data["scales_uV"] if scale_data else fit_scales(prepared)
    for row in prepared:
        del row["resolved_path"]
    manifest = {"protocol": DATA_PROTOCOL, "sample_rate": 500, "scales_uV": scales,
                "public_qc_version": PUBLIC_QC_VERSION,
                "scale_source": str(Path(args.scale_manifest).resolve()) if args.scale_manifest else "fit_train_only_this_manifest",
                "B5_weights_used": False, "records": prepared}
    path = output / "manifest.json"
    pending = output / "manifest.pending.json"
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    read_manifest(pending)
    pending.replace(path)
    print(json.dumps({"manifest": str(path), "records": len(prepared), "verified_reused": reused,
                      "optimizer_updates": 0}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--index", help="JSON records with source/subject/split/QC and raw waveform path")
    choice.add_argument("--public-source", choices=("public10", "ptbxl", "chapman", "ptbdb"))
    p.add_argument("--root", help="Public dataset root when auto-indexing")
    p.add_argument("--scale-manifest", help="Reuse train-fitted B7 scales for a subsequent stage")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true", help="Verify/reuse a partial cache against raw waveforms, then continue")
    a = p.parse_args()
    if a.public_source and not a.root:
        p.error("--public-source requires --root")
    prepare(a)


if __name__ == "__main__":
    main()
