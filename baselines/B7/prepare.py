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
    rows = [line.split() for line in lines[1:declared + 1]]
    # WFDB may store different leads in different files (PTB: 12 in .dat, 3 in .xyz).
    standard = {k.lower() for k in D12_LEADS}
    requested = [r for r in rows if r[-1].lower() in standard]
    if len(requested) != 12 or len({r[0] for r in requested}) != 1:
        raise ValueError("All standard12 leads must be present in one format16 payload")
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
            sex = {"0": 1, "1": 0}.get(info.get("sex"))  # PTB-XL 0=female, 1=male.
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
        rows.append({"record_id": f"{source}:{path.stem}", "subject_id": f"{source}:{patient}",
                     "target_record_id": f"{source}:{path.stem}", "split": split, "source": source,
                     "group": source, "path": str(path.resolve()), "input_reliable": True,
                     "metadata": {"numeric": numeric, "sex": sex if sex is not None else 2,
                                  "field_mask": mask, "age_topcoded": [topcoded]}})
    return rows


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new cache directory; raw data are never modified")
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
    # Validate metadata and patient isolation before creating any output.
    splits, ids, targets = {}, set(), {}
    for row in rows:
        required = ("record_id", "target_record_id", "subject_id", "source", "group", "path", "split", "input_reliable")
        if any(k not in row for k in required) or row["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid index row")
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
    output.mkdir(parents=True)
    (output / "records").mkdir()
    prepared = []
    for i, original in enumerate(rows):
        row = dict(original)
        raw_path = Path(row["path"])
        raw_path = (base / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
        anchor, target, fs, repair = read_raw(raw_path)
        if not math.isfinite(fs) or fs <= 0:
            raise ValueError("Sampling rate must be finite and positive")
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
        metadata = row.pop("metadata", {})
        destination = output / "records" / f"{i:07d}.npz"
        np.savez(destination, anchor_uV=anchor, target_uV=target, lead_mask=mask, **{
            k: np.asarray(v) for k, v in metadata.items()})
        row.update(path=str(destination.relative_to(output)), resolved_path=str(destination),
                   raw_path=str(raw_path), reader=repair, real_samples=target.shape[-1])
        prepared.append(row)
    scales = read_manifest(args.scale_manifest)["scales_uV"] if args.scale_manifest else fit_scales(prepared)
    for row in prepared:
        del row["resolved_path"]
    manifest = {"protocol": DATA_PROTOCOL, "sample_rate": 500, "scales_uV": scales,
                "scale_source": str(Path(args.scale_manifest).resolve()) if args.scale_manifest else "fit_train_only_this_manifest",
                "B5_weights_used": False, "records": prepared}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    read_manifest(path)
    print(json.dumps({"manifest": str(path), "records": len(prepared), "optimizer_updates": 0}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--index", help="JSON records with source/subject/split/QC and raw waveform path")
    choice.add_argument("--public-source", choices=("public10", "ptbxl", "chapman", "ptbdb"))
    p.add_argument("--root", help="Public dataset root when auto-indexing")
    p.add_argument("--scale-manifest", help="Reuse train-fitted B7 scales for a subsequent stage")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.public_source and not a.root:
        p.error("--public-source requires --root")
    prepare(a)


if __name__ == "__main__":
    main()
