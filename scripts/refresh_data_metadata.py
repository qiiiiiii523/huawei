"""Refresh the three changed signal sources and demographics; freeze existing patient splits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.contracts import D12_LEADS, D6_LEADS
from ecg12gen.csv_io import read_csv_rows
from ecg12gen.raw_task1 import parse_d12_xml
from ecg12gen.raw_task2 import parse_body_scale_record


def subject_id(value):
    match = re.fullmatch(r"HNU[\s_-]*(\d+)", value.strip(), re.I)
    return "HNU" + match.group(1) if match else ""


def stable_id(kind, filename, key=""):
    return kind + "_" + hashlib.sha1((filename + "|" + key).encode()).hexdigest()[:14]


def write_rows(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def header(path):
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip() == "Index":
                break
            if row and ":" in row[0]:
                k, v = row[0].split(":", 1)
                result[k.strip()] = v.strip()
    return result


def refresh(data_root: Path, repository_root: Path):
    metadata = repository_root / "metadata"
    old, _ = read_csv_rows(metadata / "raw_record_manifest.csv")
    old_splits, _ = read_csv_rows(metadata / "subject_split.csv")
    split_by_subject = {r["subject_id"]: r["split"] for r in old_splits}
    if len(split_by_subject) != len(old_splits):
        raise ValueError("Existing subject split contains duplicates")
    old_by_file = {(r["device_type"], r["source_file"]): r for r in old}
    fields = list(old[0])
    rows = [dict(r) for r in old if r["device_type"] == "watch"]  # unchanged source metadata
    fingerprints = []
    device_counts = Counter()
    for folder, device, prefix, leads in (
        ("心电图机d12", "ecg_machine_d12", "d12", D12_LEADS),
        ("心电图机d6", "ecg_machine_d6", "d6", D6_LEADS),
        ("体脂秤d6", "body_scale_d6", "scale", D6_LEADS)):
        suffix = ".csv" if prefix == "scale" else ".xml"
        paths = sorted(p for p in (data_root / "Data" / folder).rglob("*") if p.suffix.lower() == suffix)
        if len({p.name for p in paths}) != len(paths):
            raise ValueError(f"Duplicate filenames within {folder}; audit before pairing")
        for ordinal, path in enumerate(paths, 1):
            relative = path.relative_to(data_root).as_posix()
            row = {key: "" for key in fields}
            previous = old_by_file.get((device, path.name))
            row.update({"record_id": previous["record_id"] if previous else stable_id(prefix, path.name),
                        "source_file": path.name, "source_path": relative, "device_type": device,
                        "signal_type": "ecg_d6_from_scale_export" if prefix == "scale" else "ecg_" + prefix,
                        "parse_status": "ok", "quality_status": "usable", "qc_status": "usable",
                        "qc_checks": "parse,identity,leads,unit,fs,duration,finite,constant"})
            try:
                if prefix == "scale":
                    information = header(path)
                    row["externalid"] = information.get("被试ID", "")
                    row["subject_id"] = subject_id(row["externalid"])
                    signal, valid = parse_body_scale_record(path)
                    fs, unit = 500., information.get("数据单位", "")
                    stamp = path.stem
                    row["record_time"] = datetime.strptime(stamp, "%Y-%m-%d-%H-%M-%S").isoformat(sep=" ")
                else:
                    match = re.search(r"HNU\d+", path.name, re.I)
                    row["subject_id"] = subject_id(match.group(0)) if match else ""
                    row["externalid"] = row["subject_id"]
                    record = parse_d12_xml(path, leads)
                    signal, fs, unit = record.signal_uV, record.sampling_rate_hz, "uV"
                    valid = np.isfinite(signal)
                    with path.open(encoding="utf-8-sig") as handle:
                        beginning = handle.read(8192)
                    match = re.search(r'<low\s+value="([^"]+)"', beginning)
                    stamp = match.group(1) if match else ""
                    row["record_time"] = datetime.strptime(stamp[:14], "%Y%m%d%H%M%S").isoformat(sep=" ") if stamp else ""
                row.update({"native_fs": str(fs), "unit": unit, "sample_count": str(signal.shape[1]),
                            "duration_sec": f"{signal.shape[1] / fs:.9g}", "record_time_raw": stamp,
                            "lead_or_channel_info": ",".join(leads)})
                reasons = []
                if not row["subject_id"]:
                    reasons.append("unreliable_or_missing_subject_id")
                if signal.shape[1] / fs <= 90:
                    reasons.append("record_duration_at_most_90_seconds")
                if not valid.any() or not any(np.ptp(signal[c, valid[c]]) > 0 for c in range(len(leads)) if valid[c].any()):
                    reasons.append("empty_or_constant_signal")
                if prefix != "scale" and not valid.all():
                    reasons.append("nonfinite_machine_signal")
                if reasons:
                    row["quality_status"] = row["qc_status"] = "review"
                    row["quality_reason"] = row["qc_reason"] = ";".join(reasons)
            except Exception as error:
                row.update({"parse_status": "error", "quality_status": "reject", "qc_status": "reject",
                            "quality_reason": str(error), "qc_reason": str(error)})
            rows.append(row)
            device_counts[device] += 1
            fingerprints.append({"source_path": relative, "bytes": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if ordinal % 25 == 0 or ordinal == len(paths):
                print(f"Scanned {folder}: {ordinal}/{len(paths)}", flush=True)
    demographics, encoding = read_csv_rows(data_root / "Data/userinfobean.csv")
    for i, info in enumerate(demographics):
        row = {key: "" for key in fields}
        sid = subject_id(info.get("externalid", ""))
        row.update({"record_id": stable_id("userinfo", "userinfobean.csv", info.get("uniqueid", "") + "|" + str(i)),
                    "subject_id": sid, "externalid": info.get("externalid", ""), "device_type": "userinfobean",
                    "signal_type": "user_info", "source_file": "userinfobean.csv", "source_path": "Data/userinfobean.csv",
                    "parse_status": "ok", "quality_status": "usable" if sid else "review",
                    "quality_reason": "" if sid else "unreliable_subject_id", "uniqueid": info.get("uniqueid", ""),
                    "healthid": info.get("healthid", ""), "record_time_raw": info.get("recordtime", "")})
        rows.append(row)
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Refreshed raw manifest has duplicate record IDs")
    targets = defaultdict(list)
    for row in rows:
        if row["device_type"] == "ecg_machine_d12" and row["subject_id"]:
            targets[row["subject_id"]].append(row)
    old_pairs = {}
    for task in ("task1", "task2"):
        previous, _ = read_csv_rows(metadata / f"pair_manifest_{task}.csv")
        old_pairs[task] = {r["input_record_id"]: r for r in previous}
    tables = {"task1": [], "task2": []}
    reviews = []
    pair_fields = list(dict.fromkeys(list(next(iter(old_pairs["task1"].values()))) +
                                    list(next(iter(old_pairs["task2"].values()))) +
                                    ["target_reuse_count", "pair_relation", "training_policy", "split"]))
    for row in rows:
        if row["device_type"] == "watch" and row["signal_type"] == "ecg":
            task, source = "task1", "watch_ecg"
        elif row["device_type"] in {"body_scale_d6", "ecg_machine_d6"}:
            task, source = "task2", row["device_type"]
        else:
            continue
        candidates = targets.get(row["subject_id"], [])
        reliable = bool(subject_id(row["subject_id"]))
        status = "paired" if reliable and len(candidates) == 1 else "review" if not reliable or len(candidates) > 1 else "unmatched"
        target = candidates[0] if status == "paired" else {}
        previous = old_pairs[task].get(row["record_id"], {})
        item = {key: "" for key in pair_fields}
        item.update({"pair_id": previous.get("pair_id") or task.upper() + "_" + hashlib.sha1(row["record_id"].encode()).hexdigest()[:14],
            "subject_id": row["subject_id"], "input_type": source, "input_record_id": row["record_id"],
            "target_record_id": target.get("record_id", ""), "input_path": row["source_path"], "target_path": target.get("source_path", ""),
            "pair_rule": "exact subject_id; unique d12 target" if status == "paired" else "no guessed mapping",
            "pair_confidence": "high" if status == "paired" else "none", "pair_status": status,
            "input_groupid": row.get("groupid", ""), "input_externalid": row["externalid"],
            "input_record_time": row["record_time"], "target_record_time": target.get("record_time", ""),
            "input_quality_status": row["quality_status"], "target_quality_status": target.get("quality_status", ""),
            "candidate_target_count": str(len(candidates)), "pair_relation": "same_subject_cross_time",
            "training_policy": "keep" if status == "paired" else "review"})
        item["input_lead_order"] = ",".join(D6_LEADS) if task == "task2" else "I"
        item["channel_mapping"] = "1->I,2->II,9->III,10->aVR,11->aVL,12->aVF" if source == "body_scale_d6" else ""
        tables[task].append(item)
        if status != "paired":
            reviews.append({"review_id": item["pair_id"], "task": task, "input_type": source,
                "input_record_id": row["record_id"], "subject_id": row["subject_id"], "raw_externalid": row["externalid"],
                "groupid": row.get("groupid", ""), "candidate_target_ids": ";".join(r["record_id"] for r in candidates),
                "candidate_subject_ids": ";".join(r["subject_id"] for r in candidates), "issue_type": status,
                "reason": "exact identity missing/ambiguous; no inference", "recommended_action": "manual identity confirmation"})
    new_subjects = sorted({r["subject_id"] for r in rows if subject_id(r["subject_id"])} - split_by_subject.keys())
    random.Random(42).shuffle(new_subjects)
    validation_new = set(new_subjects[:round(len(new_subjects) * .2)])
    split_rows = [dict(r) for r in old_splits]
    for order, sid in enumerate(new_subjects, max(int(r["assignment_order"]) for r in old_splits) + 1):
        split = "validation" if sid in validation_new else "train"
        split_by_subject[sid] = split
        split_rows.append({"subject_id": sid, "split": split, "random_seed": "42", "assignment_order": str(order)})
    for table in tables.values():
        reuse = Counter(r["target_record_id"] for r in table if r["pair_status"] == "paired")
        for row in table:
            row["split"] = split_by_subject.get(row["subject_id"], "")
            row["target_reuse_count"] = str(reuse[row["target_record_id"]]) if row["target_record_id"] else "0"
    for row in rows:
        row["split"] = split_by_subject.get(row["subject_id"], "")
    for row in split_rows:
        sid = row["subject_id"]
        row["record_count"] = str(sum(r["subject_id"] == sid for r in rows))
        for task, source, label in (("task1", "watch_ecg", "task1"), ("task2", "ecg_machine_d6", "task2_machine"), ("task2", "body_scale_d6", "task2_body_scale")):
            paired = [r for r in tables[task] if r["subject_id"] == sid and r["input_type"] == source and r["pair_status"] == "paired"]
            row[label + "_pair_count"] = str(len(paired))
            row["has_" + label + "_usable_pair"] = str(int(any(r["target_quality_status"] == "usable" and r["input_quality_status"] == "usable" for r in paired)))
        row["assignment_origin"] = "preserved_existing" if sid in {r["subject_id"] for r in old_splits} else "new_subject_seed42"
    if any(split_by_subject[r["subject_id"]] != r["split"] for r in old_splits):
        raise AssertionError("An existing patient changed split")
    backup = repository_root / "reports/data_backups" / datetime.now().strftime("refresh_%Y%m%d_%H%M%S")
    backup.mkdir(parents=True, exist_ok=False)
    for path in metadata.glob("*.csv"):
        shutil.copy2(path, backup / path.name)
    write_rows(metadata / "raw_record_manifest.csv", rows, fields)
    write_rows(metadata / "subject_split.csv", split_rows, list(dict.fromkeys(list(old_splits[0]) + ["assignment_origin"])))
    for task, table in tables.items():
        write_rows(metadata / f"pair_manifest_{task}.csv", table, pair_fields)
    review_fields = list(reviews[0]) if reviews else ["review_id", "reason"]
    write_rows(metadata / "id_mapping_review.csv", reviews, review_fields)
    write_rows(metadata / "userinfobean_utf8.csv", demographics, list(demographics[0]))
    audit = {"raw_files_mutated": False, "existing_subject_splits_unchanged": True,
        "backup_directory": str(backup), "old_subject_count": len(old_splits), "new_subject_count": len(new_subjects),
        "subject_split_counts": dict(Counter(r["split"] for r in split_rows)), "signal_file_counts": dict(device_counts),
        "quality_counts": dict(Counter(r["device_type"] + ":" + r["quality_status"] for r in rows)),
        "pair_counts": {t: dict(Counter(r["pair_status"] for r in v)) for t, v in tables.items()},
        "demographics_rows": len(demographics), "demographics_encoding": encoding,
        "demographics_unreliable_id_rows": sum(not subject_id(r.get("externalid", "")) for r in demographics),
        "source_fingerprints": fingerprints}
    (metadata / "data_refresh_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "source_fingerprints"}, ensure_ascii=False, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    refresh(args.data_root.resolve(), ROOT)
