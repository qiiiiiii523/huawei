#!/usr/bin/env python3
"""Read-only PTB Diagnostic integrity audit. Python standard library only."""
import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
DATA_EXTENSIONS = {".hea", ".dat", ".xyz"}


def safe_path(root, name):
    name = name.replace("\\", "/")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or ":" in name:
        raise ValueError("Unsafe relative dataset path: " + name)
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset path/symlink escapes its root: " + name)
    return path, relative.as_posix()


def digest(path):
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed during hashing; wait until uploading finishes")
    return value.hexdigest(), after.st_size


def checksum_entries(root):
    source = root / "SHA256SUMS.txt"
    entries = {}
    for number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        if match is None:
            raise ValueError(f"Malformed SHA256SUMS.txt line {number}")
        _, name = safe_path(root, match[2].strip())
        if name in entries:
            raise ValueError("Duplicate checksum entry: " + name)
        entries[name] = match[1].lower()
    if not entries:
        raise ValueError("Empty SHA256SUMS.txt")
    return entries


def record_entries(root):
    entries = []
    for line in (root / "RECORDS").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            _, name = safe_path(root, line.strip())
            if not re.fullmatch(r"patient\d+/[^/]+", name) or Path(name).suffix:
                raise ValueError("Unexpected RECORDS path: " + name)
            entries.append(name)
    if not entries or len(set(entries)) != len(entries):
        raise ValueError("Empty/duplicate RECORDS entries")
    return entries


def audit_header(root, name, checksums):
    row = {"record": name, "patient": name.split("/")[0], "status": "error", "fs_hz": None,
           "samples": None, "seconds": None, "signals": None, "standard_leads": "",
           "duration_ge50s": False, "reason": ""}
    try:
        header, _ = safe_path(root, name + ".hea")
        # Technical header fields are ASCII. Comments may use legacy encodings.
        lines = [line.strip() for line in header.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        first = lines[0].split()
        signals, fs, samples = int(first[1]), float(first[2].split("/")[0]), int(first[3])
        if first[0] != PurePosixPath(name).name or signals < 12 or not math.isfinite(fs) or fs <= 0 or samples <= 0:
            raise ValueError("Invalid record name/signal count/sampling rate/sample count")
        if len(lines) != signals + 1:
            raise ValueError("Signal-line count differs from header declaration")
        row.update(signals=signals, fs_hz=fs, samples=samples, seconds=samples / fs,
                   duration_ge50s=samples / fs >= 50)
        lead_names, files = [], defaultdict(list)
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 9:
                raise ValueError("Incomplete signal line")
            lead_names.append(fields[-1].lower())
            reference = str(PurePosixPath(name).parent / fields[0])
            path, reference = safe_path(root, reference)
            if not path.is_file():
                raise ValueError("Header references missing file: " + reference)
            if reference not in checksums:
                raise ValueError("Header references file absent from checksum manifest: " + reference)
            files[reference].append(fields[1])
        row["standard_leads"] = ",".join(x for x in LEADS if x in lead_names)
        if any(lead_names.count(x) != 1 for x in LEADS):
            raise ValueError("Missing/duplicated standard lead name")
        for filename, formats in files.items():
            # PTB Diagnostic uses multiplexed 16-bit files. Check structural size too.
            if any(fmt != "16" for fmt in formats):
                raise ValueError("Unexpected PTB signal storage format; inspect before preprocessing")
            path, _ = safe_path(root, filename)
            expected = samples * len(formats) * 2
            if path.stat().st_size != expected:
                raise ValueError(f"Binary size mismatch: {filename}, expected {expected}, got {path.stat().st_size}")
        if fs != 1000:
            raise ValueError("Unexpected PTB sampling rate; current adapter expects 1000Hz")
        row["status"] = "ok"
    except (OSError, ValueError, IndexError) as exc:
        row["reason"] = str(exc)
    return row


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(root, output, strict_root=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Put reports OUTSIDE the original dataset directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new/empty report directory; do not put tee logs inside it")
    if not root.is_dir():
        raise FileNotFoundError("Dataset root does not exist: " + str(root))
    output.mkdir(parents=True, exist_ok=True)
    errors, warnings, file_rows, records = [], [], [], []
    for filename in ("SHA256SUMS.txt", "RECORDS"):
        if not (root / filename).is_file():
            errors.append("Missing required root manifest: " + filename)
    checksums, expected_records = {}, []
    if not errors:
        try:
            checksums = checksum_entries(root)
            expected_records = record_entries(root)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    for number, (name, expected) in enumerate(sorted(checksums.items()), 1):
        path, _ = safe_path(root, name)
        patient_data = bool(re.match(r"patient\d+/", name))
        required = patient_data or name == "RECORDS" or strict_root
        row = {"path": name, "required": required, "status": "", "size_bytes": None,
               "expected_sha256": expected, "actual_sha256": "", "message": ""}
        if not path.is_file():
            row["status"] = "missing" if required else "optional_missing"
        else:
            try:
                actual, size = digest(path)
                row.update(actual_sha256=actual, size_bytes=size, status="ok" if actual == expected else "mismatch")
            except (OSError, ValueError) as exc:
                row.update(status="read_error", message=str(exc))
        file_rows.append(row)
        if number % 100 == 0 or number == len(checksums):
            print(f"SHA256 {number}/{len(checksums)}", flush=True)
    bad_files = [r for r in file_rows if r["status"] in ("missing", "mismatch", "read_error")]
    if bad_files:
        errors.append(f"{len(bad_files)} missing/corrupt/unreadable files; see files.csv")
    optional_missing = [r["path"] for r in file_rows if r["status"] == "optional_missing"]
    if optional_missing:
        warnings.append("Optional root metadata omitted: " + ", ".join(optional_missing))
    data_headers = {str(PurePosixPath(p).with_suffix("")) for p in checksums if p.endswith(".hea")}
    if checksums and data_headers != set(expected_records):
        errors.append("RECORDS and checksum manifest disagree on the header set")
    for name in expected_records:
        records.append(audit_header(root, name, checksums))
    header_errors = [r for r in records if r["status"] != "ok"]
    if header_errors:
        errors.append(f"{len(header_errors)} invalid/incomplete record headers or referenced binaries; see records.csv")
    local_data = {p.relative_to(root).as_posix() for folder in root.glob("patient*") if folder.is_dir()
                  for p in folder.iterdir() if p.is_file() and p.suffix.lower() in DATA_EXTENSIONS}
    unexpected = sorted(local_data - set(checksums))
    if unexpected:
        errors.append("Patient waveform files absent from checksum manifest: " + ", ".join(unexpected[:10]))
    by_duration = Counter()
    for row in records:
        if row["status"] == "ok":
            seconds = row["seconds"]
            by_duration["under_50s" if seconds < 50 else "50_to_under_120s" if seconds < 120 else "at_least_120s"] += 1
    summary = {"ready_for_preprocessing": not errors, "read_only_dataset": True, "training_started": False,
        "root": str(root), "checksum_reference": "provided SHA256SUMS.txt (not independently authenticated)",
        "checksum_manifest_sha256": digest(root / "SHA256SUMS.txt")[0] if checksums else None,
        "strict_root": strict_root, "expected_records": len(expected_records),
        "expected_patients": len({p.split("/")[0] for p in expected_records}),
        "files_checked": len(file_rows), "file_status_counts": dict(Counter(r["status"] for r in file_rows)),
        "local_waveform_file_counts": dict(Counter(PurePosixPath(p).suffix for p in local_data)),
        "missing_required_files": [r["path"] for r in file_rows if r["status"] == "missing"],
        "checksum_mismatches": [r["path"] for r in file_rows if r["status"] == "mismatch"],
        "optional_missing_files": optional_missing, "unexpected_patient_files": unexpected,
        "invalid_records": [r["record"] for r in header_errors], "duration_counts": dict(by_duration),
        "errors": errors, "warnings": warnings,
        "scope": "File completeness/calibration-header structure only. Waveform QC and patient split follow in PTB preparation."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "files.csv", file_rows, ("path", "required", "status", "size_bytes", "expected_sha256", "actual_sha256", "message"))
    write_csv(output / "records.csv", records, ("record", "patient", "status", "fs_hz", "samples", "seconds",
              "signals", "standard_leads", "duration_ge50s", "reason"))
    status = "通过，可以进行后续预处理" if not errors else "未通过，先修复缺失或损坏文件"
    lines = ["# PTB文件完整性检查", "", status, "", f"清单记录：{len(expected_records)}；患者：{summary['expected_patients']}。",
             "", "本检查不修改原数据、不训练模型；短于50秒不等于文件损坏，只是本轮S50不使用。", "",
             "## 错误", "", *(["- " + e for e in errors] or ["无"]), "", "## 缺失文件", "",
             *(["- " + p for p in summary["missing_required_files"]] or ["无"]), "", "## 提示", "",
             *(["- " + w for w in warnings] or ["无"])]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("ready_for_preprocessing", "expected_records", "expected_patients",
                     "file_status_counts", "duration_counts", "missing_required_files", "errors")}, ensure_ascii=False), flush=True)
    print("Report: " + str(output), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/qht/huawei/data/PTB-Diagnostic")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strict-root", action="store_true", help="Also require optional root files listed by SHA256SUMS")
    args = parser.parse_args()
    try:
        summary = audit(args.root, args.output_dir, args.strict_root)
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")
    return 0 if summary["ready_for_preprocessing"] else 2


if __name__ == "__main__":
    sys.exit(main())
