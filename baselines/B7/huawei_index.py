"""Bridge existing Huawei metadata to B7's explicit same-record-I index.

Watch/d6 records are independent contexts in B5, not point-aligned anchors.
This bridge preserves the target-I experiment and does not align those contexts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path


def read_csv(path):
    blob = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return list(csv.DictReader(io.StringIO(blob.decode(encoding))))
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Unsupported CSV encoding: {path}")


def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise ValueError(f"Missing/duplicate {key}: {value}")
        result[value] = row
    return result


def build_index(metadata, data_root, anchor_source, d12_root=None):
    if anchor_source != "same-record-target-I":
        raise ValueError("Only the explicitly declared synchronous target-I protocol is supported")
    metadata, data_root = Path(metadata), Path(data_root).resolve()
    d12_root = Path(d12_root).resolve() if d12_root else data_root / "心电图机d12"
    if not d12_root.is_dir():
        raise FileNotFoundError(f"Missing d12 directory; supply --d12-root: {d12_root}")
    splits = unique(read_csv(metadata / "subject_split.csv"), "subject_id")
    raw = unique(read_csv(metadata / "raw_record_manifest.csv"), "record_id")
    qc = unique(read_csv(metadata / "device_interpretation_qc.csv"), "record_id")
    files = defaultdict(list)
    for path in d12_root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".xml":
            files[path.name].append(path.resolve())
    rows, seen, owners, masks = [], set(), {}, {}
    excluded = Counter()
    groups = {"watch_ecg": "task1", "ecg_machine_d6": "machine", "body_scale_d6": "body"}
    for filename in ("pair_manifest_task1.csv", "pair_manifest_task2.csv"):
        for pair in read_csv(metadata / filename):
            if pair["pair_status"] != "paired" or pair["target_quality_status"] != "usable":
                excluded["unpaired_or_unusable_target"] += 1
                continue
            if pair.get("training_policy") in ("review", "exclude", "drop"):
                excluded["excluded_pair_policy"] += 1
                continue
            subject, target_id = pair["subject_id"], pair["target_record_id"]
            split = splits[subject]["split"]
            if split not in ("train", "validation", "test") or pair["split"] != split:
                raise ValueError(f"Pair differs from authoritative split: {pair['pair_id']}")
            source, quality = raw[target_id], qc[target_id]
            if source["subject_id"] != subject or source["split"] != split or quality["split"] != split:
                raise ValueError(f"Target identity/split disagreement: {target_id}")
            if quality["device_type"] != "ecg_machine_d12":
                raise ValueError(f"Target QC is not d12: {target_id}")
            tokens = quality["target_lead_mask"].split("|")
            if len(tokens) != 12 or set(tokens) - {"0", "1"}:
                raise ValueError(f"Invalid fixed lead QC: {target_id}")
            mask = [v == "1" for v in tokens]
            if source["quality_status"] != "usable":
                excluded["unusable_raw_target"] += 1
                continue
            if split == "train" and (quality["d12_direct_supervision_eligible"] != "true" or not mask[0]):
                excluded["ineligible_training_target_or_I"] += 1
                continue
            if target_id in owners and (owners[target_id] != (subject, split) or masks[target_id] != mask):
                raise ValueError(f"Physical target identity/QC changed: {target_id}")
            owners[target_id], masks[target_id] = (subject, split), mask
            # Resolve by unique filename to handle Windows paths, Data/data relocation,
            # and historic damaged directory names. Ambiguity is never guessed.
            basename = pair["target_path"].replace("\\", "/").rsplit("/", 1)[-1]
            if basename != source["source_path"].replace("\\", "/").rsplit("/", 1)[-1]:
                raise ValueError(f"Pair/raw target path disagreement: {target_id}")
            candidates = files[basename]
            if len(candidates) != 1:
                raise ValueError(f"Expected one raw XML for {basename}, found {len(candidates)}")
            record_id = pair["pair_id"]
            if record_id in seen:
                raise ValueError(f"Duplicate pair ID: {record_id}")
            seen.add(record_id)
            group = groups[pair["input_type"]]
            if (filename == "pair_manifest_task1.csv") != (group == "task1"):
                raise ValueError("Pair task/source disagreement")
            rows.append({"record_id": record_id, "subject_id": subject,
                         "target_record_id": target_id, "source": "huawei", "group": group,
                         "split": split, "path": str(candidates[0]), "lead_mask": mask,
                         "input_reliable": mask[0], "anchor_source": "same_record_target_I",
                         "original_context_record_id": pair["input_record_id"],
                         "original_context_used": False})
    for split in ("train", "validation"):
        for group in groups.values():
            eligible = [r for r in rows if r["split"] == split and r["group"] == group and r["input_reliable"]]
            if not eligible:
                raise ValueError(f"No reliable same-record-I rows for {split}/{group}")
            if split == "validation" and any(not any(r["lead_mask"][j] for r in eligible) for j in range(1, 12)):
                raise ValueError(f"Validation group has a missing QC-eligible lead: {group}")
    return rows, {"anchor_source": "same_record_target_I", "context_used": False,
                  "demographics": "unknown (no demographic mapping in this bridge)",
                  "counts": dict(Counter(f"{r['split']}/{r['group']}" for r in rows)),
                  "excluded": dict(excluded), "records": len(rows), "raw_data_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--d12-root", type=Path, help="Defaults to DATA_ROOT/心电图机d12")
    parser.add_argument("--anchor-source", choices=("same-record-target-I",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError("Use a new index output; existing files are preserved")
    rows, audit = build_index(args.metadata, args.data_root, args.anchor_source, args.d12_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
