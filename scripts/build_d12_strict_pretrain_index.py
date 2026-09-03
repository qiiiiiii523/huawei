"""Build a de-duplicated, train-only index for strict d12 self-supervised pretraining."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.dataset import ECGDataConfig

FIELDS = [
    "strict_id", "source_task_id", "source_split", "source_array_index", "subject_id",
    "target_record_id", "window_id", "window_index", "start_sample_500hz",
    "end_sample_500hz_exclusive", "dedup_key",
]


def rows_for_task(config: ECGDataConfig, task_id: str) -> list[dict[str, str]]:
    task_dir = config.path(f"{task_id}_output")
    path = task_dir / f"{task_id}_window_metadata.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        if row.get("split") != "train" or row.get("quality_status") != "usable":
            continue
        target_id = row.get("target_record_id", "")
        start = row.get("start_sample_500hz", "")
        end = row.get("end_sample_500hz_exclusive", "")
        if not target_id or not start or not end:
            raise ValueError(f"{task_id} row lacks d12 de-duplication fields: {row.get('window_id')}")
        result.append({
            "source_task_id": task_id,
            "source_split": "train",
            "source_array_index": row["array_index"],
            "subject_id": row["subject_id"],
            "target_record_id": target_id,
            "window_id": row["window_id"],
            "window_index": row["window_index"],
            "start_sample_500hz": start,
            "end_sample_500hz_exclusive": end,
            "dedup_key": f"{target_id}:{start}:{end}",
        })
    return result


def build(config_path: str | Path, output_path: str | Path) -> list[dict[str, str]]:
    config = ECGDataConfig.from_yaml(config_path)
    split_path = config.path("subject_split_csv")
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        subject_split = {row["subject_id"]: row["split"] for row in csv.DictReader(handle)}
    candidates = rows_for_task(config, "task1") + rows_for_task(config, "task2")
    selected: dict[str, dict[str, str]] = {}
    # Deterministic preference avoids duplicate d12 windows while keeping one readable source array.
    for row in sorted(candidates, key=lambda item: (item["dedup_key"], item["source_task_id"], int(item["source_array_index"]))):
        if subject_split.get(row["subject_id"]) != "train":
            raise ValueError(f"Validation or unknown subject reached strict index: {row['subject_id']}")
        selected.setdefault(row["dedup_key"], row)
    indexed = []
    for ordinal, row in enumerate(sorted(selected.values(), key=lambda item: (item["target_record_id"], int(item["start_sample_500hz"])))):
        indexed.append({"strict_id": f"D12_STRICT_{ordinal:07d}", **row})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(indexed)
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "common.yaml"))
    parser.add_argument("--output", default=str(ROOT / "metadata" / "d12_strict_pretrain_index.csv"))
    args = parser.parse_args()
    rows = build(args.config, args.output)
    print(f"Wrote {len(rows)} unique train-only d12 windows to {args.output}")


if __name__ == "__main__":
    main()