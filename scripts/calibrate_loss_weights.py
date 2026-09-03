"""Summarize unweighted loss components logged from train-only dry-run batches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Train-only dry-run JSONL: split, batch_index, and loss component values.")
    parser.add_argument("--output", required=True, help="YAML summary path (do not overwrite configs/losses.yaml automatically).")
    parser.add_argument("--max-batches", type=int, default=200)
    args = parser.parse_args()
    rows = []
    with Path(args.input_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("Calibration accepts train-only loss logs")
            rows.append(row)
            if len(rows) >= args.max_batches:
                break
    if not rows:
        raise ValueError("No train loss rows supplied")
    excluded = {"split", "batch_index", "step"}
    components = sorted({key for row in rows for key, value in row.items() if key not in excluded and isinstance(value, (int, float))})
    summary = {"calibration_batches": len(rows), "split": "train", "median_unweighted_losses": {key: sorted(float(row[key]) for row in rows if key in row)[len([row for row in rows if key in row]) // 2] for key in components}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Wrote train-only calibration summary for {len(rows)} batches to {output}")


if __name__ == "__main__":
    main()