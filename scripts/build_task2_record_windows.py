"""Build independent Task2 caches only when explicitly run; no splitting or training."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.contracts import D12_LEADS, D6_LEADS
from ecg12gen.raw_task1 import parse_d12_xml
from ecg12gen.raw_task2 import parse_body_scale_record
from ecg12gen.record_cache import IndependentRecordCacheBuilder
from scripts.build_task1_record_windows import _read_csv, _resolve, _resample_d12


def build(data_root: Path, repository_root: Path, output_dir: Path):
    metadata = repository_root / "metadata"
    pairs = _read_csv(metadata / "pair_manifest_task2.csv")
    splits = {r["subject_id"]: r["split"] for r in _read_csv(metadata / "subject_split.csv")}
    writer = IndependentRecordCacheBuilder("task2")
    rejected = []
    for pair in pairs:
        if pair["pair_status"] != "paired" or pair["target_quality_status"] != "usable":
            continue
        if splits.get(pair["subject_id"]) != pair["split"]:
            raise ValueError("Pair crosses authoritative subject split")
        target_record = parse_d12_xml(_resolve(data_root, pair["target_path"]), D12_LEADS)
        target = _resample_d12(target_record.signal_uV, target_record.sampling_rate_hz)
        if target.shape[1] < 60000:
            rejected.append({"pair_id": pair["pair_id"], "reason": "target_under_120_seconds"})
            continue
        context, valid = None, None
        if pair["input_quality_status"] == "usable":
            path = _resolve(data_root, pair["input_path"])
            if pair["input_type"] == "body_scale_d6":
                context, valid = parse_body_scale_record(path)
            elif pair["input_type"] == "ecg_machine_d6":
                record = parse_d12_xml(path, D6_LEADS)
                context = _resample_d12(record.signal_uV, record.sampling_rate_hz)
            else:
                raise ValueError("Unknown Task2 source")
        writer.add_pair(pair, target, context, valid)
    if not writer.target_rows:
        raise ValueError("No eligible 120-second target records")
    audit = writer.write(output_dir)
    audit["rejected_short_targets"] = rejected
    audit["subject_split_source"] = str(metadata / "subject_split.csv")
    (output_dir / "task2_build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "task2_record_context_v3")
    args = parser.parse_args()
    print(json.dumps(build(args.data_root.resolve(), ROOT, args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
