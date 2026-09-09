"""Build record-level device interpretation QC without changing raw data or windows.

Only explicit device signal-quality statements are actionable. Expected d6 chest
lead dropout is retained for audit and never marks a limb context channel bad.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ecg12gen.contracts import D12_LEADS, D6_LEADS

QUALITY_RE = re.compile(r"\u4fe1\u53f7\u8d28\u91cf\u5dee\s*[\(\uFF08]\s*([^\)\uFF09]*)\s*[\)\uFF09]")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _statements(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    texts: list[str] = []
    for annotation in root.iter():
        if _local_name(annotation.tag) != "annotation":
            continue
        has_statement_code = any(
            _local_name(child.tag) == "code" and child.attrib.get("code") == "MDC_ECG_INTERPRETATION_STATEMENT"
            for child in annotation
        )
        if not has_statement_code:
            continue
        for child in annotation:
            if _local_name(child.tag) == "value":
                value = "".join(child.itertext()).strip()
                if value:
                    texts.append(value)
    return texts


def _named_bad_leads(statements: list[str]) -> list[str]:
    named: set[str] = set()
    for statement in statements:
        for match in QUALITY_RE.finditer(statement):
            for token in re.split(r"[,\uFF0C\u3001/\s]+", match.group(1).strip()):
                if token in D12_LEADS:
                    named.add(token)
    return [lead for lead in D12_LEADS if lead in named]


def _mask_text(mask: list[bool]) -> str:
    return "|".join("1" if value else "0" for value in mask)


def build(raw_manifest: Path, data_root: Path) -> list[dict[str, str]]:
    with raw_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    results: list[dict[str, str]] = []
    for row in source_rows:
        device_type = row.get("device_type", "")
        if device_type not in {"ecg_machine_d6", "ecg_machine_d12"}:
            continue
        xml_path = data_root / row["source_path"]
        statements = _statements(xml_path)
        bad_all = _named_bad_leads(statements)
        bad_observed = [lead for lead in bad_all if lead in D6_LEADS] if device_type == "ecg_machine_d6" else []
        target_mask = [lead not in bad_all for lead in D12_LEADS] if device_type == "ecg_machine_d12" else [False] * len(D12_LEADS)
        input_mask = [lead not in bad_observed for lead in D6_LEADS] if device_type == "ecg_machine_d6" else [False] * len(D6_LEADS)
        reliable_target_count = sum(target_mask)
        results.append({
            "record_id": row["record_id"], "device_type": device_type,
            "source_path": row["source_path"], "split": row["split"],
            "interpretation_statements": " | ".join(statements),
            "has_signal_quality_warning": str(bool(bad_all)).lower(),
            "bad_leads_all": ",".join(bad_all),
            "bad_observed_input_leads": ",".join(bad_observed),
            "target_lead_mask": _mask_text(target_mask),
            "input_lead_mask": _mask_text(input_mask),
            "reliable_target_lead_count": str(reliable_target_count) if device_type == "ecg_machine_d12" else "",
            "d12_direct_supervision_eligible": str(device_type == "ecg_machine_d12" and reliable_target_count >= 6).lower(),
            "d6_context_training_eligible": str(device_type == "ecg_machine_d6" and not bad_observed).lower(),
        })
    return sorted(results, key=lambda item: item["record_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the device-interpretation QC sidecar from raw ECG XML.")
    parser.add_argument("--raw-manifest", default="metadata/raw_record_manifest.csv")
    parser.add_argument("--data-root", default="..", help="Directory containing Data/")
    parser.add_argument("--output", default="metadata/device_interpretation_qc.csv")
    args = parser.parse_args()
    rows = build(Path(args.raw_manifest), Path(args.data_root))
    fields = [
        "record_id", "device_type", "source_path", "split", "interpretation_statements",
        "has_signal_quality_warning", "bad_leads_all", "bad_observed_input_leads",
        "target_lead_mask", "input_lead_mask", "reliable_target_lead_count",
        "d12_direct_supervision_eligible", "d6_context_training_eligible",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {output}")


if __name__ == "__main__":
    main()