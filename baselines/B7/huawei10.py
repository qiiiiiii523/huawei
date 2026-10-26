"""Immutable 10s views of aligned prepared Huawei records; no waveform recaching."""
import argparse
import copy
import json
import numpy as np
from pathlib import Path
from .checkpoint import file_sha256
from .config import read_json
from .data import GROUPS, load_record, read_manifest
from .public_qc import PUBLIC_SOURCES

VIEW_PROTOCOL = "B7-Huawei-real10-nonoverlap-v1"


def build(source, public, output):
    source, public, output = map(lambda p: Path(p).resolve(), (source, public, output))
    if output.exists():
        raise FileExistsError("Use a new Huawei10 index path")
    original, scale_source = read_manifest(source), read_manifest(public)
    rows = []
    for row in original["records"]:
        if row["source"] in PUBLIC_SOURCES or row["group"] not in GROUPS:
            raise ValueError("Huawei-only paired manifest required")
        if "window_start" in row or "window_stop" in row:
            raise ValueError("Source must be unsliced prepared records")
        anchor, target, mask, _ = load_record(row)
        length = anchor.shape[-1]
        if length < 5000 or length % 5000:
            raise ValueError("Need complete aligned real10s windows, no padding or discarded tail")
        for start in range(0, length, 5000):
            item = copy.deepcopy(row)
            item["path"] = row["resolved_path"]
            item.pop("resolved_path", None)
            item.update(record_id=f"{row['record_id']}:10s={start}",
                        target_record_id=f"{row['target_record_id']}:10s={start}",
                        parent_record_id=row["record_id"], parent_target_record_id=row["target_record_id"],
                        window_start=start, window_stop=start + 5000)
            segment = target[:, start:start + 5000]
            # Preserve parent QC; exclude an exactly constant or nonfinite short lead.
            safe = np.where(mask[:, None], segment, 0.)
            view_mask = mask & np.isfinite(safe).all(-1) & (np.std(safe, axis=-1) > 0)
            item["view_lead_mask"] = view_mask.tolist()
            item["input_reliable"] = bool(row['input_reliable'] and np.std(anchor[:, start:start + 5000]) > 0)
            rows.append(item)
    value = {k: v for k, v in original.items() if k != "records"}
    value.update(records=rows, huawei_window_protocol=VIEW_PROTOCOL,
                 scales_uV=scale_source['scales_uV'], scale_source=str(public),
                 parent_scales_uV=original['scales_uV'],
                 window_seconds=10, view_source_manifest=str(source), view_source_sha256=file_sha256(source),
                 public_scale_manifest=str(public), public_scale_manifest_sha256=file_sha256(public),
                 alignment_policy="inherit prepared anchor/target simultaneous slicing; no new alignment inferred")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate before publishing, including inherited patient/target ownership.
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    read_manifest(tmp)
    tmp.replace(output)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--public-manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    value = build(a.source_manifest, a.public_manifest, a.output)
    print(json.dumps({"manifest": a.output, "records": len(value['records']), "waveform_files_written": 0}))


if __name__ == "__main__":
    main()
