"""P-prime ONLY: QC-v2 public10, random initialization, final EMA validation.

This entry point never launches Huawei/PTBDB or an A/Q/B pipeline.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from .config import read_json
from .data import load_record, read_manifest
from .public_qc import PUBLIC_QC_VERSION, PUBLIC_SOURCES
from .train import train, validate_settings


def check_public_run(config, manifest):
    _, lc, sc = validate_settings(config)
    if config["stage"] != "public10" or sc.window_seconds != 10 or sc.core_seconds != 10:
        raise ValueError("P-prime runs public10 only, with full real 10s input/output")
    if lc.pcc_leads != 11:
        raise ValueError("P-prime requires PCC11 supervision")
    if config["training"].get("validation_mode") != "final_only":
        raise ValueError("P-prime uses final-only full validation")
    if manifest.get("public_qc_version") != PUBLIC_QC_VERSION:
        raise ValueError("P-prime requires the corrected QC-v2 manifest/cache")
    if {r["source"] for r in manifest["records"]} != PUBLIC_SOURCES:
        raise ValueError("P-prime requires PTB-XL and Chapman only")
    if not any(r["split"] == "train" and r["input_reliable"] for r in manifest["records"]):
        raise ValueError("P-prime needs reliable training records")
    print(f"[P-prime preflight] checking {len(manifest['records'])} cached records; no GPU generation", flush=True)
    for index, row in enumerate(manifest["records"], 1):
        anchor, target, mask, _ = load_record(row)
        if anchor.shape[-1] != 5000 or target.shape[-1] != 5000:
            raise ValueError("P-prime needs actual 5000-point records; no short-record padding")
        if not row["input_reliable"] or not mask[0]:
            raise ValueError("QC-v2 P-prime must exclude unreliable input I")
        if index % 2000 == 0 or index == len(manifest["records"]):
            print(f"[P-prime preflight] checked={index}/{len(manifest['records'])}", flush=True)


def run(args):
    config = read_json(args.config)
    manifest = read_manifest(args.manifest)
    check_public_run(config, manifest)
    print(f"[P-prime] QC-v2 checked; 500Hz/5000 samples; head={config['model'].get('output_head', 'linear')}; "
          f"updates={config['training']['updates']}; resume={args.resume}; no downstream stages", flush=True)
    return train(Namespace(config=args.config, manifest=args.manifest, output=args.output,
                           device=args.device, resume=args.resume, init_checkpoint=None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Shared QC-v2 public10/manifest.json")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/b7_pprime.json"))
    p.add_argument("--output", required=True, help="New independent training directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true", help="Resume this exact run's last.pt")
    run(p.parse_args())


if __name__ == "__main__":
    main()
