"""Freeze the audited PTB-XL/Chapman public corpus before physiology-decomposed regression pretraining."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import PROTOCOL
from .common import sha, write
from .public_data import audit_public
from baselines.B5.config import load_config
from baselines.B5.data import load_preprocessor, preprocessing_config
from ecg12gen.preprocessing import ECGPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl-root", required=True)
    parser.add_argument("--chapman-root", required=True)
    parser.add_argument("--output", required=True, help="Empty study directory")
    parser.add_argument("--chapman-patient-map")
    parser.add_argument("--huawei-config", required=True,
                        help="Existing B5 config; its frozen Huawei scale is copied into this study")
    parser.add_argument("--scales", help="Absolute path to the existing frozen Huawei preprocessing_scales.npz")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty --output directory")
    # Validate the frozen normalization artifact before the expensive public audit.
    huawei = load_config(args.huawei_config, data_only=True)
    preprocessor = (ECGPreprocessor.load(preprocessing_config(huawei), args.scales)
                    if args.scales else load_preprocessor(huawei))
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_public(args.ptbxl_root, args.chapman_root, output / "public", args.chapman_patient_map)
    preprocessor.save(output / "scales.npz")
    write(output / "study.json", {"protocol": PROTOCOL, "completed": True, "public_audit": audit,
                                  "manifest_sha256": sha(output / "public" / "manifest.json"),
                                  "audit_summary_sha256": sha(output / "public" / "audit_summary.json"),
                                  "scales_sha256": sha(output / "scales.npz")})
    print(output / "study.json")


if __name__ == "__main__":
    main()
