"""Read-only dataset preflight and explicit train-only scale fitting. No Torch or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path
from .data import HuaweiTrainDataset, HuaweiValidationDataset, demographics_table, fit_huawei_scales, load_preprocessor
from .public_adapter import PTBXLDataset, locate_ptbxl


def preflight(config: dict[str, Any], fit_scales: bool = False, check_public: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {"training_started": False, "downloads_started": False,
                              "stage": config["stage"], "errors": []}
    preprocessor = None
    try:
        if fit_scales:
            scales_path = resolve_path(config, "scales")
            if scales_path.exists():
                raise FileExistsError("Scale artifact already exists; reuse it or choose a new scales path, do not silently refit")
            preprocessor, scale_report = fit_huawei_scales(config)
            scales_path.parent.mkdir(parents=True, exist_ok=True)
            preprocessor.save(scales_path)
            report["scale_fit"] = scale_report
        else:
            preprocessor = load_preprocessor(config)
        report["scales"] = {source: values.tolist() for source, values in preprocessor.scale_uV_by_source.items()}
        demographics = demographics_table(config)
        report["demographics"] = demographics.audit()
        train = HuaweiTrainDataset(config, preprocessor, demographics)
        first = train[0]
        report["huawei_train"] = {"windows": len(train), "subjects": len({r['subject_id'] for r in train.rows}),
                                  "manifest_sha256": train.manifest_digest,
                                  "anchor_shape": list(first["anchor"].shape), "target_shape": list(first["target"].shape)}
        report["huawei_validation"] = {}
        for task in config["validation"]["tasks"]:
            dataset = HuaweiValidationDataset(config, task, preprocessor, demographics)
            dataset[0]
            report["huawei_validation"][task] = {"windows": len(dataset),
                "pairs": len({r['pair_id'] for r in dataset.rows}), "manifest_sha256": dataset.manifest_digest}
    except (OSError, ValueError) as exc:
        report["errors"].append(f"Huawei/scales: {exc}")
    if check_public or config["stage"] == "public":
        try:
            root = locate_ptbxl(resolve_path(config, "data_root"), resolve_path(config, "ptbxl_root"))
            reports = {}
            for split in ("train", "validation", "test"):
                dataset = PTBXLDataset(root, split, preprocessor, check_files=False)
                reports[split] = dataset.file_audit()
                if preprocessor is not None and reports[split]["missing_count"] == 0:
                    dataset[0]  # Physical-unit and lead-order validation, not model inference/training.
            report["public"] = {"root": str(root), "folds": reports}
            if any(value["missing_count"] for value in reports.values()):
                report["errors"].append("PTB-XL download is incomplete; records500 is required. Training was not started.")
        except (OSError, ValueError, ImportError) as exc:
            report["errors"].append(f"PTB-XL: {exc}")
    report["ready"] = not report["errors"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fit-scales", action="store_true", help="Fit scales using the deduplicated Huawei train index only")
    parser.add_argument("--check-public", action="store_true", help="Check local PTB-XL download completeness, no download")
    parser.add_argument("--data-root", help="Override public/user-info Data directory")
    parser.add_argument("--huawei-data-root", help="Override directory containing task1_output_v2/task2_output")
    parser.add_argument("--scales", help="Override scale artifact path")
    parser.add_argument("--report", help="Optional JSON report output")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("data_root", "huawei_data_root", "scales"):
        value = getattr(args, key)
        if value:
            config["paths"][key] = str(Path(value).resolve())
    report = preflight(config, args.fit_scales, args.check_public)
    from . import ARCHITECTURE_ID
    from .config import ModelConfig
    report['architecture_id'] = ARCHITECTURE_ID
    report['architecture_hash'] = ModelConfig.from_dict(config['model']).fingerprint
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
