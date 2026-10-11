"""Build a body-only independent B overlay from raw records; canonical A remains read-only."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.body_scale_ablation import (MANIFEST, OVERLAY_METADATA, OVERLAY_PROTOCOL,
                                        protected_paths, read_rows, sha256_file)
from ecg12gen.body_scale_detrend import B_VARIANT, FILTER_SPEC, detrend_body_scale_record
from ecg12gen.dataset import ECGDataConfig
from ecg12gen.raw_task2 import parse_body_scale_record
from ecg12gen.record_context import window_context_record
from ecg12gen.record_cache import CONTEXT_FIELDS


def snapshots(config):
    paths = protected_paths(config)
    base = config.path("task2_output")
    for split in ("train", "validation"):
        for kind in ("target", "context", "context_valid_mask"):
            paths[f"{split}_{kind}"] = base / f"task2_{split}_{kind}.npy"
    return {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()}


def build(config_path, output_dir):
    config = ECGDataConfig.from_yaml(config_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; use a new output directory")
    base = config.path("task2_output")
    before = snapshots(config)
    assignments = {r["subject_id"]: r["split"] for r in read_rows(config.path("subject_split_csv"))}
    pairs = {r["input_record_id"]: r for r in read_rows(config.path("task2_pair_manifest_csv"))
             if r["input_type"] == "body_scale_d6" and r["pair_status"] == "paired"}
    groups = defaultdict(list)
    for r in read_rows(base / "task2_context_window_metadata.csv"):
        if r["input_type"] == "body_scale_d6":
            groups[(r["split"], r["input_record_id"])].append(r)
    values, valid_arrays = {}, {}
    for split in ("train", "validation"):
        values[split] = np.load(base / f"task2_{split}_context.npy", mmap_mode="r")
        valid_arrays[split] = np.load(base / f"task2_{split}_context_valid_mask.npy", mmap_mode="r")
    windows = {s: [] for s in ("train", "validation")}
    masks = {s: [] for s in windows}
    metadata, records = [], []
    try:
        for ordinal, ((split, record_id), rows) in enumerate(sorted(groups.items()), 1):
            rows.sort(key=lambda r: int(r["start_sample_500hz"]))
            pair = pairs[record_id]
            if assignments.get(pair["subject_id"]) != split or pair["split"] != split:
                raise ValueError("Body context crosses authoritative patient split")
            path = config.path("data_root") / pair["input_path"]
            raw, valid = parse_body_scale_record(path)
            original = window_context_record(raw, "body_scale_d6", valid)
            indices = [int(r["array_index"]) for r in rows]
            if (not np.array_equal(original.raw_uV, values[split][indices]) or
                    not np.array_equal(original.time_mask, valid_arrays[split][indices])):
                raise ValueError(f"Raw body record differs from canonical A: {record_id}")
            filtered, filtered_valid, filtering = detrend_body_scale_record(raw, valid)
            prepared = window_context_record(filtered, "body_scale_d6", filtered_valid)
            if (not np.array_equal(prepared.time_mask, original.time_mask) or
                    not np.array_equal(prepared.valid_lengths, original.valid_lengths) or
                    not np.array_equal(prepared.starts, original.starts)):
                raise ValueError("B changed validity, time positions or tails")
            for w, row in enumerate(rows):
                local = len(windows[split])
                windows[split].append(prepared.raw_uV[w])
                masks[split].append(prepared.time_mask[w])
                metadata.append({**row, "array_index": str(local), "canonical_array_index": row["array_index"],
                                 "input_record_baseline_uV": "|".join(f"{float(v):.9g}" for v in prepared.baseline_uV),
                                 "body_scale_variant": B_VARIANT})
            records.append({"input_record_id": record_id, "split": split, "subject_id": pair["subject_id"],
                            "raw_source": str(path.resolve()), "raw_sha256": sha256_file(path),
                            "physical_samples": raw.shape[1], "context_windows": len(rows),
                            "tail_samples": int(prepared.valid_lengths[-1]), **filtering})
            if ordinal % 25 == 0 or ordinal == len(groups):
                print(f"Prepared body B records: {ordinal}/{len(groups)}", flush=True)
        if not groups:
            raise ValueError("No canonical body contexts")
        if before != snapshots(config):
            raise ValueError("Canonical A/metadata changed during B construction")
        output.mkdir(parents=True, exist_ok=False)
        for split in windows:
            for name, data, dtype in (("context_B", windows[split], np.float32),
                                       ("context_valid_mask_B", masks[split], bool)):
                path = output / f"body_scale_{split}_{name}.npy"
                if data:
                    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(len(data), 6, 5000))
                    for i, item in enumerate(data):
                        array[i] = item
                    array.flush()
                    array._mmap.close()
                else:
                    np.save(path, np.empty((0, 6, 5000), dtype=dtype))
        with (output / OVERLAY_METADATA).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CONTEXT_FIELDS + ["canonical_array_index", "body_scale_variant"])
            writer.writeheader()
            writer.writerows(metadata)
        payloads = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
        manifest = {"protocol": OVERLAY_PROTOCOL, "variant": B_VARIANT, "filter": FILTER_SPEC,
                    "canonical_cache": str(base.resolve()),
                    "canonical_metadata_sha256": {k: v["sha256"] for k, v in before.items() if k in protected_paths(config)},
                    "canonical_files_unchanged": before, "overlay_files_sha256": payloads,
                    "record_counts": dict(Counter(r["split"] for r in records)),
                    "window_counts": {s: len(windows[s]) for s in windows}, "records": records,
                    "model_scales_fitted": False, "training_started": False,
                    "target_anchor_machine_changed": False}
        (output / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "README.md").write_text(
            "# 新数据体脂秤 B 消融缓存\n\n"
            "仅保存body context的B overlay；A、同步I、target、机器d6、划分、QC、评分均不变。\n\n"
            "完整记录采用0.2 Hz二阶Butterworth SOS前后向高通；有缺口时逐导联连续有效段各自滤波，"
            "不插值、不跨缺口。odd padding=min(7500,段长-2)，≤2点段保持不变。"
            "滤波后按全记录有效点统计median，独立切窗保留尾段，mask与A相同。\n\n"
            "读取：JointAnchorDataset(config,'task2',split,body_scale_variant='B_detrend_0p2Hz_then_window')。"
            "默认仍A。B训练需准备并冻结B体脂秤context scale，d12/anchor尺度与A保持兼容；"
            "checkpoint记录variant/filter协议。本次不拟合模型scale或训练。\n\n"
            "测试传原始完整可见context给prepare_record_context_inference，指定相同body_scale_variant；"
            "不要预先滤波后重复处理。Task2正式总分仍要求两设备预测，分别算分再1:1加权。\n\n"
            "滤波API：[SciPy butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)、"
            "[sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)。\n",
            encoding="utf-8")
        return manifest
    finally:
        for array in list(values.values()) + list(valid_arrays.values()):
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/common.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = ECGDataConfig.from_yaml(args.config)
    result = build(args.config, args.output_dir or cfg.path("task2_body_scale_ablation"))
    print("PASS body B build", json.dumps({k: result[k] for k in ("record_counts", "window_counts")}), flush=True)


if __name__ == "__main__":
    main()
