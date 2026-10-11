"""Read-only real-data audit after refresh: splits, masks, record lengths and strict de-duplication."""
from __future__ import annotations
import csv
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.csv_io import read_csv_rows
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.contracts import SupervisionMode
from ecg12gen.evaluate import evaluate_record_predictions
from ecg12gen.record_context import collate_record_context
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def audit():
    config = ECGDataConfig.from_yaml(ROOT / "configs/common.yaml")
    refresh = json.loads((ROOT / "metadata/data_refresh_audit.json").read_text(encoding="utf-8"))
    previous, _ = read_csv_rows(Path(refresh["backup_directory"]) / "subject_split.csv")
    current, _ = read_csv_rows(config.path("subject_split_csv"))
    assignments = {r["subject_id"]: r["split"] for r in current}
    assert all(assignments[r["subject_id"]] == r["split"] for r in previous)
    data_root = ROOT.parent
    for row in refresh["source_fingerprints"]:
        source = data_root / row["source_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"], source
    raw_info, encoding = read_csv_rows(data_root / "Data/userinfobean.csv")
    canonical_info, _ = read_csv_rows(config.path("demographics_csv"))
    assert raw_info == canonical_info
    preprocessor_config = PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml")
    scales = {source: np.ones(c, np.float32) for source, c in preprocessor_config.expected_leads.items()}
    preprocessor = ECGPreprocessor(preprocessor_config, scales)  # audit-only, NOT a fitted training artifact
    result = {"existing_subjects_preserved": len(previous), "total_subjects": len(current),
              "raw_signal_hashes_verified": len(refresh["source_fingerprints"]),
              "demographics_rows": len(raw_info), "demographics_source_encoding": encoding, "datasets": {}}
    for task in ("task1", "task2"):
        for split in ("train", "validation"):
            dataset = JointAnchorDataset(config, task, split)
            for start in range(0, len(dataset._targets), 64):
                assert np.isfinite(dataset._targets[start:start + 64]).all()
            seen_context, seen_pairs = set(), set()
            kinds, fallbacks = Counter(), Counter()
            padding = gaps = 0
            samples = []
            for i in range(len(dataset)):
                row = dataset._rows[dataset._indices[i]]
                if row["pair_id"] in seen_pairs:
                    continue
                seen_pairs.add(row["pair_id"])
                sample = dataset[i]
                kinds[sample.context_source_type] += 1
                if not sample.input_quality_mask.any() or not sample.context.time_mask.any():
                    fallbacks[sample.context_source_type] += 1
                if len(samples) < 3:
                    samples.append(sample)
                key = (sample.context_source_type, sample.context_record_id)
                if key not in seen_context:
                    seen_context.add(key)
                    context = sample.context
                    for c in range(context.raw_uV.shape[1]):
                        valid = context.raw_uV[:, c][context.time_mask[:, c]]
                        expected = np.median(valid) if valid.size else 0
                        assert np.isclose(expected, context.baseline_uV[c], rtol=1e-6, atol=1e-3)
                    for w, length in enumerate(context.valid_lengths):
                        assert not context.time_mask[w, :, length:].any()
                        padding += context.raw_uV.shape[1] * (5000 - int(length))
                        gaps += int((~context.time_mask[w, :, :length]).sum())
                assert dataset[i + 11].context is sample.context
            if samples:
                batch = collate_record_context(samples, preprocessor)
                assert np.all(batch["context"][~batch["context_time_mask"]] == 0)
                assert np.allclose(batch["context_window_weights"].sum(axis=1), batch["context_available"].astype(float))
            stats = {"cache_target_windows": len(dataset._rows), "eligible_target_windows": len(dataset),
                     "eligible_pairs": len(seen_pairs), "pair_device_counts": dict(kinds),
                     "anchor_only_fallback_pairs": dict(fallbacks), "context_padding_lead_points": padding,
                     "context_gap_lead_points": gaps}
            if split == "validation":
                identity, _ = evaluate_record_predictions(dataset._targets, dataset._targets, task, dataset._rows)
                assert np.isclose(identity["r_missing11"], 1)
                if task == "task2":
                    assert identity["task2_rmse_bonus_score"] == 10
                    assert identity["task2_missing_lead_mean_rmse_uV"] == 0
                stats["identity_r_missing11"] = identity["r_missing11"]
            result["datasets"][task + "_" + split] = stats
            print("PASS:", task, split, json.dumps(stats, ensure_ascii=False), flush=True)
            dataset.close()
            del dataset, samples, sample
            gc.collect()
    strict = StrictD12PretrainDataset(config, SupervisionMode.D12_I_PRETRAIN.value)
    assert len({r["dedup_key"] for r in strict.rows}) == len(strict)
    assert all(assignments[r["subject_id"]] == "train" for r in strict.rows)
    for sample in strict:
        assert sample.target_quality_mask[0] and np.array_equal(sample.X_ecg, sample.Y_12lead[:1])
    result["strict"] = {"windows": len(strict), "target_records": len({r["target_record_id"] for r in strict.rows}),
                        "subjects": len({r["subject_id"] for r in strict.rows}),
                        "source_task_windows": dict(Counter(r["source_task_id"] for r in strict.rows))}
    for array in strict.targets.values():
        array._mmap.close()
    return result


if __name__ == "__main__":
    result = audit()
    output = ROOT / "reports/refreshed_data_checks.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: complete refreshed data audit", json.dumps(result["strict"], ensure_ascii=False), flush=True)
