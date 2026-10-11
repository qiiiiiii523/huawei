"""Temporary synthetic cache tests; never reads/writes production waveforms or trains."""
from __future__ import annotations
import csv
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.contracts import ContractError
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from ecg12gen.record_context import (window_context_record, window_target_record, collate_record_context,
                                   prepare_record_context_inference)
from ecg12gen.record_cache import IndependentRecordCacheBuilder
from ecg12gen.raw_task2 import parse_body_scale_record


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair(name, source, context_id):
    return {"pair_id": name, "subject_id": "patient", "split": "train", "input_type": source,
            "input_record_id": context_id, "target_record_id": "target", "target_quality_status": "usable",
            "input_quality_status": "usable", "pair_status": "paired"}


def main():
    config = PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml")
    scales = {"d12": np.linspace(200, 800, 12).astype(np.float32), "watch_ecg": np.asarray([500], np.float32),
              "ecg_machine_d6": np.full(6, 500, np.float32), "body_scale_d6": np.full(6, 500, np.float32)}
    scales["ecg_machine_i"] = scales["d12"][:1].copy()
    preprocessor = ECGPreprocessor(config, scales)
    target = np.broadcast_to(np.sin(np.linspace(0, 40, 61000))[None] * 300 + 150, (12, 61000)).astype(np.float32).copy()
    context = np.arange(7600, dtype=np.float32)[None] + 100
    valid = np.ones(7600, bool)
    valid[100:200] = False
    context[:, 100:200] = 1e9  # must never influence baseline
    prepared = window_context_record(context, "watch_ecg", valid)
    assert prepared.raw_uV.shape == (2, 1, 5000)
    assert prepared.valid_lengths.tolist() == [5000, 2600]
    assert prepared.baseline_uV[0] == np.median(context[0, valid])
    assert not prepared.time_mask[1, :, 2600:].any()
    assert not prepared.time_mask[0, :, 100:200].any()
    assert window_target_record(target).shape == (12, 12, 5000)
    try:
        window_target_record(target[:, :59999])
        raise AssertionError("Short target was padded")
    except ContractError:
        pass
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        body_csv = root / "body.csv"
        with body_csv.open("w", encoding="utf-8", newline="") as handle:
            handle.write("采样率:500\n数据单位:mV\nIndex,1,2,9,10,11,12\n100,1,2,3,4,5,6\n101,2,3,4,5,6,7\n103,3,4,5,6,7,8\n")
        body_raw, body_valid = parse_body_scale_record(body_csv)
        assert body_raw.shape == (6, 4) and not body_valid[:, 2].any()
        assert body_raw[0, 0] == 1000  # mV -> uV, nonzero initial index is not padded
        pairs1 = [pair("watch", "watch_ecg", "watch_record"), pair("short", "watch_ecg", "short_record"),
                  pair("absent", "watch_ecg", "")]
        builder = IndependentRecordCacheBuilder("task1")
        builder.add_pair(pairs1[0], target, context, valid)
        builder.add_pair(pairs1[1], target, np.ones((1, 700), np.float32))
        builder.add_pair(pairs1[2], target, None)
        audit = builder.write(root / "task1")
        assert audit["target_window_counts"]["train"] == 36
        assert audit["context_window_counts"]["train"] == 3
        pairs2 = [pair("machine", "ecg_machine_d6", "machine_record"), pair("body", "body_scale_d6", "body_record")]
        b2 = IndependentRecordCacheBuilder("task2")
        b2.add_pair(pairs2[0], target, np.ones((6, 23000), np.float32))
        b2.add_pair(pairs2[1], target, np.ones((6, 6300), np.float32))
        b2.write(root / "task2")
        write_csv(root / "split.csv", [{"subject_id": "patient", "split": "train"}])
        write_csv(root / "pairs1.csv", pairs1)
        write_csv(root / "pairs2.csv", pairs2)
        quality = np.ones(12, bool)
        quality[8] = False
        qc = [{"record_id": "target", "device_type": "ecg_machine_d12",
               "target_lead_mask": "|".join(str(int(v)) for v in quality), "input_lead_mask": "1|1|1|1|1|1",
               "reliable_target_lead_count": "11", "d12_direct_supervision_eligible": "true", "d6_context_training_eligible": "false"},
              {"record_id": "machine_record", "device_type": "ecg_machine_d6", "target_lead_mask": "|".join(["1"] * 12),
               "input_lead_mask": "0|1|1|1|1|1", "reliable_target_lead_count": "12",
               "d12_direct_supervision_eligible": "false", "d6_context_training_eligible": "false"}]
        write_csv(root / "qc.csv", qc)
        common = ECGDataConfig.from_yaml(ROOT / "configs/common.yaml")
        paths = {"task1_output": str(root / "task1"), "task2_output": str(root / "task2"),
                 "subject_split_csv": str(root / "split.csv"), "task1_pair_manifest_csv": str(root / "pairs1.csv"),
                 "task2_pair_manifest_csv": str(root / "pairs2.csv"), "device_interpretation_qc_csv": str(root / "qc.csv")}
        cfg = ECGDataConfig({**common.raw, "paths": {**common.raw["paths"], **paths}}, ROOT)
        ds = JointAnchorDataset(cfg, "task1", "train")
        assert len(ds) == 36
        assert ds[0].context is ds[11].context  # all target windows share full context
        assert np.array_equal(ds[11].context_ecg, ds[0].context_ecg)
        assert not ds[0].target_quality_mask[8]
        batch = collate_record_context([ds[0], ds[12], ds[24]], preprocessor)
        assert batch["context"].shape == (3, 2, 1, 5000)
        assert batch["context_window_mask"].tolist() == [[True, True], [True, False], [False, False]]
        assert batch["context_available"].tolist() == [True, True, False]
        assert np.all(batch["context"][~batch["context_time_mask"]] == 0)
        assert np.allclose(batch["context_window_weights"][0], [4900 / 7500, 2600 / 7500])
        assert np.allclose(batch["target"][0] * scales["d12"][:, None], ds[0].Y_12lead, atol=1e-4)
        dropped = collate_record_context([ds[0]], preprocessor, 1, np.random.default_rng(42))
        assert not dropped["context_available"].any() and not dropped["context_time_mask"].any()
        ds2 = JointAnchorDataset(cfg, "task2", "train")
        assert len(ds2) == 24  # bad context never deletes valid target
        batch2 = collate_record_context([ds2[0], ds2[12]], preprocessor)
        assert batch2["context_available"].tolist() == [False, True]
        assert {r["input_type"] for r in batch2["evaluation_metadata"]} == {"ecg_machine_d6", "body_scale_d6"}
        ds.close()
        ds2.close()
    inference = prepare_record_context_inference(target[:1], context, "watch_ecg", preprocessor, valid)
    assert inference["anchor_i"].shape == (12, 1, 5000)
    assert inference["context"].shape == (2, 1, 5000)
    assert "target" not in inference
    print("PASS: target 120s independent of context; retained masked tail/gaps; valid-only median; record association; batch padding/dropout; QC/missing context fallback; target-free inference")


if __name__ == "__main__":
    main()
