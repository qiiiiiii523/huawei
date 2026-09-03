"""检查 main 中固定的实验协议，不训练模型也不读取原始信号。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.evaluate import competition_score
from ecg12gen.training import load_training_protocol, seed_everything


def main() -> None:
    with (ROOT / "configs" / "data_spec.yaml").open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    with (ROOT / "configs" / "common.yaml").open(encoding="utf-8") as handle:
        common = yaml.safe_load(handle)
    with (ROOT / "configs" / "quality_rules.yaml").open(encoding="utf-8") as handle:
        quality = yaml.safe_load(handle)
    with (ROOT / "configs" / "preprocessing.yaml").open(encoding="utf-8") as handle:
        preprocessing = yaml.safe_load(handle)
    protocol = load_training_protocol(ROOT / "configs" / "training_protocol_v1.yaml")

    assert spec["ecg"]["model_internal_unit"] == "μV" and common["signal"]["ecg_unit"] == "μV"
    assert spec["ecg"]["model_internal_sampling_rate_hz"] == 500 and common["signal"]["ecg_sampling_rate_hz"] == 500
    assert spec["windowing"] == {"length_sec": 10, "step_sec": 10}
    assert common["signal"]["window_seconds"] == 10 and common["signal"]["window_step_seconds"] == 10 and common["signal"]["window_samples"] == 5000
    assert spec["ecg"]["twelve_lead_order"] == ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert spec["ppg"]["raw_sampling_rate_hz"] == 100 and spec["ppg"]["input_to_model_v1"] is False
    assert common["signal"]["acceleration_sampling_rate_hz"] == 100 and common["runtime"]["use_acceleration"] is False
    assert quality["training_eligibility"]["under_30_sec"] == "exclude_from_training"
    assert common["paths"]["training_protocol_config"] == "configs/training_protocol_v1.yaml"

    assert preprocessing["protocol_status"] == "frozen_for_b0_b1_b2_m1"
    assert preprocessing["raw_data_mutation"] is False
    assert preprocessing["target_baseline_policy"] == "train_label_only_predict_at_inference"
    assert preprocessing["output_contract"]["prohibited_at_inference"] == "using_true_target_baseline_uV"
    assert protocol["data"]["preprocessing"] == "unified_runtime_preprocessing_v1"
    assert protocol["model_family_policy"]["b0_b1_b2_adapter"] == "forbidden"
    assert protocol["validation"]["official_view"] == "raw_uV"

    with (ROOT / "metadata" / "subject_split.csv").open(encoding="utf-8-sig", newline="") as handle:
        splits = [row["split"] for row in csv.DictReader(handle)]
    assert splits.count("train") == 88 and splits.count("validation") == 22
    assert protocol["reproducibility"]["seed"] == 42
    score = competition_score(0.8, 0.6, 140.0)
    assert score["main_score"] == 0.7 and score["task2_rmse_bonus_score"] == 5.0 and score["competition_total_score"] == 5.7
    state = seed_everything(42, deterministic=True)
    print("PASS: v1.0 protocol; frozen unified preprocessing; 88/22 subject split; seed=42;", state)


if __name__ == "__main__":
    main()