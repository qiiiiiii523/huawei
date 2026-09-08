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
    with (ROOT / "configs" / "context_fusion_protocol.yaml").open(encoding="utf-8") as handle:
        context_protocol = yaml.safe_load(handle)

    assert spec["ecg"]["model_internal_unit"] == "μV" and common["signal"]["ecg_unit"] == "μV"
    assert spec["ecg"]["model_internal_sampling_rate_hz"] == 500 and common["signal"]["ecg_sampling_rate_hz"] == 500
    assert spec["windowing"] == {"length_sec": 10, "step_sec": 10}
    assert common["signal"]["window_seconds"] == 10 and common["signal"]["window_step_seconds"] == 10 and common["signal"]["window_samples"] == 5000
    assert spec["ecg"]["twelve_lead_order"] == ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert spec["ppg"]["raw_sampling_rate_hz"] == 100 and spec["ppg"]["input_to_model_v1"] is False
    assert common["signal"]["acceleration_sampling_rate_hz"] == 100 and common["runtime"]["use_acceleration"] is False
    assert quality["training_eligibility"]["under_30_sec"] == "exclude_from_training"
    assert common["paths"]["training_protocol_config"] == "configs/training_protocol_v1.yaml"

    assert preprocessing["protocol_status"] == "frozen_joint_anchor"
    assert preprocessing["raw_data_mutation"] is False
    assert preprocessing["target_baseline_policy"] == "train_label_only_predict_at_inference"
    assert preprocessing["output_contract"]["prohibited_at_inference"] == "using_true_target_baseline_uV"
    assert protocol["two_stage_protocol"]["joint_anchor_adaptation"]["context_target_relation"] == "same_subject_cross_time"
    assert protocol["two_stage_protocol"]["joint_anchor_adaptation"]["anchor_target_relation"] == "same_record_same_window"
    assert protocol["validation"]["official_view"] == "raw_uV"
    assert protocol["two_stage_protocol"]["joint_anchor_adaptation"]["initialization"] == "required_P0_strict_pretrained_weights"
    assert protocol["two_stage_protocol"]["joint_anchor_adaptation"]["experiments"] == ["P1-C1", "P1-C2", "P1-C3"]
    assert protocol["validation"]["required_metrics"] == ["r_raw_12", "r_submit_12", "r_missing11"]
    for task in ("task1", "task2"):
        with (ROOT / "configs" / "experiments" / f"{task}_joint_anchor.yaml").open(encoding="utf-8") as handle:
            experiment = yaml.safe_load(handle)
        assert "P1_context_conditioned" not in experiment["experiments"]
        assert {"P1-C1", "P1-C2", "P1-C3"}.issubset(experiment["experiments"])
        assert experiment["experiments"]["P0_anchor_only"]["training_output_i_replacement"] == "forbidden"

    with (ROOT / "metadata" / "subject_split.csv").open(encoding="utf-8-sig", newline="") as handle:
        splits = [row["split"] for row in csv.DictReader(handle)]
    assert splits.count("train") == 88 and splits.count("validation") == 22
    assert protocol["reproducibility"]["seed"] == 42
    assert protocol["context_fusion_protocol"]["config"] == "configs/context_fusion_protocol.yaml"
    assert protocol["context_fusion_protocol"]["stages"] == ["P0_anchor_only", "P1-C1", "P1-C2", "P1-C3"]
    assert context_protocol["protocol_status"] == "frozen_public_experiment_standard"
    stages = context_protocol["experiment_stages"]
    assert set(stages) == {"P0_anchor_only", "P1-C1", "P1-C2", "P1-C3"}
    assert stages["P0_anchor_only"]["input"] == "machine I(C)"
    assert stages["P0_anchor_only"]["target"] == "machine d12(C)"
    assert stages["P0_anchor_only"]["context_enabled"] is False
    assert stages["P0_anchor_only"]["initialization"] == "from_scratch_strict_pretraining"
    expected_fusions = {"P1-C1": "film", "P1-C2": "gated_residual", "P1-C3": "film_gated_residual"}
    for name, fusion_mode in expected_fusions.items():
        stage = stages[name]
        assert stage["fusion_mode"] == fusion_mode
        assert stage["initialization"] == "compatible_same_architecture_P0_checkpoint_required"
        compatibility = stage["p0_checkpoint_compatibility"]
        assert compatibility["required"] is True
        assert compatibility["same_network_architecture"] is True
        assert compatibility["structure_configuration_compatible"] is True
        assert set(compatibility["required_metadata"]) == {"architecture_id", "architecture_config_hash"}
    fusion = context_protocol["fusion_definitions"]
    assert fusion["C1"]["gate_enabled"] is False and fusion["C1"]["residual_enabled"] is False
    assert fusion["C2"]["gate_enabled"] is True and fusion["C2"]["residual_enabled"] is True
    assert fusion["C3"]["operation_order"] == ["film", "gated_residual"]
    initialization = fusion["initialization"]
    assert initialization["gate_initial_value"] == 0.05
    assert initialization["gate_logit_bias"] == -2.944439
    assert initialization["gate_mlp_final_weight_zero_init"] is True
    assert initialization["residual_last_layer_weight_zero_init"] is True
    assert initialization["residual_last_layer_bias_zero_init"] is True
    task2_protocol = context_protocol["task_inputs"]["task2"]
    assert task2_protocol["mutually_exclusive_context_source_variants"] is True
    assert task2_protocol["combined_context_variant_declared"] is False
    assert "both" not in yaml.safe_dump(task2_protocol).lower()
    for forbidden_name in ("context_target_pointwise_loss", "cross_time_waveform_hard_alignment", "r_peak_pseudo_pairing", "training_stage_i_replacement"):
        assert context_protocol["forbidden"][forbidden_name] is True
        assert context_protocol["shared_p1_rules"][forbidden_name] == "forbidden"
    assert context_protocol["shared_p1_rules"]["loss"] == "joint_anchor_sync_loss"
    assert context_protocol["shared_p1_rules"]["checkpoint_selection"] == "best_validation_official_raw_uV_v0"
    for task_name in ("task1", "task2"):
        with (ROOT / "configs" / "experiments" / f"{task_name}_joint_anchor.yaml").open(encoding="utf-8") as handle:
            experiment = yaml.safe_load(handle)
        for stage_name, fusion_mode in expected_fusions.items():
            stage = experiment["experiments"][stage_name]
            assert stage["protocol_stage"] == stage_name
            assert stage["fusion_mode"] == fusion_mode
            assert stage["initialization"] == "compatible_same_architecture_P0_checkpoint_required"
        if task_name == "task2":
            assert experiment["joint_stage"]["mutually_exclusive_context_source_variants"] is True
            assert "both" not in yaml.safe_dump(experiment).lower()
    score = competition_score(0.8, 0.6, 140.0)
    assert score["main_score"] == 0.7 and score["task2_rmse_bonus_score"] == 5.0 and score["competition_total_score"] == 5.7
    state = seed_everything(42, deterministic=True)
    print("PASS: v2 joint-anchor protocol; frozen preprocessing; 88/22 subject split; seed=42;", state)


if __name__ == "__main__":
    main()
