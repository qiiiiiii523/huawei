from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.dataset import ECGDataConfig, UnifiedECGDataset
from ecg12gen.evaluate import (
    competition_score,
    evaluate_centered_diagnostic,
    evaluate_predictions,
    evaluate_task2_diagnostics,
    write_competition_score,
    write_report,
    write_task2_diagnostics,
)
from ecg12gen.models.b0_ridge import StreamingRidge
from ecg12gen.preprocessing import (
    ECGPreprocessor,
    PreprocessingConfig,
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    return config


def load_preprocessor(
    preprocessing_config_path: Path,
    scales_path: Path,
) -> ECGPreprocessor:
    config = PreprocessingConfig.from_yaml(
        preprocessing_config_path
    )

    with np.load(scales_path, allow_pickle=False) as data:
        scales = {
            key: np.asarray(data[key], dtype=np.float32).copy()
            for key in data.files
        }

    missing_sources = (
        set(config.expected_leads)
        - set(scales)
    )

    if missing_sources:
        raise ValueError(
            f"Missing preprocessing scales: {sorted(missing_sources)}"
        )

    return ECGPreprocessor(
        config=config,
        scale_uV_by_source=scales,
    )


def save_manifest(
    path: Path,
    rows: list[dict[str, str | int]],
) -> None:
    if not rows:
        raise ValueError("Prediction manifest is empty")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def run_task(
    task_id: str,
    data_config: ECGDataConfig,
    preprocessor: ECGPreprocessor,
    model: StreamingRidge,
    output_dir: Path,
    predicted_baseline_uV: float,
    write_centered: bool,
) -> dict:
    dataset = UnifiedECGDataset(
        data_config,
        task_id=task_id,
        split="validation",
    )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str | int]] = []

    print(
        f"\nEvaluating {task_id}: "
        f"{len(dataset)} usable validation windows"
    )

    for index, sample in enumerate(dataset):
        source_type = str(sample.meta["device_type"])

        input_model_signal = preprocessor.transform_window(
            sample.X_ecg,
            source_type=source_type,
        ).model_signal

        prediction_model_signal = model.predict(
            input_model_signal[None, :, :]
        )[0]

        # 标准B0：使用d12训练尺度恢复到μV，但预测基线固定为0
        baseline = np.full(
            12,
            predicted_baseline_uV,
            dtype=np.float32,
        )

        prediction_raw_uV = (
            preprocessor.compose_raw_d12_prediction(
                prediction_model_signal,
                predicted_baseline_uV=baseline,
            )
        )

        predictions.append(prediction_raw_uV)
        targets.append(
            np.asarray(sample.Y_12lead, dtype=np.float32)
        )

        metadata_rows.append(
            {
                "subject_id": str(sample.meta["subject_id"]),
                "input_type": source_type,
            }
        )

        manifest_rows.append(
            {
                "prediction_index": index,
                "task_id": task_id,
                "subject_id": str(sample.meta["subject_id"]),
                "window_id": str(sample.meta["window_id"]),
                "pair_id": str(sample.meta["pair_id"]),
                "input_type": source_type,
            }
        )

        if (index + 1) % 50 == 0 or index + 1 == len(dataset):
            print(
                f"\rProcessed {index + 1}/{len(dataset)}",
                end="",
                flush=True,
            )

    print()

    prediction_array = np.stack(
        predictions,
        axis=0,
    ).astype(np.float32)

    target_array = np.stack(
        targets,
        axis=0,
    ).astype(np.float32)

    task_output_dir = output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = (
        task_output_dir
        / f"{task_id}_validation_prediction_raw_uV.npy"
    )

    np.save(
        prediction_path,
        prediction_array,
    )

    manifest_path = (
        task_output_dir
        / f"{task_id}_prediction_manifest.csv"
    )

    save_manifest(
        manifest_path,
        manifest_rows,
    )

    # 官方raw-μV V0
    overall, lead_details = evaluate_predictions(
        prediction_array,
        target_array,
        task_id=task_id,
    )

    report_paths = write_report(
        task_output_dir,
        overall,
        lead_details,
        title=f"B0 {task_id} official raw-uV V0 report",
    )

    # Task2补充分设备与受试者统计
    if task_id == "task2":
        subject_rows, device_rows = evaluate_task2_diagnostics(
            prediction_array,
            target_array,
            metadata_rows,
        )

        write_task2_diagnostics(
            task_output_dir,
            subject_rows,
            device_rows,
            report_paths[-1],
        )

    # 中心化指标只作为诊断，不替代官方结果
    if write_centered:
        centered_overall, centered_details = (
            evaluate_centered_diagnostic(
                prediction_array,
                target_array,
                task_id=task_id,
            )
        )

        centered_dir = (
            task_output_dir
            / "centered_diagnostic"
        )

        write_report(
            centered_dir,
            centered_overall,
            centered_details,
            title=(
                f"B0 {task_id} centered morphology "
                f"diagnostic - not official"
            ),
        )

    print(f"Prediction saved: {prediction_path}")
    print(f"Official report: {report_paths[-1]}")

    return overall


def main() -> None:
    b0_config = load_yaml(
        ROOT / "configs" / "b0_ridge.yaml"
    )

    common_config_path = (
        ROOT
        / b0_config["data"]["common_config"]
    )

    data_config = ECGDataConfig.from_yaml(
        common_config_path
    )

    output_dir = (
        ROOT
        / b0_config["output"]["directory"]
    )

    scales_path = (
        output_dir
        / "preprocessing_scales.npz"
    )

    preprocessing_config_path = data_config.path(
        "preprocessing_config"
    )

    preprocessor = load_preprocessor(
        preprocessing_config_path,
        scales_path,
    )

    task1_model = StreamingRidge.load(
        output_dir / "b0_task1_ridge.npz"
    )

    task2_model = StreamingRidge.load(
        output_dir / "b0_task2_ridge.npz"
    )

    predicted_baseline_uV = float(
        b0_config["preprocessing"][
            "predicted_baseline_uv"
        ]
    )

    if predicted_baseline_uV != 0.0:
        raise ValueError(
            "Standard B0 requires predicted baseline = 0 uV"
        )

    write_centered = bool(
        b0_config["evaluation"].get(
            "centered_metrics_diagnostic_only",
            True,
        )
    )

    task1_metrics = run_task(
        task_id="task1",
        data_config=data_config,
        preprocessor=preprocessor,
        model=task1_model,
        output_dir=output_dir,
        predicted_baseline_uV=predicted_baseline_uV,
        write_centered=write_centered,
    )

    task2_metrics = run_task(
        task_id="task2",
        data_config=data_config,
        preprocessor=preprocessor,
        model=task2_model,
        output_dir=output_dir,
        predicted_baseline_uV=predicted_baseline_uV,
        write_centered=write_centered,
    )

    score = competition_score(
        r1=float(task1_metrics["task1_r1"]),
        r2=float(task2_metrics["task2_r2"]),
        missing_lead_rmse_uV=float(
            task2_metrics[
                "task2_missing_lead_mean_rmse_uV"
            ]
        ),
    )

    write_competition_score(
        output_dir,
        score,
    )

    summary_path = output_dir / "evaluation_summary.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "predicted_baseline_uV": predicted_baseline_uV,
                "task1": task1_metrics,
                "task2": task2_metrics,
                "competition": score,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nB0 evaluation completed.")
    print(f"Task1 r1: {score['task1_r1']:.6f}")
    print(f"Task2 r2: {score['task2_r2']:.6f}")
    print(
        "Task2 V1-V6 RMSE: "
        f"{score['task2_missing_lead_mean_rmse_uV']:.6f} uV"
    )
    print(f"Main score: {score['main_score']:.6f}")
    print(
        "Competition total score: "
        f"{score['competition_total_score']:.6f}"
    )


if __name__ == "__main__":
    main()