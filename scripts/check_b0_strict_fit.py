from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.dataset import ECGDataConfig
from ecg12gen.evaluate import evaluate_predictions, write_report
from ecg12gen.models.b0_ridge import StreamingRidge
from ecg12gen.preprocessing import (
    ECGPreprocessor,
    PreprocessingConfig,
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML: {path}")

    return config


def load_preprocessor(
    config_path: Path,
    scales_path: Path,
) -> ECGPreprocessor:
    preprocessing_config = PreprocessingConfig.from_yaml(
        config_path
    )

    with np.load(scales_path, allow_pickle=False) as data:
        scales = {
            key: np.asarray(data[key], dtype=np.float32).copy()
            for key in data.files
        }

    return ECGPreprocessor(
        config=preprocessing_config,
        scale_uV_by_source=scales,
    )


def select_indices(
    dataset_length: int,
    max_windows: int,
) -> np.ndarray:
    """
    从整个严格同步索引中均匀抽取窗口，避免只检查开头受试者。
    """
    count = min(dataset_length, max_windows)

    return np.linspace(
        0,
        dataset_length - 1,
        num=count,
        dtype=int,
    )


def diagnose_task(
    task_id: str,
    mode: str,
    input_channels: int,
    model_path: Path,
    data_config: ECGDataConfig,
    preprocessor: ECGPreprocessor,
    output_dir: Path,
    max_windows: int,
    batch_size: int,
) -> dict:
    dataset = StrictD12PretrainDataset(
        data_config,
        mode=mode,
    )

    model = StreamingRidge.load(model_path)

    indices = select_indices(
        len(dataset),
        max_windows,
    )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    d12_scale = preprocessor.scale_uV_by_source[
        "d12"
    ][None, :, None]

    print(
        f"\nStrict-fit diagnostic: {task_id}, "
        f"windows={len(indices)}"
    )

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        raw_targets = np.stack(
            [
                dataset[int(index)].Y_12lead
                for index in batch_indices
            ],
            axis=0,
        )

        # 得到训练时使用的中心化、归一化d12
        target_model_view, _, _ = (
            preprocessor.transform_batch(
                raw_targets,
                source_type="d12",
            )
        )

        input_model_view = target_model_view[
            :, :input_channels, :
        ]

        prediction_model_view = model.predict(
            input_model_view
        )

        # 恢复为中心化μV形态，不添加真实target baseline
        prediction_morphology_uV = (
            prediction_model_view * d12_scale
        )

        target_morphology_uV = (
            target_model_view * d12_scale
        )

        predictions.append(
            prediction_morphology_uV.astype(np.float32)
        )
        targets.append(
            target_morphology_uV.astype(np.float32)
        )

    prediction_array = np.concatenate(
        predictions,
        axis=0,
    )

    target_array = np.concatenate(
        targets,
        axis=0,
    )

    overall, lead_details = evaluate_predictions(
        prediction_array,
        target_array,
        task_id=task_id,
    )
    overall["split"] = "train_strict_fit_diagnostic"
    overall["evaluation_view"] = (
        "centered_strict_training_fit_not_official"
    )

    task_output_dir = output_dir / task_id

    write_report(
        task_output_dir,
        overall,
        lead_details,
        title=(
            f"B0 {task_id} strict synchronized "
            f"d12 internal-fit diagnostic"
        ),
    )

    observed_details = lead_details[:input_channels]

    observed_mean_r = float(
        np.nanmean(
            [
                float(row["pearson_r"])
                for row in observed_details
            ]
        )
    )

    observed_mean_rmse = float(
        np.mean(
            [
                float(row["rmse_uV"])
                for row in observed_details
            ]
        )
    )

    print(f"{task_id} per-lead results:")

    for row in lead_details:
        print(
            f"  {row['lead']:>3s}: "
            f"r={float(row['pearson_r']):.6f}, "
            f"RMSE={float(row['rmse_uV']):.6f} uV"
        )

    print(
        f"{task_id} observed-lead mean r: "
        f"{observed_mean_r:.6f}"
    )

    print(
        f"{task_id} observed-lead mean RMSE: "
        f"{observed_mean_rmse:.6f} uV"
    )

    return {
        "task_id": task_id,
        "mode": mode,
        "diagnostic_windows": int(len(indices)),
        "observed_leads": input_channels,
        "observed_lead_mean_r": observed_mean_r,
        "observed_lead_mean_rmse_uV": observed_mean_rmse,
        "twelve_lead_mean_r": float(
            overall["twelve_lead_mean_pearson_r"]
        ),
        "twelve_lead_mean_rmse_uV": float(
            overall["twelve_lead_mean_rmse_uV"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check B0 fitting on strictly synchronized d12 data."
        )
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    if args.max_windows <= 0:
        raise ValueError("--max-windows must be positive")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    b0_config = load_yaml(
        ROOT / "configs" / "b0_ridge.yaml"
    )

    common_path = (
        ROOT
        / b0_config["data"]["common_config"]
    )

    data_config = ECGDataConfig.from_yaml(
        common_path
    )

    output_root = (
        ROOT
        / b0_config["output"]["directory"]
    )

    preprocessor = load_preprocessor(
        data_config.path("preprocessing_config"),
        output_root / "preprocessing_scales.npz",
    )

    diagnostic_output = (
        output_root
        / "strict_fit_diagnostic"
    )

    task1_summary = diagnose_task(
        task_id="task1",
        mode="d12_i_pretrain",
        input_channels=1,
        model_path=output_root / "b0_task1_ridge.npz",
        data_config=data_config,
        preprocessor=preprocessor,
        output_dir=diagnostic_output,
        max_windows=args.max_windows,
        batch_size=args.batch_size,
    )

    task2_summary = diagnose_task(
        task_id="task2",
        mode="d12_six_pretrain",
        input_channels=6,
        model_path=output_root / "b0_task2_ridge.npz",
        data_config=data_config,
        preprocessor=preprocessor,
        output_dir=diagnostic_output,
        max_windows=args.max_windows,
        batch_size=args.batch_size,
    )

    summary = {
        "purpose": (
            "training-fit audit only; "
            "not a validation/generalization score"
        ),
        "task1": task1_summary,
        "task2": task2_summary,
    }

    summary_path = (
        diagnostic_output
        / "strict_fit_summary.json"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()