from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.dataset import ECGDataConfig, UnifiedECGDataset
from ecg12gen.models.b0_ridge import StreamingRidge
from ecg12gen.preprocessing import (
    ECGPreprocessor,
    PreprocessingConfig,
)


def load_b0_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("B0 config must be a YAML mapping")

    return config


def window_range_uV(window: np.ndarray) -> np.ndarray:
    """计算每个导联在一个窗口内的P95-P5。"""
    p5 = np.percentile(window, 5, axis=1)
    p95 = np.percentile(window, 95, axis=1)
    return p95 - p5


def fit_preprocessor(
    data_config: ECGDataConfig,
    preprocessing_config: PreprocessingConfig,
) -> ECGPreprocessor:
    """
    仅使用训练集拟合各设备、各导联的固定尺度。
    """
    ranges_by_source: dict[str, list[np.ndarray]] = {
        source: []
        for source in preprocessing_config.expected_leads
    }

    # 严格去重d12训练数据
    strict_dataset = StrictD12PretrainDataset(
        data_config,
        mode="d12_i_pretrain",
    )

    print(f"Fitting d12 scale from {len(strict_dataset)} strict windows...")

    for sample in strict_dataset:
        ranges_by_source["d12"].append(
            window_range_uV(sample.Y_12lead)
        )

    # 手表训练集
    task1_dataset = UnifiedECGDataset(
        data_config,
        task_id="task1",
        split="train",
    )

    print(f"Fitting watch scale from {len(task1_dataset)} windows...")

    for sample in task1_dataset:
        ranges_by_source["watch_ecg"].append(
            window_range_uV(sample.X_ecg)
        )

    # Task2按设备分别拟合
    task2_dataset = UnifiedECGDataset(
        data_config,
        task_id="task2",
        split="train",
    )

    print(f"Fitting task2 device scales from {len(task2_dataset)} windows...")

    for sample in task2_dataset:
        source_type = str(sample.meta["device_type"])

        if source_type not in {
            "ecg_machine_d6",
            "body_scale_d6",
        }:
            raise ValueError(
                f"Unknown task2 device type: {source_type!r}"
            )

        ranges_by_source[source_type].append(
            window_range_uV(sample.X_ecg)
        )

    scales: dict[str, np.ndarray] = {}

    for source_type, rows in ranges_by_source.items():
        if not rows:
            raise ValueError(
                f"No training windows found for source {source_type!r}"
            )

        ranges = np.stack(rows, axis=0)

        scale = np.median(ranges, axis=0)
        scale = np.maximum(
            scale,
            preprocessing_config.minimum_scale_uV,
        ).astype(np.float32)

        expected = preprocessing_config.expected_leads[source_type]

        if scale.shape != (expected,):
            raise ValueError(
                f"{source_type} scale shape is {scale.shape}, "
                f"expected {(expected,)}"
            )

        scales[source_type] = scale

        print(
            f"{source_type}: windows={len(rows)}, "
            f"scale_uV={scale.tolist()}"
        )

    return ECGPreprocessor(
        config=preprocessing_config,
        scale_uV_by_source=scales,
    )


def save_preprocessing_scales(
    preprocessor: ECGPreprocessor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        **preprocessor.scale_uV_by_source,
    )


def train_one_task(
    data_config: ECGDataConfig,
    preprocessor: ECGPreprocessor,
    mode: str,
    input_channels: int,
    alpha: float,
    fit_intercept: bool,
    chunk_windows: int,
    output_path: Path,
) -> dict:
    dataset = StrictD12PretrainDataset(
        data_config,
        mode=mode,
    )

    model = StreamingRidge(
        input_channels=input_channels,
        output_channels=12,
        alpha=alpha,
        fit_intercept=fit_intercept,
    )

    print(
        f"\nTraining {mode}: "
        f"windows={len(dataset)}, "
        f"input_channels={input_channels}"
    )

    for start in range(0, len(dataset), chunk_windows):
        end = min(start + chunk_windows, len(dataset))

        raw_targets = np.stack(
            [
                dataset[index].Y_12lead
                for index in range(start, end)
            ],
            axis=0,
        )

        # 对完整d12统一进行中心化和训练集固定尺度归一化
        target_model_view, _, _ = preprocessor.transform_batch(
            raw_targets,
            source_type="d12",
        )

        # 严格同步输入直接取归一化d12的前1或前6个导联
        input_model_view = target_model_view[:, :input_channels, :]

        model.partial_fit(
            input_model_view,
            target_model_view,
        )

        print(
            f"\rProcessed {end}/{len(dataset)} windows",
            end="",
            flush=True,
        )

    print()

    model.finalize()
    model.save(output_path)

    print(f"Saved model: {output_path}")

    return {
        "mode": mode,
        "input_channels": input_channels,
        "output_channels": 12,
        "alpha": alpha,
        "fit_intercept": fit_intercept,
        "strict_windows": len(dataset),
        "training_points": model.n_samples_seen_,
        "model_path": str(output_path),
    }


def main() -> None:
    config_path = ROOT / "configs" / "b0_ridge.yaml"
    b0_config = load_b0_config(config_path)

    common_path = ROOT / b0_config["data"]["common_config"]
    data_config = ECGDataConfig.from_yaml(common_path)

    preprocessing_path = data_config.path(
        "preprocessing_config"
    )
    preprocessing_config = PreprocessingConfig.from_yaml(
        preprocessing_path
    )

    output_dir = ROOT / b0_config["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(b0_config["experiment"]["seed"])
    np.random.seed(seed)

    model_config = b0_config["model"]
    alpha = float(model_config["alpha"])
    fit_intercept = bool(model_config["fit_intercept"])
    chunk_windows = int(model_config["chunk_windows"])

    if b0_config["preprocessing"]["baseline_head"]:
        raise ValueError(
            "Standard B0 must not use a baseline head"
        )

    predicted_baseline = float(
        b0_config["preprocessing"]["predicted_baseline_uv"]
    )

    if predicted_baseline != 0.0:
        raise ValueError(
            "Standard B0 predicted baseline must be 0 uV"
        )

    # 训练集拟合并冻结预处理尺度
    preprocessor = fit_preprocessor(
        data_config,
        preprocessing_config,
    )

    scales_path = output_dir / "preprocessing_scales.npz"
    save_preprocessing_scales(
        preprocessor,
        scales_path,
    )

    print(f"Saved preprocessing scales: {scales_path}")

    # Task1：d12-I -> d12
    task1_summary = train_one_task(
        data_config=data_config,
        preprocessor=preprocessor,
        mode=b0_config["tasks"]["task1"]["mode"],
        input_channels=1,
        alpha=alpha,
        fit_intercept=fit_intercept,
        chunk_windows=chunk_windows,
        output_path=output_dir / "b0_task1_ridge.npz",
    )

    # Task2：d12-six -> d12
    task2_summary = train_one_task(
        data_config=data_config,
        preprocessor=preprocessor,
        mode=b0_config["tasks"]["task2"]["mode"],
        input_channels=6,
        alpha=alpha,
        fit_intercept=fit_intercept,
        chunk_windows=chunk_windows,
        output_path=output_dir / "b0_task2_ridge.npz",
    )

    summary = {
        "experiment_id": b0_config["experiment"]["id"],
        "experiment_name": b0_config["experiment"]["name"],
        "seed": seed,
        "predicted_baseline_uv": predicted_baseline,
        "preprocessing_scales": str(scales_path),
        "task1": task1_summary,
        "task2": task2_summary,
    }

    summary_path = output_dir / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved training summary: {summary_path}")
    print("B0 training completed successfully.")


if __name__ == "__main__":
    main()