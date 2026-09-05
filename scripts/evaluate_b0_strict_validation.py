from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.dataset import ECGDataConfig
from ecg12gen.evaluate import evaluate_predictions, write_report
from ecg12gen.models.b0_ridge import StreamingRidge
from ecg12gen.preprocessing import (
    ECGPreprocessor,
    PreprocessingConfig,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


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


def build_dedup_key(row: dict[str, str]) -> str:
    start = int(row["start_sample_500hz"])
    end = start + 5000

    return (
        f"{row['target_record_id']}:"
        f"{start}:{end}"
    )


def load_unique_validation_targets(
    data_config: ECGDataConfig,
) -> tuple[
    list[dict],
    dict[str, np.ndarray],
    int,
]:
    arrays: dict[str, np.ndarray] = {}
    references: list[dict] = []
    seen: dict[str, dict] = {}
    duplicate_count = 0

    subject_rows = read_csv(
        data_config.path("subject_split_csv")
    )

    subject_split = {
        row["subject_id"]: row["split"]
        for row in subject_rows
    }

    for task_id in ("task1", "task2"):
        task_dir = data_config.path(
            f"{task_id}_output"
        )

        arrays[task_id] = np.load(
            task_dir
            / f"{task_id}_validation_target.npy",
            mmap_mode="r",
        )

        metadata = [
            row
            for row in read_csv(
                task_dir
                / f"{task_id}_window_metadata.csv"
            )
            if row["split"] == "validation"
        ]

        metadata = sorted(
            metadata,
            key=lambda row: int(row["array_index"]),
        )

        if len(metadata) != len(arrays[task_id]):
            raise ValueError(
                f"{task_id} metadata and target count mismatch"
            )

        for row in metadata:
            if subject_split.get(row["subject_id"]) != "validation":
                raise ValueError(
                    f"Non-validation subject found: "
                    f"{row['subject_id']}"
                )

            array_index = int(row["array_index"])
            dedup_key = build_dedup_key(row)

            reference = {
                "dedup_key": dedup_key,
                "source_task_id": task_id,
                "source_array_index": array_index,
                "subject_id": row["subject_id"],
                "target_record_id": row["target_record_id"],
                "window_id": row["window_id"],
                "start_sample_500hz": int(
                    row["start_sample_500hz"]
                ),
            }

            if dedup_key in seen:
                previous = seen[dedup_key]

                current_target = np.asarray(
                    arrays[task_id][array_index]
                )

                previous_target = np.asarray(
                    arrays[
                        previous["source_task_id"]
                    ][
                        previous["source_array_index"]
                    ]
                )

                if not np.allclose(
                    current_target,
                    previous_target,
                    rtol=0.0,
                    atol=1e-4,
                ):
                    raise ValueError(
                        "Duplicate d12 key has different waveform: "
                        f"{dedup_key}"
                    )

                duplicate_count += 1
                continue

            seen[dedup_key] = reference
            references.append(reference)

    # 检查validation窗口没有进入严格训练索引
    strict_train_rows = read_csv(
        data_config.repository_root
        / "metadata"
        / "d12_strict_pretrain_index.csv"
    )

    train_keys = {
        row["dedup_key"]
        for row in strict_train_rows
    }

    validation_keys = {
        row["dedup_key"]
        for row in references
    }

    overlap = train_keys & validation_keys

    if overlap:
        raise ValueError(
            f"Found {len(overlap)} train/validation d12 overlaps"
        )

    return references, arrays, duplicate_count


def save_manifest(
    path: Path,
    references: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(references[0]),
        )
        writer.writeheader()
        writer.writerows(references)


def evaluate_task(
    task_id: str,
    input_channels: int,
    model: StreamingRidge,
    references: list[dict],
    arrays: dict[str, np.ndarray],
    preprocessor: ECGPreprocessor,
    output_dir: Path,
    batch_size: int,
) -> dict:
    raw_predictions: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    centered_targets: list[np.ndarray] = []

    d12_scale = preprocessor.scale_uV_by_source[
        "d12"
    ][None, :, None]

    for start in range(0, len(references), batch_size):
        batch_refs = references[
            start:start + batch_size
        ]

        target_batch = np.stack(
            [
                arrays[row["source_task_id"]][
                    row["source_array_index"]
                ]
                for row in batch_refs
            ],
            axis=0,
        ).astype(np.float32)

        target_model_view, target_baseline, _ = (
            preprocessor.transform_batch(
                target_batch,
                source_type="d12",
            )
        )

        input_model_view = target_model_view[
            :, :input_channels, :
        ]

        prediction_model_view = model.predict(
            input_model_view
        )

        # d12固定尺度恢复，baseline固定为0
        prediction_uV = (
            prediction_model_view * d12_scale
        ).astype(np.float32)

        target_centered_uV = (
            target_batch
            - target_baseline[:, :, None]
        ).astype(np.float32)

        raw_predictions.append(prediction_uV)
        raw_targets.append(target_batch)
        centered_targets.append(target_centered_uV)

    prediction_array = np.concatenate(
        raw_predictions,
        axis=0,
    )

    target_raw_array = np.concatenate(
        raw_targets,
        axis=0,
    )

    target_centered_array = np.concatenate(
        centered_targets,
        axis=0,
    )

    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        task_dir
        / f"{task_id}_prediction_zero_baseline_uV.npy",
        prediction_array,
    )

    # 与原始target比较：baseline固定为0
    raw_overall, raw_details = evaluate_predictions(
        prediction_array,
        target_raw_array,
        task_id,
    )

    raw_overall["split"] = (
        "validation_subject_strict_d12"
    )
    raw_overall["evaluation_view"] = (
        "strict_d12_raw_target_zero_predicted_baseline"
    )

    write_report(
        task_dir / "raw_zero_baseline",
        raw_overall,
        raw_details,
        title=(
            f"B0 {task_id} held-out strict d12 "
            f"raw target, zero predicted baseline"
        ),
    )

    # 中心化形态诊断
    centered_overall, centered_details = (
        evaluate_predictions(
            prediction_array,
            target_centered_array,
            task_id,
        )
    )

    centered_overall["split"] = (
        "validation_subject_strict_d12"
    )
    centered_overall["evaluation_view"] = (
        "strict_d12_centered_morphology_diagnostic"
    )

    write_report(
        task_dir / "centered_morphology",
        centered_overall,
        centered_details,
        title=(
            f"B0 {task_id} held-out strict d12 "
            f"centered morphology diagnostic"
        ),
    )

    print(f"\n{task_id}:")
    print(
        "  raw mean r: "
        f"{raw_overall['twelve_lead_mean_pearson_r']:.6f}"
    )
    print(
        "  raw mean RMSE: "
        f"{raw_overall['twelve_lead_mean_rmse_uV']:.6f} uV"
    )
    print(
        "  centered mean r: "
        f"{centered_overall['twelve_lead_mean_pearson_r']:.6f}"
    )
    print(
        "  centered mean RMSE: "
        f"{centered_overall['twelve_lead_mean_rmse_uV']:.6f} uV"
    )

    if task_id == "task2":
        print(
            "  centered V1-V6 RMSE: "
            f"{centered_overall['task2_missing_lead_mean_rmse_uV']:.6f} uV"
        )

    return {
        "raw_zero_baseline": raw_overall,
        "centered_morphology": centered_overall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    b0_config = load_yaml(
        ROOT / "configs" / "b0_ridge.yaml"
    )

    data_config = ECGDataConfig.from_yaml(
        ROOT / b0_config["data"]["common_config"]
    )

    b0_output = (
        ROOT
        / b0_config["output"]["directory"]
    )

    diagnostic_output = (
        b0_output
        / "strict_validation_diagnostic"
    )

    preprocessor = load_preprocessor(
        data_config.path("preprocessing_config"),
        b0_output / "preprocessing_scales.npz",
    )

    references, arrays, duplicate_count = (
        load_unique_validation_targets(
            data_config
        )
    )

    print(
        f"Unique validation d12 windows: {len(references)}"
    )
    print(
        f"Removed duplicate validation windows: "
        f"{duplicate_count}"
    )
    print("Train/validation overlap: 0")

    save_manifest(
        diagnostic_output
        / "d12_strict_validation_manifest.csv",
        references,
    )

    task1_result = evaluate_task(
        task_id="task1",
        input_channels=1,
        model=StreamingRidge.load(
            b0_output / "b0_task1_ridge.npz"
        ),
        references=references,
        arrays=arrays,
        preprocessor=preprocessor,
        output_dir=diagnostic_output,
        batch_size=args.batch_size,
    )

    task2_result = evaluate_task(
        task_id="task2",
        input_channels=6,
        model=StreamingRidge.load(
            b0_output / "b0_task2_ridge.npz"
        ),
        references=references,
        arrays=arrays,
        preprocessor=preprocessor,
        output_dir=diagnostic_output,
        batch_size=args.batch_size,
    )

    summary = {
        "purpose": (
            "held-out synchronized d12 diagnostic; "
            "not official cross-device V0"
        ),
        "unique_validation_windows": len(references),
        "removed_duplicates": duplicate_count,
        "train_validation_overlap": 0,
        "task1": task1_result,
        "task2": task2_result,
    }

    summary_path = (
        diagnostic_output
        / "strict_validation_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()