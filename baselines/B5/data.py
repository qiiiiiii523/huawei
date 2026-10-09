"""Read-only Huawei adapters and training-only scale preparation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ecg12gen.contracts import SupervisionMode
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from .config import resolve_path
from .metadata import DemographicsTable, read_csv, subject_key


def common_config(config: dict[str, Any]) -> ECGDataConfig:
    common = ECGDataConfig.from_yaml(resolve_path(config, "common_config"))
    # Allows read-only audits in a separate checkout while preserving the actual cache root.
    common.raw["paths"]["data_root"] = str(resolve_path(config, "huawei_data_root"))
    return common


def preprocessing_config(config: dict[str, Any]) -> PreprocessingConfig:
    return PreprocessingConfig.from_yaml(resolve_path(config, "preprocessing_config"))


def load_preprocessor(config: dict[str, Any]) -> ECGPreprocessor:
    return ECGPreprocessor.load(preprocessing_config(config), resolve_path(config, "scales"))


def fit_huawei_scales(config: dict[str, Any]) -> tuple[ECGPreprocessor, dict[str, Any]]:
    common = common_config(config)
    strict = StrictD12PretrainDataset(common, SupervisionMode.D12_I_PRETRAIN.value)
    split = {subject_key(row["subject_id"]): row["split"] for row in read_csv(common.path("subject_split_csv"))}
    ranges = []
    for row in strict.rows:
        if split.get(subject_key(row["subject_id"])) != "train" or row["source_split"] != "train":
            raise ValueError("Strict scale index contains a non-training subject")
        raw = np.asarray(strict.targets[row["source_task_id"]][int(row["source_array_index"])])
        if raw.shape != (12, 5000) or not np.isfinite(raw).all():
            raise ValueError("Invalid training waveform while fitting scale")
        ranges.append(np.percentile(raw, 95, axis=-1) - np.percentile(raw, 5, axis=-1))
    pc = preprocessing_config(config)
    # Streaming window ranges are exactly the public fit formula without a 938-window copy.
    scale = np.maximum(np.median(np.stack(ranges), axis=0), pc.minimum_scale_uV).astype(np.float32)
    preprocessor = ECGPreprocessor(pc, {"d12": scale, "ecg_machine_i": scale[:1].copy()})
    digest = hashlib.sha256("\n".join(row["dedup_key"] for row in strict.rows).encode()).hexdigest()
    return preprocessor, {"train_windows": len(strict), "train_subjects": len({r['subject_id'] for r in strict.rows}),
                          "train_manifest_sha256": digest, "scale_fit": "Huawei strict train only"}


class HuaweiTrainDataset:
    def __init__(self, config: dict[str, Any], preprocessor: ECGPreprocessor,
                 demographics: DemographicsTable) -> None:
        common = common_config(config)
        self.base = StrictD12PretrainDataset(common, SupervisionMode.D12_I_PRETRAIN.value)
        self.preprocessor, self.demographics = preprocessor, demographics
        self.rows = self.base.rows
        splits = {subject_key(r["subject_id"]): r["split"] for r in read_csv(common.path("subject_split_csv"))}
        if any(splits.get(subject_key(row["subject_id"])) != "train" for row in self.rows):
            raise ValueError("Huawei train index crosses subject split")

    @property
    def manifest_digest(self) -> str:
        return hashlib.sha256("\n".join(r["dedup_key"] for r in self.rows).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, row = self.base[index], self.rows[index]
        raw = sample.Y_12lead
        quality = np.asarray(sample.target_quality_mask, dtype=bool)
        if not quality[0] or not quality[1:].any():
            raise ValueError("Training requires a usable I anchor and missing target leads")
        start = row["start_sample_500hz"]
        return {"anchor": self.preprocessor.transform_window(raw[:1], "ecg_machine_i").model_signal,
                "target": self.preprocessor.transform_d12_target(raw).model_signal,
                "quality_mask": quality.copy(), **self.demographics.get(row["subject_id"]),
                "key": row["target_record_id"] + ":" + start,
                "evaluation_metadata": {"target_record_id": row["target_record_id"], "start_sample_500hz": start,
                                        "subject_id": row["subject_id"]}}


class HuaweiValidationDataset:
    """Retain public main's complete validation-pair contract; context is not a B5-U input."""
    def __init__(self, config: dict[str, Any], task_id: str, preprocessor: ECGPreprocessor,
                 demographics: DemographicsTable) -> None:
        self.base = JointAnchorDataset(common_config(config), task_id, "validation")
        self.preprocessor, self.demographics, self.task_id = preprocessor, demographics, task_id
        self.rows = [self.base._rows[i] for i in self.base._indices]
        if not len(self.base):
            raise ValueError(f"Empty {task_id} validation dataset")

    @property
    def manifest_digest(self) -> str:
        value = "\n".join(f"{r['pair_id']}|{r['target_record_id']}|{r['start_sample_500hz']}" for r in self.rows)
        return hashlib.sha256(value.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, row = self.base[index], self.rows[index]
        metadata = {key: row[key] for key in ("pair_id", "target_record_id", "subject_id", "start_sample_500hz")}
        if row.get("expected_window_count"):
            metadata["expected_window_count"] = row["expected_window_count"]
        return {"anchor": self.preprocessor.transform_window(sample.anchor_i_ecg, "ecg_machine_i").model_signal,
                "target_uV": sample.Y_12lead.copy(), **self.demographics.get(sample.subject_id),
                "key": sample.target_record_id + ":" + row["start_sample_500hz"], "evaluation_metadata": metadata}


def demographics_table(config: dict[str, Any]) -> DemographicsTable:
    return DemographicsTable(resolve_path(config, "demographics"))
