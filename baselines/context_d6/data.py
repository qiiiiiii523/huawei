"""Machine-d6-only Task-2 datasets with identical targets for control/context groups."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from baselines.B5.data import common_config, demographics_table
from ecg12gen.dataset import JointAnchorDataset
from ecg12gen.record_context import collate_record_context

from . import CONTEXT_SOURCE


class MachineD6Dataset:
    def __init__(self, config: dict[str, Any], split: str, preprocessor: Any) -> None:
        self.base = JointAnchorDataset(common_config(config), "task2", split)
        self.indices = [i for i in range(len(self.base))
                        if self.base[i].context_source_type == CONTEXT_SOURCE]
        if not self.indices:
            raise ValueError(f"No {CONTEXT_SOURCE} samples in {split}")
        self.preprocessor = preprocessor
        self.demographics = demographics_table(config)
        self.split = split

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.base[self.indices[index]]

    @property
    def manifest_digest(self) -> str:
        rows = [self.base._rows[self.base._indices[i]] for i in self.indices]
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


class MachineD6Collator:
    def __init__(self, preprocessor: Any, demographics: Any, *, context_enabled: bool,
                 context_dropout: float = 0.0, seed: int = 42) -> None:
        self.preprocessor = preprocessor
        self.demographics = demographics
        self.context_enabled = context_enabled
        self.context_dropout = context_dropout
        self.rng = np.random.default_rng(seed)

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        dropout = self.context_dropout if self.context_enabled and samples[0].split == "train" else 0.0
        batch = collate_record_context(samples, self.preprocessor, dropout, self.rng if dropout else None)
        if not self.context_enabled:
            batch["context"].fill(0)
            batch["context_time_mask"].fill(False)
            batch["context_window_mask"].fill(False)
            batch["context_window_weights"].fill(0)
            batch["context_available"].fill(False)
        demographics = [self.demographics.get(sample.subject_id) for sample in samples]
        for field in ("numeric", "sex", "field_mask", "age_topcoded"):
            batch[field] = np.stack([np.asarray(row[field]) for row in demographics])
        batch["anchor"] = batch.pop("anchor_i")
        batch["quality_mask"] = batch.pop("target_quality_mask")
        batch["keys"] = [sample.target_record_id + ":" + str(sample.meta["start_sample_500hz"])
                         for sample in samples]
        for row in batch["evaluation_metadata"]:
            row["context_used"] = bool(self.context_enabled)
        return batch

