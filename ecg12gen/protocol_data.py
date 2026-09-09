"""Shared main-protocol adapters used by architecture baselines."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from .contracts import SupervisionMode
from .dataset import ECGDataConfig, JointAnchorDataset
from .d12_pretrain import StrictD12PretrainDataset
from .preprocessing import ECGPreprocessor, PreprocessingConfig

def _source(task_id: str, source: str) -> str:
    if task_id == "task1" and source == "watch_ecg": return source
    if task_id == "task2" and source in {"ecg_machine_d6", "body_scale_d6"}: return source
    raise ValueError(f"invalid context source {task_id=} {source=}")

def fit_preprocessor(config: ECGDataConfig | str | Path, task_id: str,
                     body_scale_variant: str = "A_raw_window",
                     context_source_type: str | None = None) -> ECGPreprocessor:
    dc = config if isinstance(config, ECGDataConfig) else ECGDataConfig.from_yaml(config)
    pc = PreprocessingConfig.from_yaml(dc.path("preprocessing_config"))
    strict = StrictD12PretrainDataset(dc, SupervisionMode.D12_I_PRETRAIN.value)
    joint = JointAnchorDataset(dc, task_id, "train", body_scale_variant)
    signals: dict[str, list[np.ndarray]] = {"d12": []}
    for i in range(len(strict)): signals["d12"].append(np.asarray(strict[i].Y_12lead, np.float32))
    for i in range(len(joint)):
        s = joint[i]
        src = _source(task_id, s.context_source_type)
        if context_source_type is not None and src != context_source_type: continue
        signals.setdefault(src, []).append(np.asarray(s.context_ecg, np.float32))
    return ECGPreprocessor.fit(pc, {k: np.stack(v) for k, v in signals.items() if v})

def _context(pre: ECGPreprocessor, raw: np.ndarray, source: str, mask: np.ndarray | None) -> np.ndarray:
    if raw.shape[0] == pre.config.expected_leads[source]: return pre.transform_window(raw, source).model_signal.copy()
    raise ValueError(f"unexpected context shape {raw.shape} for {source}")

class StrictDataset(Dataset):
    def __init__(self, config: ECGDataConfig | str | Path, pre: ECGPreprocessor):
        self.src, self.pre = StrictD12PretrainDataset(config, SupervisionMode.D12_I_PRETRAIN.value), pre
    def __len__(self): return len(self.src)
    def __getitem__(self, i):
        s = self.src[i]; y = self.pre.transform_d12_target(s.Y_12lead).model_signal
        a = self.pre.transform_window(s.Y_12lead[:1], "ecg_machine_i").model_signal
        return {"anchor_i": torch.from_numpy(a.copy()), "anchor_baseline": torch.from_numpy((self.pre.transform_window(s.Y_12lead[:1], "ecg_machine_i").baseline_uV / self.pre.scale_uV_by_source["d12"][0]).copy()), "target_baseline": torch.from_numpy((self.pre.transform_d12_target(s.Y_12lead).baseline_uV / self.pre.scale_uV_by_source["d12"]).copy()), "target": torch.from_numpy(y.copy()),
                "anchor_raw": torch.from_numpy(np.asarray(s.Y_12lead[:1], np.float32).copy()),
                "target_raw": torch.from_numpy(np.asarray(s.Y_12lead, np.float32).copy()), "meta": s.meta}

class JointDataset(Dataset):
    def __init__(self, config, task_id, split, pre, body_scale_variant="A_raw_window", source_type=None):
        self.main, self.pre, self.task_id = JointAnchorDataset(config, task_id, split, body_scale_variant), pre, task_id
        self.indices = [i for i in range(len(self.main)) if source_type is None or self.main[i].context_source_type == source_type]
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        s = self.main[self.indices[i]]; src = _source(self.task_id, s.context_source_type)
        c = _context(self.pre, s.context_ecg, src, s.context_lead_mask)
        a = self.pre.transform_window(s.anchor_i_ecg, "ecg_machine_i").model_signal
        y = self.pre.transform_d12_target(s.Y_12lead).model_signal
        return {"anchor_i": torch.from_numpy(a.copy()), "anchor_baseline": torch.from_numpy((self.pre.transform_window(s.anchor_i_ecg, "ecg_machine_i").baseline_uV / self.pre.scale_uV_by_source["d12"][0]).copy()), "target_baseline": torch.from_numpy((self.pre.transform_d12_target(s.Y_12lead).baseline_uV / self.pre.scale_uV_by_source["d12"]).copy()), "context": torch.from_numpy(c.copy()),
                "context_source_type": src, "context_lead_mask": torch.from_numpy(s.context_lead_mask.copy()),
                "target": torch.from_numpy(y.copy()), "anchor_raw": torch.from_numpy(s.anchor_i_ecg.copy()),
                "target_raw": torch.from_numpy(s.Y_12lead.copy()), "subject_id": s.subject_id, "meta": s.meta}

def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("anchor_i", "anchor_baseline", "target_baseline", "target", "anchor_raw", "target_raw")
    out = {k: torch.stack([x[k] for x in batch]) for k in keys}
    if "context" in batch[0]:
        out["context"] = torch.stack([x["context"] for x in batch])
        out["context_lead_mask"] = torch.stack([x["context_lead_mask"] for x in batch])
        out["context_source_type"] = [x["context_source_type"] for x in batch]
        out["subject_id"] = [x["subject_id"] for x in batch]
    out["meta"] = [x["meta"] for x in batch]; return out
