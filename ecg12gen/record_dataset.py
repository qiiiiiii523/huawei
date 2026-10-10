"""Independent record-context Dataset; target ordinal never selects a context window."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .contracts import ContractError, D12_LEADS
from .device_qc import load_device_interpretation_qc, d12_target_mask, d6_input_mask
from .record_context import ContextWindows, RecordContextSample, CONTEXT_CHANNELS


class JointAnchorDataset:
    """One target window plus the full context record, transformed by the collator."""
    def __init__(self, config, task_id: str, split: str, body_scale_variant="A_raw_window",
                 context_channel_indices=None):
        from .dataset import ECGDataConfig, _read_csv
        self.config = config if isinstance(config, ECGDataConfig) else ECGDataConfig.from_yaml(config)
        if self.config.signal["ecg_sampling_rate_hz"] != 500 or self.config.signal["window_samples"] != 5000 or tuple(self.config.signal["twelve_lead_order"]) != D12_LEADS:
            raise ContractError("Independent caches require 500 Hz, 5000-point windows and canonical d12 order")
        if task_id not in {"task1", "task2"} or split not in {"train", "validation"}:
            raise ContractError("Invalid task/split")
        if body_scale_variant != "A_raw_window" or context_channel_indices is not None:
            raise ContractError("Independent context requires canonical channels; variants need separate caches")
        self.task_id, self.split = task_id, split
        directory = self.config.path(f"{task_id}_output")
        context_csv = directory / f"{task_id}_context_window_metadata.csv"
        if not context_csv.is_file():
            raise ContractError(f"{directory}: rebuild independent-record-context-v1 cache; legacy caches have discarded tails/targets")
        self._targets = np.load(directory / f"{task_id}_{split}_target.npy", mmap_mode="r")
        self._contexts = np.load(directory / f"{task_id}_{split}_context.npy", mmap_mode="r")
        self._masks = np.load(directory / f"{task_id}_{split}_context_valid_mask.npy", mmap_mode="r")
        c = 1 if task_id == "task1" else 6
        if self._targets.ndim != 3 or self._targets.shape[1:] != (12, 5000):
            raise ContractError("Invalid target array")
        if self._contexts.ndim != 3 or self._contexts.shape[1:] != (c, 5000) or self._masks.shape != self._contexts.shape:
            raise ContractError("Context and validity arrays must be [K,C,5000]")
        self._rows = sorted([r for r in _read_csv(directory / f"{task_id}_window_metadata.csv") if r["split"] == split], key=lambda r: int(r["array_index"]))
        context_rows = sorted([r for r in _read_csv(context_csv) if r["split"] == split], key=lambda r: int(r["array_index"]))
        for rows, array in ((self._rows, self._targets), (context_rows, self._contexts)):
            if [int(r["array_index"]) for r in rows] != list(range(len(array))):
                raise ContractError("Independent metadata/array indices disagree")
        self._subject_split = {r["subject_id"]: r["split"] for r in _read_csv(self.config.path("subject_split_csv"))}
        self._pairs = {r["pair_id"]: r for r in _read_csv(self.config.path(f"{task_id}_pair_manifest_csv"))}
        self._qc = load_device_interpretation_qc(self.config.path("device_interpretation_qc_csv"))
        self._context_groups = defaultdict(list)
        for r in context_rows:
            if self._subject_split.get(r["subject_id"]) != split:
                raise ContractError("Context crosses subject split")
            self._context_groups[(r["input_type"], r["input_record_id"])].append(r)
        grouped = defaultdict(list)
        for r in self._rows:
            grouped[r["pair_id"]].append(r)
        for rows in grouped.values():
            if len(rows) != 12 or sorted(int(r["start_sample_500hz"]) for r in rows) != list(range(0, 60000, 5000)) or any(int(r["expected_window_count"]) != 12 for r in rows):
                raise ContractError("Every target pair requires a complete real 120-second interval")
            for field in ("subject_id", "target_record_id", "input_record_id", "input_type"):
                if len({r[field] for r in rows}) != 1:
                    raise ContractError("Target pair mixes identities")
        self._prepared_context = {}
        self._indices = [i for i, r in enumerate(self._rows) if self._eligible(r)]

    def _eligible(self, row):
        if self._subject_split.get(row["subject_id"]) != self.split:
            raise ContractError("Target crosses patient split")
        pair = self._pairs.get(row["pair_id"])
        if not pair or any(pair.get(f) != row[f] for f in ("subject_id", "target_record_id", "split")) or pair.get("input_record_id", "") != row["input_record_id"]:
            raise ContractError("Metadata disagrees with authoritative pairing")
        if pair.get("input_type") and pair["input_type"] != row["input_type"]:
            raise ContractError("Context device type disagrees with authoritative pairing")
        if pair.get("pair_status") != "paired":
            return False
        source = row["input_type"]
        if source not in CONTEXT_CHANNELS or (self.task_id == "task1") != (source == "watch_ecg"):
            raise ContractError("Context source disagrees with task")
        qc = self._qc.get(row["target_record_id"])
        if not qc or qc["device_type"] != "ecg_machine_d12":
            raise ContractError("Missing target XML QC")
        if row["quality_status"] != "usable" or pair.get("target_quality_status") != "usable":
            return False
        return not (self.split == "train" and qc["d12_direct_supervision_eligible"] != "true")

    def _context(self, row):
        from .dataset import _baseline_from_text
        key = row["input_type"], row["input_record_id"]
        if key in self._prepared_context:
            return self._prepared_context[key]
        c = CONTEXT_CHANNELS[key[0]]
        rows = sorted(self._context_groups.get(key, []), key=lambda r: int(r["start_sample_500hz"]))
        if not rows:
            result = ContextWindows(np.empty((0, c, 5000), np.float32), np.empty((0, c, 5000), bool), np.empty(0, bool), np.empty(0, np.int64), np.empty(0, np.int64), np.zeros(c, np.float32))
        else:
            if any(r["subject_id"] != row["subject_id"] for r in rows):
                raise ContractError("Context belongs to a different patient")
            starts = np.asarray([int(r["start_sample_500hz"]) for r in rows])
            lengths = np.asarray([int(r["valid_length"]) for r in rows])
            if not np.array_equal(starts, np.arange(len(rows)) * 5000) or any(int(r["expected_context_window_count"]) != len(rows) for r in rows):
                raise ContractError("Context has missing/duplicate windows")
            if np.any(lengths < 1) or np.any(lengths > 5000) or np.any(lengths[:-1] != 5000):
                raise ContractError("Only the last context window can be partial")
            indices = [int(r["array_index"]) for r in rows]
            baseline = _baseline_from_text(rows[0]["input_record_baseline_uV"], c, field="context baseline")
            if any(not np.array_equal(baseline, _baseline_from_text(r["input_record_baseline_uV"], c, field="context baseline")) for r in rows):
                raise ContractError("Context has inconsistent record medians")
            masks = np.asarray(self._masks[indices], dtype=bool).copy()
            if any(masks[i, :, n:].any() for i, n in enumerate(lengths)):
                raise ContractError("Context padding marked valid")
            result = ContextWindows(np.asarray(self._contexts[indices]), masks, masks.any(axis=(1, 2)), lengths, starts, baseline)
        self._prepared_context[key] = result
        return result

    def __len__(self):
        return len(self._indices)

    def close(self):
        """Release read-only mmap handles, especially before deleting test caches on Windows."""
        for array in (self._targets, self._contexts, self._masks):
            array._mmap.close()

    @property
    def excluded_rows(self):
        return len(self._rows) - len(self._indices)

    def __getitem__(self, index):
        row = self._rows[self._indices[index]]
        context, source = self._context(row), row["input_type"]
        mask = np.ones(CONTEXT_CHANNELS[source], bool)
        if self._pairs[row["pair_id"]].get("input_quality_status") != "usable":
            mask[:] = False
        if source == "ecg_machine_d6" and row["input_record_id"]:
            qc = self._qc.get(row["input_record_id"])
            if not qc or qc["device_type"] != "ecg_machine_d6":
                raise ContractError("Missing machine d6 XML QC")
            mask &= d6_input_mask(qc)
            if qc["d6_context_training_eligible"] != "true":
                mask[:] = False
        raw = np.asarray(self._targets[int(row["array_index"])], dtype=np.float32)
        item = RecordContextSample(context, source, raw[:1].copy(), raw, d12_target_mask(self._qc[row["target_record_id"]]), mask,
            self.task_id, self.split, row["subject_id"], row["pair_id"], row["target_record_id"], row["input_record_id"], row["window_id"],
            {"start_sample_500hz": int(row["start_sample_500hz"]), "expected_window_count": 12, "context_target_sync": False,
             "context_relation": "independent_record_condition", "input_type": source, "context_record_id": row["input_record_id"]})
        item.validate()
        return item
