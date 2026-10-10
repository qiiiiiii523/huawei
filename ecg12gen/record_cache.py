"""Code-only writer for independent raw-uV target/context caches, for either task."""
from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path
import numpy as np

from .contracts import ContractError
from .record_context import CONTEXT_CHANNELS, window_context_record, window_target_record

CACHE_PROTOCOL = "independent-record-context-v1"
TARGET_FIELDS = ["array_index", "pair_id", "subject_id", "split", "input_type", "input_record_id",
                 "target_record_id", "window_id", "window_index", "start_sample_500hz",
                 "end_sample_500hz_exclusive", "target_physical_start_sample_500hz", "expected_window_count",
                 "target_record_baseline_uV", "quality_status", "context_target_sync"]
CONTEXT_FIELDS = ["array_index", "subject_id", "split", "input_type", "input_record_id",
                  "context_window_index", "start_sample_500hz", "valid_length", "valid_sample_count",
                  "context_window_valid", "input_record_baseline_uV", "expected_context_window_count"]


def _vector_text(values: np.ndarray) -> str:
    return "|".join(f"{float(v):.9g}" for v in values)


def _write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class IndependentRecordCacheBuilder:
    """Accept already parsed 500-Hz physical records; never infer pair identity or splits.

    add_pair retains 120 seconds of target regardless of context length.
    Context records are indexed once per source/record/split, independently.
    write requires a new output directory and never rewrites an existing cache.
    """
    def __init__(self, task_id: str) -> None:
        if task_id not in {"task1", "task2"}:
            raise ContractError("Invalid task")
        self.task_id = task_id
        self.targets = {s: [] for s in ("train", "validation")}
        self.contexts = {s: [] for s in ("train", "validation")}
        self.context_masks = {s: [] for s in ("train", "validation")}
        self.target_rows: list[dict] = []
        self.context_rows: list[dict] = []
        self._contexts: dict[tuple, tuple] = {}
        self._target_owners: dict[str, tuple] = {}
        self._target_signatures: dict[tuple, str] = {}
        self._pairs: set[str] = set()

    def add_pair(self, pair: dict[str, str], target_uV: np.ndarray,
                 context_uV: np.ndarray | None, context_valid_mask: np.ndarray | None = None,
                 target_start_sample: int = 0) -> None:
        split, subject, pair_id = pair["split"], pair["subject_id"], pair["pair_id"]
        source = pair.get("input_type") or "watch_ecg"
        context_id, target_id = pair.get("input_record_id", ""), pair["target_record_id"]
        if split not in self.targets or not subject or not target_id or not pair_id or pair_id in self._pairs:
            raise ContractError("Invalid or duplicate pair identity/split")
        if source not in CONTEXT_CHANNELS or (self.task_id == "task1") != (source == "watch_ecg"):
            raise ContractError("Context source disagrees with task")
        if context_uV is not None and not context_id:
            raise ContractError("Visible context needs a physical record ID")
        target = window_target_record(target_uV, target_start_sample)
        owner = (split, subject)
        previous_owner = self._target_owners.setdefault(target_id, owner)
        if previous_owner != owner:
            raise ContractError("Target record crosses subject/split identity")
        target_key = (target_id, target_start_sample)
        digest = hashlib.sha256(target.tobytes()).hexdigest()
        if self._target_signatures.setdefault(target_key, digest) != digest:
            raise ContractError("Same physical target interval has inconsistent raw content")
        context = window_context_record(context_uV, source, context_valid_mask)
        if context_id:
            key = (source, context_id)
            previous = self._contexts.get(key)
            signature = (owner, context)
            if previous is not None:
                if (previous[0] != owner or not np.array_equal(previous[1].raw_uV, context.raw_uV)
                        or not np.array_equal(previous[1].time_mask, context.time_mask)
                        or not np.array_equal(previous[1].valid_lengths, context.valid_lengths)):
                    raise ContractError("Reused context record has inconsistent content or subject/split")
            else:
                self._contexts[key] = signature
                for i in range(len(context.raw_uV)):
                    index = len(self.contexts[split])
                    self.contexts[split].append(context.raw_uV[i])
                    self.context_masks[split].append(context.time_mask[i])
                    self.context_rows.append({"array_index": index, "subject_id": subject, "split": split,
                        "input_type": source, "input_record_id": context_id, "context_window_index": i,
                        "start_sample_500hz": int(context.starts[i]), "valid_length": int(context.valid_lengths[i]),
                        "valid_sample_count": int(context.time_mask[i].sum()),
                        "context_window_valid": str(bool(context.window_mask[i])).lower(),
                        "input_record_baseline_uV": _vector_text(context.baseline_uV),
                        "expected_context_window_count": len(context.raw_uV)})
        baseline = _vector_text(np.median(target, axis=(0, 2)))  # selected-interval audit only; never subtracted
        for i in range(12):
            index = len(self.targets[split])
            self.targets[split].append(target[i])
            self.target_rows.append({"array_index": index, "pair_id": pair_id, "subject_id": subject,
                "split": split, "input_type": source, "input_record_id": context_id,
                "target_record_id": target_id, "window_id": f"{pair_id}_W{i:03d}", "window_index": i,
                "start_sample_500hz": i * 5000, "end_sample_500hz_exclusive": (i + 1) * 5000,
                "target_physical_start_sample_500hz": target_start_sample + i * 5000,
                "expected_window_count": 12, "target_record_baseline_uV": baseline,
                "quality_status": "usable", "context_target_sync": "false"})
        self._pairs.add(pair_id)

    def write(self, output_dir: str | Path) -> dict:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        channels = 1 if self.task_id == "task1" else 6
        for split in self.targets:
            for suffix, values, width, dtype in (
                ("target", self.targets[split], 12, np.float32),
                ("context", self.contexts[split], channels, np.float32),
                ("context_valid_mask", self.context_masks[split], channels, bool)):
                array = np.stack(values).astype(dtype) if values else np.empty((0, width, 5000), dtype=dtype)
                np.save(output / f"{self.task_id}_{split}_{suffix}.npy", array)
        _write_rows(output / f"{self.task_id}_window_metadata.csv", self.target_rows, TARGET_FIELDS)
        _write_rows(output / f"{self.task_id}_context_window_metadata.csv", self.context_rows, CONTEXT_FIELDS)
        audit = {"protocol": CACHE_PROTOCOL, "task_id": self.task_id, "target_seconds": 120,
                 "target_window_counts": {s: len(v) for s, v in self.targets.items()},
                 "context_window_counts": {s: len(v) for s, v in self.contexts.items()},
                 "raw_data_mutated": False, "context_target_window_alignment": False,
                 "context_tail_retained": True}
        (output / f"{self.task_id}_build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return audit
