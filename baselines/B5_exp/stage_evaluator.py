# Frozen QC-v3 record evaluator; separate from the legacy hashed experiment evaluator.
"""Raw-uV record evaluation. Only r_missing11; no preprocessing or I replacement."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
import numpy as np
from ecg12gen.contracts import D12_LEADS, ContractError

MISSING_11_LEAD_INDICES = np.arange(1, 12)
TASK2_RMSE_LEADS = D12_LEADS[6:12]
TASK2_DEVICES = ("ecg_machine_d6", "body_scale_d6")
TARGET_RECORD_AGGREGATION = "target_record_equal_after_pair_mean_v1"
TASK2_DEVICE_AGGREGATION = "machine_body_equal_weight_target_record_score_v2"
COMPETITION_SCORE_PROTOCOL = "correlation50_bonus10_target_record_v2"
TARGET_LEAD_QC_PROTOCOL = "explicit_target_lead_mask_v1"
QC_TASK2_DEVICE_AGGREGATION = "machine_body_equal_weight_target_record_qc_score_v3"
QC_COMPETITION_SCORE_PROTOCOL = "correlation50_bonus10_target_record_qc_v3"


def _validated_quality_mask(mask: np.ndarray, expected_n: int,
                            metadata_rows: list[dict[str, str]]) -> np.ndarray:
    """One immutable quality decision per physical target, shared by all pairs/windows."""
    quality = np.asarray(mask)
    if quality.shape != (expected_n, 12) or not np.isin(quality, [0, 1]).all():
        raise ContractError("target_quality_mask must be boolean/0-1 [N,12] in canonical lead order")
    if len(metadata_rows) != expected_n:
        raise ContractError("Metadata count disagrees with quality mask")
    by_target = {}
    for index, row in enumerate(metadata_rows):
        target_id = row.get("target_record_id")
        if not target_id:
            raise ContractError("QC evaluation requires target_record_id")
        if target_id in by_target and not np.array_equal(by_target[target_id], quality[index]):
            raise ContractError(f"Inconsistent target quality mask across windows/pairs/devices: {target_id}")
        by_target[target_id] = quality[index]
    return quality.astype(bool, copy=False)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(np.float64, copy=False), y.astype(np.float64, copy=False)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator > 0 else float("nan")


def _record_groups(metadata_rows: list[dict[str, str]], expected_n: int) -> list[list[int]]:
    if len(metadata_rows) != expected_n:
        raise ContractError("Record metadata count does not match prediction rows")
    grouped: dict[str, list[tuple[int, int]]] = {}
    for index, row in enumerate(metadata_rows):
        sample_id = row.get("pair_id") or row.get("target_record_id")
        if not sample_id:
            raise ContractError("Record evaluation requires pair_id or target_record_id")
        try:
            start = int(row["start_sample_500hz"])
        except (KeyError, ValueError, TypeError) as error:
            raise ContractError("Record evaluation requires integer start_sample_500hz") from error
        grouped.setdefault(sample_id, []).append((start, index))
    groups = []
    for sample_id in sorted(grouped):
        ordered = sorted(grouped[sample_id])
        starts = [start for start, _ in ordered]
        if len(starts) != len(set(starts)):
            raise ContractError(f"Duplicate window start within validation sample {sample_id}")
        if starts[0] != 0 or any(b - a != 5000 for a, b in zip(starts, starts[1:])):
            raise ContractError(f"Validation sample {sample_id} is not contiguous: expected starts 0,5000,10000,...")
        if len({metadata_rows[i].get("target_record_id") for _, i in ordered}) > 1:
            raise ContractError(f"Validation sample {sample_id} mixes target records")
        try:
            expected = {int(metadata_rows[i]["expected_window_count"])
                        for _, i in ordered if metadata_rows[i].get("expected_window_count")}
        except (ValueError, TypeError) as error:
            raise ContractError("expected_window_count must be an integer") from error
        if len(expected) > 1 or (expected and len(ordered) != next(iter(expected))):
            raise ContractError(f"Validation sample {sample_id} has an inconsistent or incomplete window count")
        if len(ordered) != 12:
            raise ContractError(f"Validation sample {sample_id} must contain 12 real windows (120 seconds); rebuild old caches")
        groups.append([i for _, i in ordered])
    return groups


def _target_record_groups(groups: list[list[int]], records: list[tuple[np.ndarray, np.ndarray]],
                          metadata_rows: list[dict[str, str]]) -> list[list[int]]:
    """Retain all pairs; each physical target gets one aggregate score."""
    by_target: dict[str, list[int]] = {}
    subjects: dict[str, set[str]] = {}
    for record_index, indices in enumerate(groups):
        row = metadata_rows[indices[0]]
        target_id = row.get("target_record_id")
        if not target_id:
            raise ContractError("Target-equal evaluation requires target_record_id for every pair")
        subject_ids = {metadata_rows[i].get("subject_id", "") for i in indices}
        if len(subject_ids) > 1:
            raise ContractError(f"Validation pair mixes subjects for target {target_id}")
        subjects.setdefault(target_id, set()).update(subject_ids)
        if len(subjects[target_id]) > 1:
            raise ContractError(f"Repeated target {target_id} has inconsistent subject_id")
        other_pairs = by_target.setdefault(target_id, [])
        if other_pairs and not np.array_equal(records[other_pairs[0]][1], records[record_index][1]):
            raise ContractError(f"Repeated target {target_id} has different raw target waveforms")
        other_pairs.append(record_index)
    return [by_target[key] for key in sorted(by_target)]


def _evaluate_record_subset(prediction: np.ndarray, target: np.ndarray, task_id: str,
                                metadata_rows: list[dict[str, str]], *,
                                target_quality_mask: np.ndarray | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Stitch pairs; average pair r within target, then targets and II--V6.

    Average pair squared errors within each target before point-weighted RMSE.
    Task-2 bonus uses V1--V6 only; r1/r2 retain record-level r_missing11.
    Undefined pair r is excluded within target; a fully undefined target r is
    excluded across targets. No prediction averaging or best-pair selection.
    """
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (12, 5000):
        raise ContractError("Record evaluation requires matching raw-uV [N,12,5000] arrays")
    if not len(prediction) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ContractError("Record evaluation requires non-empty finite predictions and targets")
    if task_id not in {"task1", "task2"}:
        raise ContractError("task_id must be task1 or task2")
    quality = None if target_quality_mask is None else _validated_quality_mask(
        target_quality_mask, len(prediction), metadata_rows)
    groups = _record_groups(metadata_rows, len(prediction))
    record_quality = np.ones((len(groups), 12), dtype=bool) if quality is None else quality[[g[0] for g in groups]]
    records = [(np.concatenate([prediction[i] for i in indices], axis=1),
                np.concatenate([target[i] for i in indices], axis=1)) for indices in groups]
    target_groups = _target_record_groups(groups, records, metadata_rows)
    mean_correlations, details = [], []
    invalid_count, invalid_pair_count = 0, 0
    for lead in MISSING_11_LEAD_INDICES:
        eligible_pairs = [i for i in range(len(records)) if record_quality[i, lead]]
        eligible_targets = [indices for indices in target_groups if record_quality[indices[0], lead]]
        if not eligible_targets:
            raise ContractError(f"No QC-eligible target records for {D12_LEADS[lead]}")
        correlations = {i: _pearson(records[i][0][lead], records[i][1][lead]) for i in eligible_pairs}
        invalid_pair_count += sum(not np.isfinite(value) for value in correlations.values())
        target_correlations = []
        for indices in eligible_targets:
            pair_values = np.asarray([correlations[i] for i in indices])
            defined = np.isfinite(pair_values)
            target_correlations.append(float(pair_values[defined].mean()) if defined.any() else float("nan"))
        target_correlations = np.asarray(target_correlations)
        valid = np.isfinite(target_correlations)
        if not valid.any():
            raise ContractError(f"No defined record Pearson r for {D12_LEADS[lead]}")
        invalid_count += int((~valid).sum())
        mean_correlations.append(float(target_correlations[valid].mean()))
        pair_squared_errors = {i: float(np.sum((records[i][0][lead].astype(np.float64) -
                                               records[i][1][lead].astype(np.float64)) ** 2))
                               for i in eligible_pairs}
        squared_error = sum(float(np.mean([pair_squared_errors[i] for i in indices]))
                            for indices in eligible_targets)
        point_count = sum(records[indices[0]][0].shape[1] for indices in eligible_targets)
        detail = {"lead": D12_LEADS[lead], "rmse_uV": float(np.sqrt(squared_error / point_count)),
                  "n_records": len(eligible_pairs), "n_undefined_record_correlations": int((~valid).sum()),
                  "n_unique_target_records": len(eligible_targets), "n_validation_points": point_count}
        if quality is not None:
            detail.update(r=float(target_correlations[valid].mean()),
                          n_defined_target_correlations=int(valid.sum()),
                          n_qc_excluded_pairs=len(records)-len(eligible_pairs),
                          n_qc_excluded_target_records=len(target_groups)-len(eligible_targets))
        details.append(detail)
    rmse = float(np.mean([row["rmse_uV"] for row in details]))
    overall = {
        "split": "validation", "task_id": task_id, "n_windows": len(prediction), "n_records": len(records),
        "n_pairs": len(records), "n_unique_target_records": len(target_groups),
        "target_record_aggregation": TARGET_RECORD_AGGREGATION,
        "evaluation_view": "raw_uV", "prediction_unit": "uV",
        "evaluation_aggregation": "target_macro_after_pair_mean_after_chronological_window_stitch",
        "rmse_aggregation": "pair_mean_squared_error_within_target_then_point_weighted_per_lead_then_lead_mean",
        "evaluation_input_contract": "joint_anchor_test_like", "checkpoint_selection_metric": "r_missing11",
        "scored_leads": ",".join(D12_LEADS[i] for i in MISSING_11_LEAD_INDICES),
        "r_missing11": float(np.mean(mean_correlations)), "missing11_mean_rmse_uV": rmse,
        "n_undefined_record_lead_correlations": invalid_count,
        "n_undefined_pair_lead_correlations": invalid_pair_count,
    }
    if quality is not None:
        overall.update(target_lead_qc_protocol=TARGET_LEAD_QC_PROTOCOL,
                       competition_score_protocol=QC_COMPETITION_SCORE_PROTOCOL,
                       quality_mask_applied=True,
                       n_qc_excluded_pair_leads=sum(row["n_qc_excluded_pairs"] for row in details),
                       n_qc_excluded_target_leads=sum(row["n_qc_excluded_target_records"] for row in details))
    if task_id == "task2":
        overall["task2_missing_lead_mean_rmse_uV"] = float(np.mean(
            [row["rmse_uV"] for row in details if row["lead"] in TASK2_RMSE_LEADS]))
        overall["task2_rmse_scored_leads"] = ",".join(TASK2_RMSE_LEADS)
    return overall, details


def rmse_bonus(rmse_uV: float) -> float:
    if not np.isfinite(rmse_uV) or rmse_uV < 0:
        raise ContractError("RMSE must be finite and nonnegative")
    return 10.0 if rmse_uV <= 70 else 700.0 / rmse_uV


def evaluate_record_predictions(prediction: np.ndarray, target: np.ndarray, task_id: str,
                                metadata_rows: list[dict[str, str]], *,
                                target_quality_mask: np.ndarray | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Target-equal r; Task2 aggregates targets within device, then weights 1:1.

    An explicit target_quality_mask opts in to lead QC; omission retains legacy scoring.
    Both device groups are mandatory. Bonus is mean(b(machine RMSE), b(body RMSE)),
    never b(mean RMSE). No window- or sample-count weighting between devices.
    """
    if task_id != "task2":
        return _evaluate_record_subset(prediction, target, task_id, metadata_rows,
                                       target_quality_mask=target_quality_mask)
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.ndim != 3 or prediction.shape != target.shape or prediction.shape[1:] != (12, 5000):
        raise ContractError("Task2 requires matching raw-uV [N,12,5000] prediction/target arrays")
    if len(metadata_rows) != len(prediction):
        raise ContractError("Metadata count disagrees with predictions")
    quality = None if target_quality_mask is None else _validated_quality_mask(
        target_quality_mask, len(prediction), metadata_rows)
    groups = _record_groups(metadata_rows, len(prediction))
    for group in groups:
        if len({metadata_rows[i].get("input_type") for i in group}) != 1:
            raise ContractError("One target pair cannot mix context device types")
    if any(r.get("input_type") not in TASK2_DEVICES for r in metadata_rows):
        raise ContractError("Task2 metadata requires input_type: ecg_machine_d6 or body_scale_d6")
    summaries, device_details = {}, {}
    for device in TASK2_DEVICES:
        indices = [i for i, r in enumerate(metadata_rows) if r["input_type"] == device]
        if not indices:
            raise ContractError(f"Task2 equal weighting requires both devices; missing {device}")
        summaries[device], device_details[device] = _evaluate_record_subset(
            prediction[indices], target[indices], task_id, [metadata_rows[i] for i in indices],
            target_quality_mask=None if quality is None else quality[indices])
    machine, body = (summaries[d] for d in TASK2_DEVICES)
    result = {**machine,
              "evaluation_aggregation": "device_equal_after_target_macro_after_pair_mean_after_chronological_window_stitch",
              "rmse_aggregation": "per_device_target_equal_squared_error_then_device_equal_diagnostic",
              "n_windows": machine["n_windows"] + body["n_windows"],
              "n_records": machine["n_records"] + body["n_records"],
              "n_pairs": machine["n_pairs"] + body["n_pairs"],
              "n_unique_target_records": machine["n_unique_target_records"] + body["n_unique_target_records"],
              "n_unique_physical_target_records": len({row["target_record_id"] for row in metadata_rows}),
              "target_deduplication_scope": "within_each_device",
              "r_missing11": (machine["r_missing11"] + body["r_missing11"]) / 2,
              "missing11_mean_rmse_uV": (machine["missing11_mean_rmse_uV"] + body["missing11_mean_rmse_uV"]) / 2,
              "task2_missing_lead_mean_rmse_uV": (machine["task2_missing_lead_mean_rmse_uV"] + body["task2_missing_lead_mean_rmse_uV"]) / 2,
              "task2_rmse_bonus_score": (rmse_bonus(machine["task2_missing_lead_mean_rmse_uV"]) + rmse_bonus(body["task2_missing_lead_mean_rmse_uV"])) / 2,
              "task2_device_aggregation": TASK2_DEVICE_AGGREGATION if quality is None else QC_TASK2_DEVICE_AGGREGATION,
              "rmse_device_average_role": "diagnostic_only_never_convert_to_bonus",
              "n_undefined_record_lead_correlations": machine["n_undefined_record_lead_correlations"] + body["n_undefined_record_lead_correlations"],
              "n_undefined_pair_lead_correlations": machine["n_undefined_pair_lead_correlations"] + body["n_undefined_pair_lead_correlations"]}
    if quality is not None:
        for field in ("n_qc_excluded_pair_leads", "n_qc_excluded_target_leads"):
            result[field] = machine[field] + body[field]
    details = []
    for device in TASK2_DEVICES:
        for field in ("r_missing11", "task2_missing_lead_mean_rmse_uV", "n_records", "n_unique_target_records"):
            result[f"{device}_{field}"] = summaries[device][field]
        result[f"{device}_rmse_bonus_score"] = rmse_bonus(summaries[device]["task2_missing_lead_mean_rmse_uV"])
        details.extend({"input_type": device, **row} for row in device_details[device])
    return result, details
