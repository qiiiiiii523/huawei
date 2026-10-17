"""Actual target-free generation, full-record filtering and reliable-I/QC scoring."""
from __future__ import annotations

import copy
import hashlib
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from baselines.B5.checkpoint import checkpoint_preprocessor, model_from_checkpoint
from baselines.B5.runtime import build_validation_datasets, INPUT_FIELDS
from baselines.B5.data import preprocessing_config
from baselines.B5.flow import sample
from baselines.B5.scoring import SCORE_PROTOCOL, scores_from_summaries, diagnostic_summary, write_score_report
from baselines.B5_best.bundle import load_bundle, validate_bundle
from baselines.B5_best.combine import combine, upsample_native
from baselines.B5_best.slow import load_slow, predict_native
from baselines.B5_best.validate import pair_indices
from ecg12gen.evaluate import evaluate_record_predictions, write_report
from .common import read, write, sha, state_sha, csv_write, fresh_directory


class HybridEvaluator:
    def __init__(self, study_path, study, device):
        self.root = Path(study_path).parent
        self.study_sha256 = sha(study_path)
        self.study, self.device = study, device
        self.config = read(self.root / "data_config.json")
        self.fixed = load_bundle(self.root / "fixed_slow_bundle.pt")
        preprocessor = checkpoint_preprocessor(self.fixed["fm"], preprocessing_config(self.config))
        self.datasets = build_validation_datasets(self.config, preprocessor)
        for name, ds in self.datasets.items():
            if ds.manifest_digest != study["huawei_manifests"][name]:
                raise ValueError("Validation population changed: " + name)
        self.scale = np.asarray(self.fixed["fm"]["scales"]["d12"], np.float32)
        self.slow = load_slow(self.fixed["slow"], device, self.fixed["provenance"]["slow_variant"])
        self.learned_cache = {}

    @torch.no_grad()
    def slow_prediction(self, raw, demographics, key):
        digest = hashlib.sha256(raw.tobytes())
        for field, value in sorted(demographics.items()):
            digest.update(field.encode())
            digest.update(np.asarray(value).tobytes())
        identity = digest.hexdigest()
        if key in self.learned_cache:
            previous, learned = self.learned_cache[key]
            if previous != identity:
                raise ValueError("Repeated target has different visible I/demographics")
            return learned, identity
        fields = {k: torch.from_numpy(np.asarray(v)[None].copy()).to(self.device) for k, v in demographics.items()}
        anchor = raw.reshape(1, 6000, 10).mean(-1) / self.scale[0]
        native = predict_native(self.slow, {"anchor": torch.from_numpy(anchor[None]).to(self.device), **fields})[0].cpu().numpy()
        learned = upsample_native(native.astype(np.float64) * self.scale[1:, None].astype(np.float64))
        self.learned_cache[key] = identity, learned
        return learned, identity

    @torch.no_grad()
    def generate(self, model, raw, demographics, key, batch_size):
        result = np.empty((12, 60000), np.float32)
        starts = list(range(0, 60000, 5000))
        for offset in range(0, 12, batch_size):
            group = starts[offset:offset + batch_size]
            inputs = {"anchor": torch.from_numpy(np.stack([raw[:, s:s + 5000] / self.scale[0] for s in group])).to(self.device)}
            inputs.update({k: torch.from_numpy(np.stack([np.asarray(v) for _ in group])).to(self.device) for k, v in demographics.items()})
            prediction = sample(model, inputs, [key + ":" + str(s) for s in group], **self.study["settings"]["sampling"])
            restored = prediction.float().cpu().numpy() * self.scale[None, :, None]
            for i, start in enumerate(group):
                result[:, start:start + 5000] = restored[i]
        return result

    @torch.no_grad()
    def evaluate(self, model, output):
        with ExitStack() as stack:
            return self._evaluate(model, output, stack)

    def _evaluate(self, model, output, stack):
        output = fresh_directory(output)
        model.eval()
        summaries = {"raw": {}, "hybrid": {}}
        generated_cache = {}
        population = {}
        prediction_hashes = {}
        batch_size = int(self.config["validation"]["batch_size"])
        for task, dataset in self.datasets.items():
            folder = output / task
            folder.mkdir()
            arrays = {}
            for name in ("prediction_uV", "fm_prediction_uV", "target_uV"):
                arrays[name] = np.lib.format.open_memmap(folder / (name + ".npy"), mode="w+", dtype=np.float32,
                                                       shape=(len(dataset), 12, 5000))
                stack.callback(arrays[name]._mmap.close)
            quality = np.lib.format.open_memmap(folder / "target_quality_mask.npy", mode="w+", dtype=bool, shape=(len(dataset), 12))
            stack.callback(quality._mmap.close)
            rows = [None] * len(dataset)
            for indices in pair_indices(dataset.rows):
                first = dataset.rows[indices[0]]
                # Only observed anchor + visible metadata enter either model. Targets are read below for scoring.
                anchor = np.concatenate([dataset.base[i].anchor_i_ecg for i in indices], axis=-1).astype(np.float32)
                demo = dataset.demographics.get(first["subject_id"])
                key = first["target_record_id"]
                learned, identity = self.slow_prediction(anchor, demo, key)
                if key in generated_cache:
                    old, original, hybrid = generated_cache[key]
                    if old != identity:
                        raise ValueError("Repeated target input changed")
                else:
                    original = self.generate(model, anchor, demo, key, batch_size)
                    hybrid = combine(original, learned, self.study["settings"]["slow_beta"])
                    generated_cache[key] = identity, original, hybrid
                for window, index in enumerate(indices):
                    start = window * 5000
                    item = dataset[index]
                    arrays["prediction_uV"][index] = hybrid[:, start:start + 5000]
                    arrays["fm_prediction_uV"][index] = original[:, start:start + 5000]
                    arrays["target_uV"][index] = item["target_uV"]
                    quality[index] = item["quality_mask"]
                    rows[index] = item["evaluation_metadata"]
            for array in arrays.values():
                array.flush()
            quality.flush()
            csv_write(folder / "window_metadata.csv", rows)
            for kind, name in (("raw", "fm_prediction_uV"), ("hybrid", "prediction_uV")):
                overall, details = evaluate_record_predictions(arrays[name], arrays["target_uV"], task, rows,
                    target_quality_mask=quality, input_quality_mask=quality[:, 0], require_reliable_input=True,
                    input_quality_source="same_record_target_I")
                overall = diagnostic_summary(overall)
                overall.update(sampling=self.study["settings"]["sampling"], slow_beta=self.study["settings"]["slow_beta"],
                               target_used_at_inference=False, model_output=kind)
                write_report(folder / kind, overall, details, title=f"PublicFM {kind} complete120s QC r")
                summaries[kind][task] = overall
            population[task] = {"metadata_sha256": sha(folder / "window_metadata.csv"),
                                "target_sha256": sha(folder / "target_uV.npy"), "QC_sha256": sha(folder / "target_quality_mask.npy")}
            for array in arrays.values():
                array._mmap.close()
            quality._mmap.close()
            prediction_hashes[task] = sha(folder / "fm_prediction_uV.npy")
        scores = {kind: scores_from_summaries(value) for kind, value in summaries.items()}
        write_score_report(output, scores["hybrid"])
        write(output / "evaluation.json", {"completed": True, "score": scores["hybrid"], "raw_FM_score": scores["raw"],
              "study_sha256": self.study_sha256,
              "score_protocol": SCORE_PROTOCOL, "population": population, "fixed_slow_state_sha256": self.fixed["slow_state_sha256"],
              "FM_prediction_sha256": prediction_hashes,
              "sampling": self.study["settings"]["sampling"], "slow_beta": self.study["settings"]["slow_beta"],
              "target_used_at_inference": False, "filter_scope": "stitched_complete_120s_501_point_edge_padded_mean"})
        return scores["hybrid"]


def candidate_bundle(checkpoint, fixed, checkpoint_path=None, reference=None, manifests=None, sources=None):
    bundle = copy.deepcopy(fixed)
    keep = ("architecture_id", "architecture_hash", "model_config", "condition_schema", "preprocessing_version", "scales", "scales_sha256", "ema")
    bundle["fm"] = {k: copy.deepcopy(checkpoint[k]) for k in keep}
    bundle["fm_state_sha256"] = state_sha(checkpoint["ema"])
    bundle["provenance"] = copy.deepcopy(fixed["provenance"])
    bundle["provenance"].update(fm_sha256=sha(checkpoint_path) if checkpoint_path else None,
                                 source_fm_epoch=checkpoint.get("epoch"), stable_reference=False,
                                 source_manifests=manifests or {})
    if sources:
        # Keep the original slow-training provenance while binding this new model's
        # reference score to the actual current validation files for --verify-reference.
        bundle["provenance"]["slow_training_source_checks"] = fixed["provenance"].get("slow_source_checks", {})
        mapping = {"demographics_csv": "demographics", "subject_split_csv": "subject_split",
                   "device_interpretation_qc_csv": "device_QC"}
        for task in ("task1", "task2"):
            mapping.update({task + "_pair_manifest": task + "_pairs", task + "_window_metadata": task + "_metadata",
                            task + "_validation_target": task + "_validation_target"})
        bundle["provenance"]["slow_source_checks"] = {key: sources[value] for key, value in mapping.items()}
        bundle["provenance"]["reference_inputs_scope"] = "current_PublicFM_study_not_original_slow_training"
    bundle["reference_qc_r"] = None if reference is None else {k: reference[k] for k in ("task1_r_missing11", "task2_r_missing11")}
    bundle["reference_score_protocol"] = SCORE_PROTOCOL
    validate_bundle(bundle)
    return bundle


@torch.no_grad()
def public_diagnostic(model, datasets, settings, device, output):
    """Fixed held-out subset, true Heun16/K16 generation, never checkpoint selection."""
    model.eval()
    report = {}
    for source, dataset in datasets.items():
        count = min(len(dataset), int(settings["public"]["validation_records_per_source"]))
        values, errors, masks = [], [], []
        for start in range(0, count, settings["public"]["batch_size"]):
            items = [dataset[i] for i in range(start, min(count, start + settings["public"]["batch_size"]))]
            inputs = {k: torch.from_numpy(np.stack([np.asarray(row[k]) for row in items])).to(device) for k in INPUT_FIELDS}
            predicted = sample(model, inputs, [row["key"] for row in items], **settings["sampling"]).float().cpu().numpy()
            for p, item in zip(predicted, items):
                y = item["target"]
                x = p.astype(np.float64) - p.mean(-1, keepdims=True)
                z = y.astype(np.float64) - y.mean(-1, keepdims=True)
                denominator = np.sqrt((x * x).sum(-1) * (z * z).sum(-1))
                r = np.divide((x * z).sum(-1), denominator, out=np.zeros(12), where=denominator > 1e-12)
                mask = item["quality_mask"].copy()
                mask[0] = False
                values.append(np.clip(r, -1, 1)); masks.append(mask)
                scale = dataset.preprocessor.scale_uV_by_source["d12"]
                errors.append(np.sqrt(np.mean(((p - y) * scale[:, None]) ** 2, axis=-1)))
        r, rmse, valid = np.stack(values), np.stack(errors), np.stack(masks)
        lead_r = [float(r[valid[:, j], j].mean()) if valid[:, j].any() else None for j in range(1, 12)]
        report[source] = {"records": count, "available_validation_records": len(dataset), "keys": [r["key"] for r in dataset.rows[:count]],
                          "missing11_r": float(np.mean([v for v in lead_r if v is not None])), "per_missing_lead_r": lead_r,
                          "missing11_RMSE_uV": float(rmse[valid].mean())}
    write(output, {"completed": True, "purpose": "fixed_public_validation_subset_diagnostic_only", "used_for_selection": False,
                   "sampling": settings["sampling"], "record_seconds": 10, "sources": report})
    return report
