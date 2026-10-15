"""Score complete hybrid outputs with the unchanged record/QC r aggregation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from baselines.B5.checkpoint import checkpoint_preprocessor
from baselines.B5.config import load_config, ModelConfig
from baselines.B5.data import preprocessing_config, common_config, demographics_path
from baselines.B5.runtime import build_validation_datasets, device_from_name, seed_all
from ecg12gen.evaluate import evaluate_record_predictions, write_report
from . import SCORE_PROTOCOL
from .bundle import load_bundle
from .common import csv_write, write, sha
from .predict import Predictor
from .scoring import scores_from_summaries, write_score_report, diagnostic_summary


def pair_indices(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["pair_id"]].append(index)
    result = []
    for indices in groups.values():
        indices.sort(key=lambda i: int(rows[i]["start_sample_500hz"]))
        if [int(rows[i]["start_sample_500hz"]) for i in indices] != list(range(0, 60000, 5000)):
            raise ValueError("Each pair requires twelve chronological real 10-second windows")
        for key in ("target_record_id", "subject_id", "input_type"):
            if len({rows[i][key] for i in indices}) != 1:
                raise ValueError("Pair mixes record/subject/device identities")
        result.append(indices)
    return result


def verify_reference_inputs(bundle, config):
    """A source checkpoint name alone is not evidence that validation data stayed fixed."""
    common = common_config(config)
    current = {"demographics_csv": demographics_path(config),
               "subject_split_csv": common.path("subject_split_csv"),
               "device_interpretation_qc_csv": common.path("device_interpretation_qc_csv")}
    for task in ("task1", "task2"):
        current[task + "_pair_manifest"] = common.path(task + "_pair_manifest_csv")
        directory = common.path(task + "_output")
        current[task + "_window_metadata"] = directory / (task + "_window_metadata.csv")
        current[task + "_validation_target"] = directory / (task + "_validation_target.npy")
    expected = bundle["provenance"]["slow_source_checks"]
    for key, path in current.items():
        if key not in expected or sha(path) != expected[key]["sha256"]:
            raise ValueError("Reference data/QC/demographics changed: " + key)


def evaluate(bundle, datasets, device, output, bundle_sha, batch_size=8, verify_reference=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty validation output directory")
    if set(datasets) != {"task1", "task2"}:
        raise ValueError("Both task validation datasets are required")
    seed_all(bundle["sampling"]["seed"], True)
    predictor = Predictor(bundle, device)
    summaries, population, cache = {}, {}, {}
    for task, ds in datasets.items():
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        shape = (len(ds), 12, 5000)
        prediction = np.lib.format.open_memmap(folder / "prediction_uV.npy", mode="w+", dtype=np.float32, shape=shape)
        target = np.lib.format.open_memmap(folder / "target_uV.npy", mode="w+", dtype=np.float32, shape=shape)
        quality = np.lib.format.open_memmap(folder / "target_quality_mask.npy", mode="w+", dtype=bool, shape=(len(ds), 12))
        metadata = [None] * len(ds)
        for indices in pair_indices(ds.rows):
            first = ds.rows[indices[0]]
            # Read the explicit observed input field; hidden Y never enters Predictor.
            anchor = np.concatenate([ds.base[i].anchor_i_ecg for i in indices], axis=-1).astype(np.float32)
            demographics = ds.demographics.get(first["subject_id"])
            digest = hashlib.sha256(anchor.tobytes())
            for key, value in sorted(demographics.items()):
                digest.update(key.encode()); digest.update(np.asarray(value).tobytes())
            key = first["target_record_id"]
            if key in cache:
                previous_digest, generated = cache[key]
                if previous_digest != digest.hexdigest():
                    raise ValueError("Repeated target has different visible I/demographics")
            else:
                generated = predictor.predict_encoded(anchor, demographics, key, batch_size)
                cache[key] = (digest.hexdigest(), generated)
            for start, index in enumerate(indices):
                item = ds[index]
                prediction[index] = generated[:, start * 5000:(start + 1) * 5000]
                target[index] = item["target_uV"]
                quality[index] = item["quality_mask"]
                metadata[index] = item["evaluation_metadata"]
            print(f"{task}: {key} complete", flush=True)
        prediction.flush(); target.flush(); quality.flush()
        overall, details = evaluate_record_predictions(prediction, target, task, metadata, target_quality_mask=quality)
        overall = diagnostic_summary(overall)
        overall.update(sampling=bundle["sampling"], model="B5-best-output-hybrid",
                       target_used_at_inference=False, target_record_seconds=120)
        write_report(folder, overall, details, title=f"B5-best {task} full-record QC correlation")
        csv_write(folder / "window_metadata.csv", metadata)
        summaries[task] = overall
        population[task] = {"metadata_sha256": sha(folder / "window_metadata.csv"),
                            "target_sha256": sha(folder / "target_uV.npy"),
                            "QC_sha256": sha(folder / "target_quality_mask.npy")}
        del prediction, target, quality
    score = scores_from_summaries(summaries)
    write_score_report(output, score)
    reference = bundle.get("reference_qc_r")
    matched = None if reference is None else all(
        abs(score[k] - reference[k]) <= bundle["reference_tolerance"] for k in reference)
    write(output / "evaluation.json", {"completed": True, "checkpoint_sha256": bundle_sha,
          "score_protocol": SCORE_PROTOCOL, "sampling": bundle["sampling"], "population": population,
          "score": score, "reference_qc_r": reference, "reference_matched": matched,
          "target_used_at_inference": False, "provenance": bundle["provenance"]})
    if verify_reference and matched is not True:
        raise ValueError("Historical r not reproduced: keep this report and check inputs/numerics before promoting the bundle")
    print(json.dumps(score, ensure_ascii=False), flush=True)
    return score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="May use a data config from the original /home/qht/huawei checkout")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--verify-reference", action="store_true")
    args = p.parse_args()
    bundle, cfg = load_bundle(args.checkpoint), load_config(args.config, data_only=True)
    if args.verify_reference:
        verify_reference_inputs(bundle, cfg)
    if ModelConfig.from_dict(cfg["model"]).fingerprint != bundle["fm"]["architecture_hash"]:
        raise ValueError("Validation config and bundled FM architecture disagree")
    preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(cfg))
    datasets = build_validation_datasets(cfg, preprocessor)
    source = bundle["provenance"].get("source_manifests", {})
    for task, ds in datasets.items():
        if source.get(task) and source[task] != ds.manifest_digest:
            raise ValueError("Validation population differs from source: " + task)
    evaluate(bundle, datasets, device_from_name(args.device), args.output_dir,
             sha(args.checkpoint), args.batch_size, args.verify_reference)


if __name__ == "__main__":
    main()
