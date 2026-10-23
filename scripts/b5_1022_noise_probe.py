"""Evaluate the complete B5-1022 pipeline with independent or antithetic K16 noise.

The FM/P bundle, G120 correction and selected C3 adapter are all frozen.  This
is an inference-only experiment: it regenerates the visible FM/P base cache,
then applies the same G120 and device routes used by B5-1022.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sampler(mode: str, scale: float, noise_seed: int):
    """Return a B5.flow.sample-compatible K16 sampler.

    ``antithetic`` uses exactly eight keyed noises and their negatives.  It
    therefore still performs 16 Heun16 rollouts and averages 16 ECGs.
    """
    import torch
    from baselines.B5.flow import integrate, keyed_noise

    @torch.no_grad()
    def sample(model, inputs, keys, seed=42, steps=16, solver="heun", samples=16):
        if model.training or steps != 16 or solver != "heun" or samples != 16:
            raise ValueError("This probe is pinned to eval Heun16/K16")
        anchor = inputs["anchor"]
        if len(keys) != len(anchor):
            raise ValueError("Window keys and input batch disagree")
        condition = model.encode_conditions(anchor, inputs["numeric"], inputs["sex"],
                                            inputs["field_mask"], inputs["age_topcoded"])
        total = torch.zeros(len(anchor), 11, anchor.shape[-1], device=anchor.device, dtype=anchor.dtype)
        if mode == "independent":
            signed = [(index, 1.0) for index in range(16)]
        elif mode == "antithetic":
            signed = [(index, sign) for index in range(8) for sign in (1.0, -1.0)]
        else:
            raise ValueError("Unknown noise mode")
        for index, sign in signed:
            noise = keyed_noise(keys, anchor.shape[-1], noise_seed, index, anchor.device, anchor.dtype)
            if scale != 1.0 or sign != 1.0:
                noise = noise * (scale * sign)
            total += integrate(model, condition, noise, steps, solver)
        return torch.cat((condition.anchor_prediction, total / 16), dim=1)

    return sample


def generate_base(args, bundle: dict, checkpoint: Path, config: dict) -> tuple[Path, dict]:
    """Generate a current-QC, no-context FM/P validation cache."""
    from baselines.B5.checkpoint import checkpoint_preprocessor
    from baselines.B5.config import ModelConfig
    from baselines.B5.data import preprocessing_config
    from baselines.B5.runtime import build_validation_datasets, device_from_name, seed_all
    from baselines.B5.scoring import SCORE_PROTOCOL, diagnostic_summary, scores_from_summaries, write_score_report
    from baselines.B5_best.predict import Predictor
    from baselines.B5_best.validate import pair_indices
    import baselines.B5_best.predict as predict_module
    from ecg12gen.evaluate import evaluate_record_predictions, write_report

    if ModelConfig.from_dict(config["model"]).fingerprint != bundle["fm"]["architecture_hash"]:
        raise ValueError("Validation config and bundled FM architecture disagree")
    datasets = build_validation_datasets(
        config, checkpoint_preprocessor(bundle["fm"], preprocessing_config(config)))
    expected = bundle["provenance"].get("source_manifests", {})
    for task, dataset in datasets.items():
        if expected.get(task) and expected[task] != dataset.manifest_digest:
            raise ValueError("Validation population differs from bundled source: " + task)

    base = Path(args.output) / "base_evaluation"
    base.mkdir(parents=True)
    seed_all(42, True)
    predictor = Predictor(bundle, device_from_name(args.device))
    summaries, population, generated = {}, {}, {}
    try:
        with patch.object(predict_module, "sample", sampler(args.mode, args.noise_scale, args.noise_seed)):
            for task, dataset in datasets.items():
                folder = base / task
                folder.mkdir()
                shape = (len(dataset), 12, 5000)
                prediction = np.lib.format.open_memmap(folder / "prediction_uV.npy", mode="w+", dtype=np.float32, shape=shape)
                target = np.lib.format.open_memmap(folder / "target_uV.npy", mode="w+", dtype=np.float32, shape=shape)
                quality = np.lib.format.open_memmap(folder / "target_quality_mask.npy", mode="w+", dtype=bool, shape=(len(dataset), 12))
                metadata = [None] * len(dataset)
                for indices in pair_indices(dataset.rows):
                    first = dataset.rows[indices[0]]
                    anchor = np.concatenate([dataset.base[i].anchor_i_ecg for i in indices], axis=-1).astype(np.float32)
                    demographics = dataset.demographics.get(first["subject_id"])
                    digest = hashlib.sha256(anchor.tobytes())
                    for key, value in sorted(demographics.items()):
                        digest.update(key.encode())
                        digest.update(np.asarray(value).tobytes())
                    target_id = first["target_record_id"]
                    if target_id in generated:
                        prior, output = generated[target_id]
                        if prior != digest.hexdigest():
                            raise ValueError("Repeated target has different visible inputs")
                    else:
                        output = predictor.predict_encoded(anchor, demographics, target_id, args.batch_size)
                        generated[target_id] = digest.hexdigest(), output
                    for order, index in enumerate(indices):
                        item = dataset[index]
                        prediction[index] = output[:, order * 5000:(order + 1) * 5000]
                        target[index] = item["target_uV"]
                        quality[index] = item["quality_mask"]
                        metadata[index] = item["evaluation_metadata"]
                    print(f"{args.mode} scale={args.noise_scale} seed={args.noise_seed}: {task} {target_id} complete", flush=True)
                prediction.flush(); target.flush(); quality.flush()
                overall, details = evaluate_record_predictions(
                    prediction, target, task, metadata, target_quality_mask=quality,
                    input_quality_mask=quality[:, 0], require_reliable_input=True,
                    input_quality_source="same_record_target_I")
                overall = diagnostic_summary(overall)
                overall.update(model="B5-1022-noise-probe-base", sampling=bundle["sampling"],
                               slow_beta=bundle.get("slow_beta", 1.0), target_used_at_inference=False,
                               noise_mode=args.mode, noise_scale=args.noise_scale,
                               noise_seed=args.noise_seed)
                write_report(folder, overall, details, title=f"B5-1022 noise probe {task}")
                write_csv(folder / "window_metadata.csv", metadata)
                summaries[task] = overall
                population[task] = {"metadata_sha256": sha(folder / "window_metadata.csv"),
                                    "target_sha256": sha(folder / "target_uV.npy"),
                                    "QC_sha256": sha(folder / "target_quality_mask.npy")}
                del prediction, target, quality
    finally:
        for dataset in datasets.values():
            closer = getattr(dataset, "close", None)
            if callable(closer):
                closer()
    score = scores_from_summaries(summaries)
    write_score_report(base, score)
    write_json(base / "evaluation.json", {
        "completed": True, "checkpoint_sha256": sha(checkpoint), "score_protocol": SCORE_PROTOCOL,
        "sampling": bundle["sampling"], "slow_beta": bundle.get("slow_beta", 1.0),
        "population": population, "score": score, "require_reliable_input": True,
        "target_used_at_inference": False,
        "noise_sampling": {"mode": args.mode, "noise_scale": args.noise_scale,
                           "noise_seed": args.noise_seed, "K": 16,
                           "keyed_indices": list(range(16)) if args.mode == "independent" else list(range(8)),
                           "signs": "all +1" if args.mode == "independent" else "+1,-1 per index"},
    })
    return base, score


def evaluate_b5_1022(args, base_dir: Path, base_score: dict) -> dict:
    """Apply frozen G120, FIR25 and frozen C3 to the candidate FM/P cache."""
    from baselines.B5.runtime import device_from_name
    from baselines.B5.scoring import scores_from_summaries, write_score_report
    from baselines.B5_best.common import csv_write
    from baselines.B5_best_context.adapters import ContextRouter, load_settings
    from baselines.B5_best_context.evaluate import HistoryProvider, _record_metrics, _score, load_baseline
    from baselines.B5_final.body_scope import transform
    from baselines.B5_final.g120_c3 import _adapter, check_cache
    from baselines.B5_slow120.common import load_correction, load_study
    from baselines.B5_slow120.evaluate import array_batch
    from baselines.B5_slow120.model import correct_base, inputs
    from ecg12gen.evaluate import write_report

    device = device_from_name(args.device)
    study = load_study(args.slow_run)
    g120, g120_payload = load_correction(args.g120_checkpoint, study,
                                          Path(args.slow_run) / "study.json", device)
    if g120_payload.get("group") != "G120":
        raise ValueError("--g120-checkpoint must be a G120 correction checkpoint")
    g120.eval()
    c3_manifest = check_cache(args.c3_cache)
    c3, c3_payload = _adapter(c3_manifest, device, args.c3_checkpoint)
    c3.model.eval()
    c3.model.requires_grad_(False)

    _, data, population = load_baseline(base_dir)
    # The candidate prediction digest is supposed to differ from the historical
    # independent-K16 cache.  Data, targets and QC must remain identical.
    for task in ("task1", "task2"):
        for key in ("metadata_sha256", "target_sha256", "QC_sha256"):
            if population[task].get(key) != study["population"][task].get(key):
                raise ValueError("Candidate base cache changed B5-1022 validation data/QC: " + task)
    provider = HistoryProvider(study["config"], data["task2"])
    router = ContextRouter(load_settings(), machine_mode="fir25")
    c3_rows = {row["pair_id"]: row for row in c3_manifest["records"] if row["split"] == "validation"}
    out = Path(args.output) / "final"
    out.mkdir()
    scale = np.asarray(study["scale_uV"], float)
    tasks, summaries, status_rows, metrics = {}, {}, [], []
    try:
        for task, item in data.items():
            folder = out / task
            folder.mkdir()
            prediction = np.lib.format.open_memmap(folder / "prediction_uV.npy", mode="w+", dtype=np.float32,
                                                    shape=item["prediction"].shape)
            assigned = set()
            for entry in (row for row in study["validation"] if row["task"] == task):
                ids = entry["window_indices"]
                row = item["rows"][ids[0]]
                if [int(item["rows"][index]["start_sample_500hz"]) for index in ids] != list(range(0, 60000, 5000)):
                    raise ValueError("Study indices do not identify the candidate's chronological windows")
                candidate = np.concatenate([item["prediction"][index] for index in ids], axis=-1)
                with np.load(Path(args.slow_run) / entry["file"], allow_pickle=False) as slow:
                    anchor = np.asarray(slow["anchor_uV"], np.float32)
                    delta = g120(**inputs(array_batch(slow, device)))[0].detach().cpu().numpy()
                corrected = correct_base(candidate, delta, scale)
                if task == "task2" and row["input_type"] == "body_scale_d6":
                    cached = c3_rows.get(row["pair_id"])
                    if cached is None:
                        raise ValueError("C3 cache lacks validation body pair: " + row["pair_id"])
                    with np.load(Path(args.c3_cache) / cached["file"], allow_pickle=False) as arrays:
                        if not np.array_equal(anchor, arrays["anchor"]):
                            raise ValueError("C3 cache anchor differs from frozen B5-1022 study")
                        corrected_ii, _, _ = c3.predict_ii_features(
                            arrays["anchor"], corrected[1], arrays["beat"], arrays["stats"], arrays["available"])
                    enhanced = transform(corrected, anchor, corrected_ii, projection=True, scope="five")
                    status = {"adapter": "G120_C3", "history_record_id": cached.get("history_visible_sha256", "")}
                else:
                    history = provider(row) if task == "task2" and row["input_type"] == "ecg_machine_d6" else None
                    enhanced, status = router.apply(corrected, anchor, task_id=task,
                                                    input_type=row["input_type"], history=history)
                if not np.array_equal(enhanced[0], candidate[0]):
                    raise AssertionError("A B5-1022 postprocessor changed the visible I head")
                truth = np.concatenate([item["target"][index] for index in ids], axis=-1)
                metrics.append(_record_metrics(enhanced, truth, item["quality"][ids[0]], row, task))
                status_rows.append({"task": task, "pair_id": row["pair_id"], "input_type": row["input_type"], **status})
                for order, index in enumerate(ids):
                    if index in assigned:
                        raise ValueError("Duplicate candidate validation window")
                    assigned.add(index)
                    prediction[index] = enhanced[:, order * 5000:(order + 1) * 5000]
            if assigned != set(range(len(item["rows"]))):
                raise ValueError("Incomplete candidate validation coverage")
            prediction.flush()
            overall, details = _score(prediction, item, task)
            write_report(folder, overall, details, title="B5-1022 noise-probe final " + task)
            csv_write(folder / "window_metadata.csv", item["rows"])
            summaries[task] = overall
            prediction._mmap.close()
        score = scores_from_summaries(summaries)
        row = {"group": f"B5-1022_{args.mode}_scale{args.noise_scale}_seed{args.noise_seed}",
               "task1_r": score["task1_r_missing11"],
               "machine_r": summaries["task2"]["ecg_machine_d6_r_missing11"],
               "body_r": summaries["task2"]["body_scale_d6_r_missing11"],
               "task2_r": score["task2_r_missing11"], "mean_r": score["main_correlation"]}
        write_score_report(out, score)
        write_csv(out / "summary.csv", [row])
        write_csv(out / "pair_metrics.csv", metrics)
        write_csv(out / "context_records.csv", status_rows)
        write_json(out / "final.json", {"completed": True, "summary": row, "score": score,
            "base_score": base_score, "noise_sampling": {"mode": args.mode,
            "noise_scale": args.noise_scale, "noise_seed": args.noise_seed, "K": 16},
            "g120_checkpoint": str(Path(args.g120_checkpoint).resolve()),
            "g120_checkpoint_sha256": sha(args.g120_checkpoint),
            "g120_epoch": g120_payload.get("epoch"),
            "c3_checkpoint": str(Path(args.c3_checkpoint).resolve()),
            "c3_checkpoint_sha256": sha(args.c3_checkpoint),
            "c3_protocol": c3_payload.get("protocol"),
            "target_used_at_inference": False})
        print(json.dumps({"completed": True, "base_score": base_score, "final": row}, ensure_ascii=False), flush=True)
        return row
    finally:
        provider.close()
        for item in data.values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default="/home/qht/huawei")
    parser.add_argument("--slow-run", required=True, help="B5-1022 shared slow120 study directory")
    parser.add_argument("--g120-checkpoint", required=True)
    parser.add_argument("--c3-cache", required=True,
                        help="Completed G120/C3 feature cache for the selected frozen G120 checkpoint")
    parser.add_argument("--c3-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Must not already exist")
    parser.add_argument("--mode", choices=("independent", "antithetic"), required=True)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.noise_scale < 0 or not np.isfinite(args.noise_scale) or args.batch_size < 1:
        raise ValueError("Noise scale must be finite/nonnegative and batch size positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; this probe never overwrites a prior run")
    output.mkdir(parents=True)

    from baselines.B5.config import load_config
    from baselines.B5_best.bundle import load_bundle
    from baselines.B5_final.profile import BASE_SHA

    study_path = Path(args.slow_run).resolve() / "study.json"
    study = json.loads(study_path.read_text(encoding="utf-8-sig"))
    checkpoint = Path(study["base_checkpoint"]).resolve()
    if sha(checkpoint) != BASE_SHA:
        raise ValueError("B5-1022 study does not use the pinned FM/P base bundle")
    bundle = load_bundle(checkpoint)
    config = load_config(study["config"], data_only=True)
    base, base_score = generate_base(args, bundle, checkpoint, config)
    final = evaluate_b5_1022(args, base, base_score)
    write_json(output / "completed.json", {"completed": True, "base_evaluation": str(base),
        "final_output": str(output / "final"), "base_score": base_score, "final": final})
    print("Completed: " + str(output / "final" / "summary.csv"), flush=True)


if __name__ == "__main__":
    main()
