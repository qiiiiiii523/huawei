"""Direction4: train-only shared-error structure and matched linear predictability probes."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import numpy as np

from baselines.B5_best.common import read, write, csv_write, sha
from baselines.B5_best.combine import slow_curve

PROTOCOL = "B5-final-independent-shared-slow-screen-v1"
RIDGE = .01


def centered(x):
    return np.asarray(x, np.float64) - np.asarray(x, np.float64).mean(axis=-1, keepdims=True)


def pca_basis(values, weights):
    """All arrays must be six QC-valid channels; missing channels are NEVER zero-filled."""
    cov = np.zeros((6, 6), np.float64)
    for x, w in zip(values, weights):
        x = centered(x)
        if x.shape[0] != 6 or not np.isfinite(x).all():
            raise ValueError("PCA requires six finite QC-valid channels")
        cov += w * (x @ x.T) / x.shape[1]
    cov /= np.sum(weights)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order], np.maximum(eigenvalues[order], 0)


def masked_oracle(x, basis, mask):
    """Uses target-derived coefficients: structural upper bound, NEVER deployment prediction."""
    mask = np.asarray(mask, bool)
    design = basis[mask]
    if mask.sum() < basis.shape[1] or np.linalg.matrix_rank(design) < basis.shape[1]:
        return None
    observed = centered(x)[mask]
    coef = np.linalg.lstsq(design, observed, rcond=None)[0]
    return observed, design @ coef


def local_features(anchor, demographics, scale_i):
    """Only visible I and demographics; each time point sees less than 50 seconds."""
    raw = np.asarray(anchor, np.float64).reshape(60000) / float(scale_i)
    blocks = raw.reshape(600, 100)
    mean = blocks.mean(1)
    rms = np.sqrt((blocks * blocks).mean(1))
    smooth = slow_curve(raw[None])[0].reshape(600, 100).mean(1)
    sequences = np.stack((mean, rms, smooth), axis=1)
    columns = [sequences[np.clip(np.arange(600) + lag, 0, 599)] for lag in range(-120, 121, 10)]
    sex = np.eye(3)[int(demographics["sex"])]
    meta = np.r_[demographics["numeric"], sex, demographics["field_mask"], demographics["age_topcoded"]]
    columns += [np.repeat(meta[None], 600, axis=0), np.ones((600, 1))]
    return np.concatenate(columns, axis=1).astype(np.float32)


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def prepare(args, out):
    import torch
    from baselines.B5_final.profile import resolve_assets
    from baselines.B5_best.bundle import load_bundle
    from baselines.B5_best.slow import load_slow, predict_native
    from baselines.B5.runtime import seed_all, device_from_name
    from baselines.B5.config import load_config
    from baselines.B5.data import common_config, demographics_table
    from baselines.B5_best_context.evaluate import load_baseline, pairs
    from ecg12gen.device_qc import load_device_interpretation_qc, d12_target_mask
    root = Path(__file__).resolve().parents[2]
    source, incoming = Path(args.source_run).resolve(), root / "incoming/scope_data"
    assets = resolve_assets(root, checkpoint="models/base.pt", body_checkpoint="models/body.pt")
    bundle = load_bundle(assets.base)
    scale = np.asarray(bundle["fm"]["scales"]["d12"], np.float64)
    seed_all(42, True)
    device = device_from_name(args.device)
    slow = load_slow(bundle["slow"], device, bundle["provenance"]["slow_variant"])
    cfg = load_config(source / "inputs/validation.json", data_only=True)
    common = common_config(cfg)
    demo = demographics_table(cfg)
    qc = load_device_interpretation_qc(common.path("device_interpretation_qc_csv"))
    subjects = {r["subject_id"]: r["split"] for r in _rows(common.path("subject_split_csv"))}
    checks = bundle["provenance"]["slow_training_source_checks"]
    files = {"task1_train_target": incoming / "task1_train_target.npy",
        "task1_window_metadata": incoming / "task1_window_metadata.csv",
        "task2_train_target": common.path("task2_output") / "task2_train_target.npy",
        "task2_window_metadata": common.path("task2_output") / "task2_window_metadata.csv"}
    for key, path in files.items():
        if sha(path) != checks[key]["sha256"]:
            raise ValueError("Source version mismatch: " + key)
    loaded = load_baseline(source / "base_inference")
    cache = out / "slow_cache"
    cache.mkdir()
    records, seen = [], {}
    def add(raw, quality, row, split):
        target, subject = row["target_record_id"], row["subject_id"]
        if subjects.get(subject) != split:
            raise ValueError("Patient split mismatch")
        if not quality[0]:
            return
        raw = np.asarray(raw, np.float32)
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        if target in seen:
            previous = seen[target]
            if previous != (split, subject, digest):
                raise ValueError("Repeated target has inconsistent identity/signal")
            return
        seen[target] = split, subject, digest
        meta = demo.get(subject)
        anchor = raw[:1]
        anchor50 = anchor.reshape(1, 6000, 10).mean(-1) / np.float32(scale[0])
        inputs = {"anchor": torch.from_numpy(anchor50[None]).to(device),
            **{k: torch.from_numpy(np.asarray(v)[None].copy()).to(device) for k, v in meta.items()}}
        with torch.no_grad():
            native = predict_native(slow, inputs)[0, 5:11].cpu().numpy().astype(np.float64) * scale[6:, None]
        target_slow = slow_curve(raw[6:]).reshape(6, 6000, 10).mean(-1)
        error = target_slow - .75 * native
        relative = split + "/" + hashlib.sha256(target.encode()).hexdigest()[:24] + ".npz"
        path = cache / relative
        path.parent.mkdir(exist_ok=True)
        with path.open("wb") as f:
            np.savez_compressed(f, target_slow=target_slow.astype(np.float32),
                predicted_slow=native.astype(np.float32), error=error.astype(np.float32),
                features=local_features(anchor, meta, scale[0]), quality=np.asarray(quality, bool))
        records.append({"split": split, "subject_id": subject, "target_record_id": target,
            "file": relative, "sha256": sha(path), "complete_chest_QC": bool(quality[6:].all()),
            "valid_chest_leads": int(quality[6:].sum())})
        if len(records) % 20 == 0:
            print({"phase": "slow_cache", "unique_records": len(records)}, flush=True)
    try:
        for task in ("task1", "task2"):
            target_array = np.load(files[task + "_train_target"], mmap_mode="r")
            rows = [r for r in _rows(files[task + "_window_metadata"]) if r["split"] == "train"]
            rows.sort(key=lambda r: int(r["array_index"]))
            if len(rows) != len(target_array):
                raise ValueError("Training metadata/array count mismatch")
            authoritative = {r["pair_id"]: r for r in _rows(common.path(task + "_pair_manifest_csv"))}
            for ids in pairs(rows):
                row = rows[ids[0]]
                pair = authoritative[row["pair_id"]]
                quality = d12_target_mask(qc[row["target_record_id"]])
                if (pair["pair_status"] != "paired" or pair["target_quality_status"] != "usable"
                        or row["quality_status"] != "usable"
                        or qc[row["target_record_id"]]["d12_direct_supervision_eligible"] != "true"):
                    continue
                raw = np.concatenate([target_array[int(rows[i]["array_index"])] for i in ids], axis=-1)
                add(raw, quality, row, "train")
            target_array._mmap.close()
        for task, item in loaded[1].items():
            for ids in item["pairs"]:
                raw = np.concatenate([item["target"][i] for i in ids], axis=-1)
                add(raw, item["quality"][ids[0]], item["rows"][ids[0]], "validation")
    finally:
        for item in loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()
    train_subjects = {r["subject_id"] for r in records if r["split"] == "train"}
    if train_subjects & {r["subject_id"] for r in records if r["split"] == "validation"}:
        raise ValueError("Patient leakage")
    manifest = {"protocol": PROTOCOL, "completed": True, "records": records,
        "scale_chest": scale[6:].tolist(), "source_run": str(source),
        "base_sha256": sha(assets.base), "body_sha256": sha(assets.body),
        "baseline": "ORIGINAL B5-final C0: original body only II; NOT direction2 C3",
        "target_definition": "MA501(full raw chest) block-mean50Hz minus .75*P-native; per-record constants removed only for structure/probe targets",
        "sources": {k: {"path": str(p), "sha256": sha(p)} for k, p in files.items()},
        "FM_recomputed": False, "P_frozen": True, "history_reference_recheck_deferred": True}
    write(cache / "cache.json", manifest)
    return manifest


def diagnose(manifest, out):
    cache = out / "slow_cache"
    records = manifest["records"]
    arrays = {}
    for row in records:
        if sha(cache / row["file"]) != row["sha256"]:
            raise ValueError("Slow cache changed")
        with np.load(cache / row["file"], allow_pickle=False) as f:
            arrays[row["target_record_id"]] = {k: f[k].copy() for k in f.files}
    train = [r for r in records if r["split"] == "train" and r["complete_chest_QC"]]
    if len({r["subject_id"] for r in train}) < 5:
        raise ValueError("Too few complete-QC training patients for this first PCA design")
    counts = Counter(r["subject_id"] for r in train)
    weights = np.array([1 / counts[r["subject_id"]] for r in train])
    scale = np.asarray(manifest["scale_chest"])[:, None]
    bases = {}
    spectrum = []
    for kind in ("target_slow", "error"):
        basis, eigenvalues = pca_basis([arrays[r["target_record_id"]][kind] / scale for r in train], weights)
        bases[kind] = basis
        for j, value in enumerate(eigenvalues):
            spectrum.append({"kind": kind, "rank": j + 1, "eigenvalue": float(value),
                "cumulative_train_energy_fraction": float(eigenvalues[:j + 1].sum() / max(eigenvalues.sum(), 1e-12))})
    structure = []
    for kind in ("target_slow", "error"):
        for split in ("train", "validation"):
            rows = [r for r in records if r["split"] == split]
            subject_counts = Counter(r["subject_id"] for r in rows)
            for rank in (1, 2, 3):
                energy = remaining = 0.
                used = 0
                fractions = []
                for r in rows:
                    a = arrays[r["target_record_id"]]
                    pair = masked_oracle(a[kind] / scale, bases[kind][:, :rank], a["quality"][6:])
                    if pair is None:
                        continue
                    target, reconstructed = pair
                    total = float((target * target).sum())
                    if total < 1e-12:
                        continue
                    loss = float(((target - reconstructed) ** 2).sum())
                    weight = 1 / subject_counts[r["subject_id"]]
                    energy += total * weight; remaining += loss * weight
                    used += 1; fractions.append(1 - loss / total)
                structure.append({"kind": kind, "split": split, "rank": rank,
                    "oracle_only": True, "records_used": used, "records_skipped": len(rows) - used,
                    "explained_energy_fraction": 1 - remaining / energy if energy else None,
                    "median_record_fraction": float(np.median(fractions)) if fractions else None})
    csv_write(out / "pca_spectrum.csv", spectrum)
    csv_write(out / "heldout_structure.csv", structure)
    write(out / "pca.json", {"basis_error": bases["error"].tolist(),
        "basis_true_slow": bases["target_slow"].tolist(), "training_records_used": len(train),
        "training_subjects_used": len(counts), "PCA_fit": "training complete-six-QC records ONLY; patient balanced",
        "missing_QC": "not zero filled; heldout oracle fits coefficients only to valid observed channels",
        "oracle_warning": "Heldout coefficients use target signals: structural upper bound, not a predictor."})
    # Same data/features/ridge for all probes. Shared probes constrain the output subspace.
    total_weight = float(weights.sum())
    mean = sum(w * arrays[r["target_record_id"]]["features"].astype(float).mean(0)
               for r, w in zip(train, weights)) / total_weight
    variance = sum(w * ((arrays[r["target_record_id"]]["features"] - mean) ** 2).mean(0)
                   for r, w in zip(train, weights)) / total_weight
    std = np.sqrt(np.maximum(variance, 1e-8))
    mean[-1], std[-1] = 0., 1.
    dimension = len(mean)
    gram, cross = np.zeros((dimension, dimension)), np.zeros((dimension, 6))
    for r, w in zip(train, weights):
        a = arrays[r["target_record_id"]]
        x = (a["features"].astype(float) - mean) / std
        y = centered((a["error"] / scale).reshape(6, 600, 10).mean(-1)).T
        gram += w * (x.T @ x) / 600
        cross += w * (x.T @ y) / 600
    gram /= total_weight; cross /= total_weight
    penalty = np.eye(dimension) * RIDGE; penalty[-1, -1] = 0
    full = np.linalg.solve(gram + penalty, cross)
    coefficients = {"independent_ridge": full}
    for rank in (2, 3):
        basis = bases["error"][:, :rank]
        coefficients[f"shared{rank}_ridge"] = full @ basis @ basis.T
    with (out / "train_fitted_probe.npz").open("wb") as f:
        np.savez(f, mean=mean, std=std, **coefficients)
    predictions = {name: {} for name in coefficients}
    predictability = []
    validation_counts = Counter(r["subject_id"] for r in records if r["split"] == "validation")
    for name, coefficient in coefficients.items():
        energy = remaining = 0.
        for r in records:
            if r["split"] != "validation":
                continue
            a = arrays[r["target_record_id"]]
            x = (a["features"].astype(float) - mean) / std
            predicted = (x @ coefficient).T
            predictions[name][r["target_record_id"]] = predicted * scale
            mask = a["quality"][6:]
            y = centered((a["error"] / scale).reshape(6, 600, 10).mean(-1))[mask]
            p = centered(predicted)[mask]
            weight = 1 / validation_counts[r["subject_id"]]
            energy += weight * float((y * y).sum()); remaining += weight * float(((y - p) ** 2).sum())
        predictability.append({"probe": name, "heldout_error_explained_fraction": 1 - remaining / energy,
            "training_records": len(train), "ridge": RIDGE, "input_features": dimension,
            "interpretation": "Lightweight local-I/demographic linear screening, not final nonlinear architecture"})
    csv_write(out / "predictability.csv", predictability)
    return predictions


def evaluate_probes(manifest, predictions, out):
    from baselines.B5_best_context.evaluate import load_baseline, _score, _record_metrics
    from baselines.B5.scoring import scores_from_summaries, write_score_report
    from ecg12gen.evaluate import write_report
    source = Path(manifest["source_run"])
    loaded = load_baseline(source / "base_inference")
    summary = []
    # Restore ORIGINAL B5-final context, never direction2's new five-limb weights.
    baseline = {"task1": loaded[1]["task1"]["prediction"],
        "task2": np.load(source / "study/C0_no_projection_II/task2/prediction_uV.npy", mmap_mode="r")}
    try:
        for name in ("original_B5_final", *predictions):
            group = out / "scores" / name
            tasks, pair_metrics = {}, []
            for task, item in loaded[1].items():
                folder = group / task
                folder.mkdir(parents=True)
                target_path = folder / "prediction_uV.npy"
                output = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float32,
                    shape=item["prediction"].shape)
                output[:] = baseline[task]
                for ids in item["pairs"]:
                    row = item["rows"][ids[0]]
                    if name != "original_B5_final" and row["target_record_id"] in predictions[name]:
                        delta5 = predictions[name][row["target_record_id"]]
                        centers = np.arange(600) * 100 + 49.5
                        delta = np.stack([np.interp(np.arange(60000), centers, v) for v in delta5])
                        for j, index in enumerate(ids):
                            output[index, 6:] = (baseline[task][index, 6:].astype(float)
                                + delta[:, j * 5000:(j + 1) * 5000]).astype(np.float32)
                    result = np.concatenate([output[i] for i in ids], axis=-1)
                    truth = np.concatenate([item["target"][i] for i in ids], axis=-1)
                    pair_metrics.append(_record_metrics(result, truth, item["quality"][ids[0]], row, task))
                if not np.array_equal(output[:, :6], baseline[task][:, :6]):
                    raise AssertionError("Direction4 changed limb/context outputs")
                output.flush()
                overall, details = _score(output, item, task)
                tasks[task] = overall
                write_report(folder, overall, details, title=f"Direction4 independent screen {name} {task}")
                csv_write(folder / "window_metadata.csv", item["rows"])
                output._mmap.close()
            score = scores_from_summaries(tasks)
            write_score_report(group, score)
            csv_write(group / "pair_metrics.csv", pair_metrics)
            row = {"candidate": name, "task1_r": score["task1_r_missing11"],
                "machine_r": tasks["task2"]["ecg_machine_d6_r_missing11"],
                "body_r": tasks["task2"]["body_scale_d6_r_missing11"],
                "task2_r": score["task2_r_missing11"], "mean_r": score["main_correlation"]}
            if name == "original_B5_final":
                expected = next(r["mean_r"] for r in read(source / "study/final.json")["summary"]
                    if r["group"] == "C0_no_projection_II")
                if abs(row["mean_r"] - expected) > 1e-10:
                    raise ValueError("Original B5-final baseline did not reproduce this run's C0")
            row["delta_mean_r_vs_original"] = row["mean_r"] - (summary[0]["mean_r"] if summary else row["mean_r"])
            summary.append(row)
            print({"phase": "probe_score", **row}, flush=True)
        csv_write(out / "summary.csv", summary)
        return summary
    finally:
        baseline["task2"]._mmap.close()
        for item in loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Use a new independent direction4 output")
    out.mkdir(parents=True)
    write(out / "protocol.json", {"protocol": PROTOCOL,
        "baseline": "Original B5-final; direction2 C3/body continuation NOT applied",
        "fit_on": "train patients only; fixed ridge0.01 and ranks2/3, no validation tuning",
        "screening_candidates": ["independent_ridge", "shared2_ridge", "shared3_ridge"],
        "scope": "Structural diagnosis and linear predictability screen; neural shared+private candidates not trained yet",
        "PCA_oracle_is_not_deployable": True, "QC_missing_never_zero_filled": True})
    manifest = prepare(args, out)
    predictions = diagnose(manifest, out)
    summary = evaluate_probes(manifest, predictions, out)
    write(out / "completed.json", {"completed": True, "summary": summary,
        "baseline": "Original B5-final independent from direction2",
        "structure": read(out / "pca.json"), "training_records": len([r for r in manifest["records"] if r["split"] == "train"]),
        "validation_records": len([r for r in manifest["records"] if r["split"] == "validation"]),
        "FM_updates": 0, "P_updates": 0, "neural_candidate_training_started": False,
        "next": "Use shared error structure, heldout predictability and actual full r to select independent/shared+private neural candidates."})


if __name__ == "__main__":
    main()
