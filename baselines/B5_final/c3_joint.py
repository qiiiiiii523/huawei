"""C3 paired continuation: fixed-I projection plus one learned five-limb residual."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from baselines.B5_best.common import read, write, sha, csv_write
from baselines.B5_final.body_scope import transform
from baselines.B5_best_context.fir import B, C

PROTOCOL = "B5-C3-five-limb-continuation-v1"


def project(base, anchor):
    x = np.asarray(base, np.float64)
    i = np.asarray(anchor, np.float64).reshape(1, -1)
    q = (B * (x[1:6] - C * i)).sum(0) / float((B * B).sum())
    return (B * q + C * i).astype(np.float32)


def joint_prediction(model_output, encoded_base, projected, scales):
    """All signal tensors are normalized by the current-I scale, not target statistics."""
    import torch
    coeff = torch.as_tensor([1., 1., -.5, -.5, 1.], device=model_output.device)
    residual = model_output - encoded_base
    return projected + coeff[None, :, None] * residual[:, None, :]


def correlation_loss(prediction, target, mask, weight, residual):
    import torch
    p = prediction - prediction.mean(-1, keepdim=True)
    t = target - target.mean(-1, keepdim=True)
    den = p.norm(dim=-1) * t.norm(dim=-1)
    r = (p * t).sum(-1) / den.clamp_min(1e-7)
    valid = mask & (t.square().sum(-1) > 1e-10)
    w = valid.to(prediction.dtype) * weight[:, None]
    if not bool(valid.any()):
        return residual.square().mean() * 0
    fit = ((1 - r) * w).sum() / w.sum().clamp_min(1e-7)
    reg = .01 * (residual.square().mean(-1) * weight).sum() / weight.sum().clamp_min(1e-7)
    return fit + reg


def assert_disjoint(records):
    for key in ("subject_id", "target_record_id"):
        train = {r[key] for r in records if r["split"] == "train"}
        validation = {r[key] for r in records if r["split"] == "validation"}
        if train & validation:
            raise ValueError("Train/validation overlap: " + key)


def check_manifest(root):
    cache = read(root / "cache.json")
    if cache.get("protocol") != PROTOCOL or not cache.get("completed"):
        raise ValueError("Incomplete or incompatible C3 cache")
    assert_disjoint(cache["records"])
    for row in cache["records"]:
        if sha(root / row["file"]) != row["sha256"]:
            raise ValueError("Changed cache record: " + row["file"])
    return cache


def prepare(args):
    from baselines.B5_final.profile import resolve_assets, BASE_SHA, check_payloads
    from baselines.B5_best.bundle import load_bundle
    from baselines.B5_best.predict import Predictor
    from baselines.B5.runtime import seed_all, device_from_name
    from baselines.B5.config import load_config
    from baselines.B5.data import common_config, demographics_table
    from baselines.B5_best_context.body_model import BodyAdapter
    from baselines.B5_best_context.body_features import relation_features, feature_encoding
    from baselines.B5_best_context.body_history import history_policy, process_history
    from baselines.B5_best_context.adapters import history_from_sample, _visible_digest
    from baselines.B5_best_context.evaluate import load_baseline, pairs
    from ecg12gen.dataset import JointAnchorDataset
    root, source = Path(args.output).resolve(), Path(args.source_run).resolve()
    repo = Path(__file__).resolve().parents[2]
    assets = resolve_assets(repo, checkpoint="models/base.pt", body_checkpoint="models/body.pt")
    cfg_path = source / "inputs/validation.json"
    cfg = load_config(cfg_path, data_only=True)
    common = common_config(cfg)
    bundle = load_bundle(assets.base)
    engine = BodyAdapter(assets.body, BASE_SHA, device_from_name(args.device), expected_group="context")
    check_payloads(bundle, engine.payload)
    loaded = load_baseline(source / "base_inference")
    folder = common.path("task2_output")
    input_files = {"config": cfg_path, "common": Path(cfg["paths"]["common_config"]),
        "demographics": Path(cfg["paths"]["demographics"]),
        "split": common.path("subject_split_csv"), "qc": common.path("device_interpretation_qc_csv"),
        "pair_manifest": common.path("task2_pair_manifest_csv"),
        "window_metadata": folder / "task2_window_metadata.csv",
        "history_metadata": folder / "task2_context_window_metadata.csv"}
    for split in ("train", "validation"):
        for kind in ("target", "context", "context_valid_mask"):
            input_files[split + "_" + kind] = folder / f"task2_{split}_{kind}.npy"
    reference = read(source / "base_inference/evaluation.json")
    source_checks = reference["provenance"]["slow_training_source_checks"]
    for key in ("task2_train_target", "task2_validation_target"):
        split = "train" if key == "task2_train_target" else "validation"
        if sha(input_files[split + "_target"]) != source_checks[key]["sha256"]:
            raise ValueError("Training/validation targets differ from frozen source: " + key)
    frozen = {"base_sha256": sha(assets.base), "body_sha256": sha(assets.body),
        "base_report_sha256": sha(source / "base_inference/evaluation.json"),
        "population": loaded[2], "source_run": str(source),
        "inputs": {k: {"path": str(p), "sha256": sha(p)} for k, p in input_files.items()},
        "history_preprocessing": history_policy(engine.payload),
        "feature_encoding": feature_encoding(engine.payload)}
    ledger = root / "cache.json"
    if ledger.exists():
        manifest = read(ledger)
        if not args.resume or manifest["frozen"] != frozen:
            raise ValueError("Resume changed cache/source data or --resume missing")
    else:
        if root.exists():
            raise FileExistsError("Use a new cache output")
        root.mkdir(parents=True)
        manifest = {"protocol": PROTOCOL, "completed": False, "frozen": frozen, "records": []}
        write(ledger, manifest)
    existing = {(r["split"], r["pair_id"]): r for r in manifest["records"]}
    demographics = demographics_table(cfg)
    predictor, generated = None, {}
    datasets = []
    try:
        validation = loaded[1]["task2"]
        lookup = {(r["pair_id"], int(r["start_sample_500hz"])): i
                  for i, r in enumerate(validation["rows"])}
        for split in ("validation", "train"):
            ds = JointAnchorDataset(common, "task2", split)
            datasets.append(ds)
            rows = [ds._rows[i] for i in ds._indices]
            for ids in pairs(rows):
                row = rows[ids[0]]
                if row["input_type"] != "body_scale_d6":
                    continue
                key = split, row["pair_id"]
                if key in existing:
                    if sha(root / existing[key]["file"]) != existing[key]["sha256"]:
                        raise ValueError("Changed partial record")
                    continue
                samples = [ds[i] for i in ids]
                anchor = np.concatenate([s.anchor_i_ecg for s in samples], axis=-1).astype(np.float32)
                truth = np.concatenate([s.Y_12lead[1:6] for s in samples], axis=-1).astype(np.float32)
                quality = np.asarray(samples[0].target_quality_mask, bool)
                if any(not np.array_equal(s.target_quality_mask, quality) for s in samples):
                    raise ValueError("QC changed within record")
                history = history_from_sample(samples[0])
                processed = process_history(history, history_policy(engine.payload))
                beat, stats, available = relation_features(processed, encoding=feature_encoding(engine.payload))
                if split == "validation":
                    indices = [lookup[(row["pair_id"], j * 5000)] for j in range(12)]
                    for j, index in enumerate(indices):
                        if (not np.array_equal(samples[j].Y_12lead, validation["target"][index])
                                or not np.array_equal(quality, validation["quality"][index])):
                            raise ValueError("Validation targets/QC changed")
                    base = np.concatenate([validation["prediction"][i] for i in indices], axis=-1)
                else:
                    meta = demographics.get(row["subject_id"])
                    digest = hashlib.sha256(anchor.tobytes())
                    for name, val in sorted(meta.items()):
                        digest.update(name.encode()); digest.update(np.asarray(val).tobytes())
                    target_id = row["target_record_id"]
                    if target_id not in generated:
                        if predictor is None:
                            seed_all(42, True)
                            predictor = Predictor(bundle, device_from_name(args.device))
                        base = predictor.predict_encoded(anchor, meta, target_id, 8)
                        generated[target_id] = digest.hexdigest(), base
                    else:
                        prev, base = generated[target_id]
                        if prev != digest.hexdigest():
                            raise ValueError("Repeated target has different visible inputs")
                relative = split + "/" + hashlib.sha256(row["pair_id"].encode()).hexdigest()[:24] + ".npz"
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                tmp = path.with_suffix(".pending")
                with tmp.open("wb") as f:
                    np.savez_compressed(f, anchor=anchor[0], base_ii=base[1],
                        projected=project(base, anchor), target_limb=truth,
                        quality=quality, beat=beat, stats=stats, available=np.array(available))
                tmp.replace(path)
                manifest["records"].append({"split": split, "pair_id": row["pair_id"],
                    "subject_id": row["subject_id"], "target_record_id": row["target_record_id"],
                    "history_visible_sha256": _visible_digest(history), "file": relative,
                    "sha256": sha(path), "history_available": bool(available),
                    "eligible": bool(quality[0] and quality[1:6].any())})
                assert_disjoint(manifest["records"])
                write(ledger, manifest)
                print({"phase": "prepare", "split": split,
                       "cached_records": sum(r["split"] == split for r in manifest["records"])}, flush=True)
            if split == "validation":
                # Real zero-step deployment check before generating the training cache.
                temp = root / "epoch0_preflight"
                score = assess(engine, manifest, root, source, temp)
                expected = read(source / "study/final.json")["summary"][-1]["mean_r"]
                if abs(score["mean_r"] - expected) > 1e-10:
                    raise ValueError("Epoch0 does not reproduce this run's C3")
                write(root / "preflight.json", {"epoch0_C3_mean_r": score["mean_r"],
                    "expected_C3_mean_r": expected, "matched": True,
                    "historical_reference_recheck_deferred": True, "FM_updates": 0, "slow_updates": 0})
                print({"phase": "epoch0_C3_verified", **score}, flush=True)
        if any(not any(r["split"] == s and r["eligible"] for r in manifest["records"])
               for s in ("train", "validation")):
            raise ValueError("Empty eligible split")
        manifest["completed"] = True
        write(ledger, manifest)
        check_manifest(root)
        print({"phase": "cache_completed", "counts": dict(Counter(r["split"] for r in manifest["records"]))}, flush=True)
    finally:
        for ds in datasets:
            ds.close()
        for item in loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()


def assess(engine, manifest, cache_root, source, output=None):
    """Official full120s score, fixed Task1/machine/chest and target-equal lead QC."""
    from baselines.B5_best_context.evaluate import load_baseline, _score, _record_metrics
    loaded = load_baseline(source / "base_inference")
    try:
        item = loaded[1]["task2"]
        ref = source / "study/C3_projection_five/task2/prediction_uV.npy"
        fixed = np.load(ref, mmap_mode="r")
        prediction = np.array(fixed, copy=True)
        fixed._mmap.close()
        lookup = {item["rows"][ids[0]]["pair_id"]: ids for ids in item["pairs"]}
        metrics = []
        for row in manifest["records"]:
            if row["split"] != "validation":
                continue
            ids = lookup[row["pair_id"]]
            with np.load(cache_root / row["file"], allow_pickle=False) as a:
                corrected, _, _ = engine.predict_ii_features(a["anchor"], a["base_ii"],
                    a["beat"], a["stats"], a["available"])
                base = np.concatenate([item["prediction"][i] for i in ids], axis=-1)
                result = transform(base, a["anchor"], corrected, projection=True, scope="five")
                if not np.array_equal(result[[0, *range(6, 12)]], base[[0, *range(6, 12)]]):
                    raise AssertionError("C3 altered I/chest")
                for j, index in enumerate(ids):
                    prediction[index] = result[:, j * 5000:(j + 1) * 5000]
                truth = np.concatenate([item["target"][i] for i in ids], axis=-1)
                metrics.append(_record_metrics(result, truth, item["quality"][ids[0]], item["rows"][ids[0]], "task2"))
        overall, details = _score(prediction, item, "task2")
        task1 = loaded[0]["score"]["task1_r_missing11"]
        score = {"task1_r": task1, "machine_r": overall["ecg_machine_d6_r_missing11"],
            "body_r": overall["body_scale_d6_r_missing11"], "task2_r": overall["r_missing11"],
            "mean_r": (task1 + overall["r_missing11"]) / 2}
        if output is not None:
            from ecg12gen.evaluate import write_report
            output.mkdir(parents=True, exist_ok=False)
            write_report(output, overall, details, title="C3 five-limb continuation")
            csv_write(output / "pair_metrics.csv", metrics)
            write(output / "score.json", score)
        return score
    finally:
        for item in loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()


def train(args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from baselines.B5.runtime import seed_all, device_from_name
    from baselines.B5_best_context.body_model import BodyAdapter
    from baselines.B5_best_context.body_features import encode_windows
    from baselines.B5_final.profile import BASE_SHA, BODY_SHA
    root, out = Path(args.cache).resolve(), Path(args.output).resolve()
    cache = check_manifest(root)
    source = Path(cache["frozen"]["source_run"])
    repo = Path(__file__).resolve().parents[2]
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Positive training settings required")
    if out.exists():
        raise FileExistsError("Use a new training directory")
    seed_all(args.seed, True)
    device = device_from_name(args.device)
    engine = BodyAdapter(repo / "models/body.pt", BASE_SHA, device, expected_group="context")
    if sha(repo / "models/body.pt") != BODY_SHA or cache["frozen"]["body_sha256"] != BODY_SHA:
        raise ValueError("Wrong initial body weights")
    model = engine.model
    records = [r for r in cache["records"] if r["split"] == "train" and r["eligible"]]
    counts = Counter(r["target_record_id"] for r in records)
    values = []
    for row in records:
        with np.load(root / row["file"], allow_pickle=False) as a:
            signal, base, scales = encode_windows(a["anchor"], a["base_ii"])
            for j in range(12):
                if scales[j] <= 0:
                    continue
                valid = a["quality"][1:6].copy()
                if args.objective == "II":
                    valid[1:] = False
                target = a["target_limb"][:, j * 5000:(j + 1) * 5000] / scales[j]
                valid &= np.std(target.astype(float), axis=-1) > 0
                if not valid.any():
                    continue
                values.append((signal[j], base[j], a["projected"][:, j * 5000:(j + 1) * 5000] / scales[j],
                    target, a["beat"], a["stats"], np.float32(a["available"]), valid,
                    np.float32(1 / counts[row["target_record_id"]]), np.float32(scales[j])))
    if not values:
        raise ValueError("No usable training windows")
    tensors = [torch.from_numpy(np.stack([x[k] for x in values]).copy()) for k in range(10)]
    data = TensorDataset(*tensors)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    model.requires_grad_(True)
    out.mkdir(parents=True)
    config = {"protocol": PROTOCOL, "objective": args.objective,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "seed": args.seed, "cache_sha256": sha(root / "cache.json"),
        "base_sha256": BASE_SHA, "initial_body_sha256": BODY_SHA,
        "initialization": "existing adapted body.pt; epoch0 retains frozen C3",
        "loss": "QC-masked limb window PCC + 0.01 normalized residual L2; inverse repeated-target weight",
        "selection": "official full120s mean_r with fixed Task1/machine/chest; includes epoch0",
        "FM_updates": 0, "slow_updates": 0, "training_windows": len(data),
        "historical_reference_recheck_deferred": True,
        "sampling": {"seed": 42, "solver": "heun", "steps": 16, "samples": 16},
        "slow_beta": .75, "history_preprocessing": cache["frozen"]["history_preprocessing"],
        "feature_encoding": cache["frozen"]["feature_encoding"]}
    write(out / "config.json", config)
    model.eval()
    initial = assess(engine, cache, root, source, out / "validation/epoch_000")
    expected = read(source / "study/final.json")["summary"][-1]["mean_r"]
    if abs(initial["mean_r"] - expected) > 1e-10:
        raise ValueError("Training epoch0 changed C3")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    best, best_epoch, updates = initial["mean_r"], 0, 0
    history = [{"epoch": 0, "updates": 0, "train_loss": None, **initial,
        "best_epoch": 0, "best_mean_r": best, "delta_mean_r_vs_epoch0": 0.}]
    def save(name, epoch):
        dest = out / name
        temp = dest.with_suffix(".pending")
        torch.save({"protocol": PROTOCOL, "config": config, "epoch": epoch,
            "updates": updates, "best_epoch": best_epoch, "best_mean_r": best,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "inference": "original base II + raw history B/O16 -> delta; project base then propagate [1,1,-.5,-.5,1]",
            "source_run": str(source), "cache": str(root)}, temp)
        temp.replace(dest)
    save("best.pt", 0)
    print({"phase": "training", "objective": args.objective, **history[0]}, flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0., 0
        for batch in loader:
            x = [v.to(device) for v in batch]
            output = model(x[0], x[1], x[4], x[5], x[6])
            predicted = joint_prediction(output, x[1], x[2], x[9])
            loss = correlation_loss(predicted, x[3], x[7], x[8], output - x[1])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total += float(loss.detach()) * len(output)
            count += len(output)
        model.eval()
        score = assess(engine, cache, root, source, out / f"validation/epoch_{epoch:03d}")
        if score["mean_r"] > best:
            best, best_epoch = score["mean_r"], epoch
            save("best.pt", epoch)
        save("last.pt", epoch)
        row = {"epoch": epoch, "updates": updates, "train_loss": total / count,
               **score, "best_epoch": best_epoch, "best_mean_r": best,
               "delta_mean_r_vs_epoch0": score["mean_r"] - initial["mean_r"]}
        history.append(row)
        csv_write(out / "learning_curve.csv", history)
        print({"phase": "training", "objective": args.objective, **row}, flush=True)
    write(out / "completed.json", {"completed": True, "objective": args.objective,
        "best_epoch": best_epoch, "best_mean_r": best, "epoch0_mean_r": initial["mean_r"],
        "gain_over_frozen_C3": best - initial["mean_r"], "last_mean_r": history[-1]["mean_r"],
        "adapter_updates": updates, "FM_updates": 0, "slow_updates": 0,
        "best_checkpoint_sha256": sha(out / "best.pt"), "history_reference_recheck_deferred": True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--source-run", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--resume", action="store_true")
    prep.add_argument("--device", default="cuda")
    tr = subs.add_parser("train")
    tr.add_argument("--cache", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--objective", choices=("II", "five"), default="five")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--learning-rate", type=float, default=2e-4)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--device", default="cuda")
    pipe = subs.add_parser("pipeline")
    pipe.add_argument("--source-run", required=True)
    pipe.add_argument("--output", required=True)
    pipe.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    else:
        out = Path(args.output).resolve()
        if out.exists():
            raise FileExistsError("Use a new pipeline directory")
        out.mkdir(parents=True)
        base = [sys.executable, "-u", "-m", "baselines.B5_final.c3_joint"]
        subprocess.run(base + ["prepare", "--source-run", args.source_run,
            "--output", str(out / "cache"), "--device", args.device], check=True)
        for objective in ("five", "II"):
            subprocess.run(base + ["train", "--cache", str(out / "cache"),
                "--output", str(out / ("train_" + objective)), "--objective", objective,
                "--device", args.device], check=True)
        write(out / "completed.json", {"completed": True,
            "five": read(out / "train_five/completed.json"),
            "II_control": read(out / "train_II/completed.json")})


if __name__ == "__main__":
    main()
