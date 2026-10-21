"""Matched single/multi/no-history C3 continuation and prespecified history controls."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from baselines.B5_best.common import read, write, csv_write, sha
from baselines.B5_final.c3_joint import check_manifest, joint_prediction, correlation_loss, assert_disjoint
from baselines.B5_best_context.body_multitemplate import cluster_bank, current_queries, mix_templates, K, TEMPERATURE

PROTOCOL = "B5-C3-history-multitemplate-v1"
SHUFFLES = (101, 202, 303)


def prepare(args):
    from baselines.B5.config import load_config
    from baselines.B5.data import common_config
    from ecg12gen.dataset import JointAnchorDataset
    from baselines.B5_best_context.adapters import history_from_sample, _visible_digest
    from baselines.B5_best_context.body_features import relation_features, BEAT16_ENCODING
    from baselines.B5_best_context.body_history import process_history, B_POLICY
    from baselines.B5_best_context.body_study import shuffled_donors
    c3 = Path(args.c3_run).resolve()
    oldroot = c3 / "cache"
    source = check_manifest(oldroot)
    source_run = Path(source["frozen"]["source_run"])
    init = c3 / "train_five/best.pt"
    if sha(init) != read(c3 / "train_five/completed.json")["best_checkpoint_sha256"]:
        raise ValueError("C3 initial checkpoint changed")
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Use a new feature directory")
    out.mkdir(parents=True)
    cfg_path = source_run / "inputs/validation.json"
    common = common_config(load_config(cfg_path, data_only=True))
    records = []
    for split in ("train", "validation"):
        ds = JointAnchorDataset(common, "task2", split)
        try:
            rows = [ds._rows[i] for i in ds._indices]
            lookup = {(r["pair_id"], int(r["start_sample_500hz"])): i for i, r in enumerate(rows)}
            for row in source["records"]:
                if row["split"] != split:
                    continue
                sample = ds[lookup[(row["pair_id"], 0)]]
                raw_history = history_from_sample(sample)
                if _visible_digest(raw_history) != row["history_visible_sha256"]:
                    raise ValueError("Visible historical input changed")
                processed = process_history(raw_history, B_POLICY)
                bank = {}
                beat, stats, available = relation_features(processed, encoding=BEAT16_ENCODING, beat_bank=bank)
                with np.load(oldroot / row["file"], allow_pickle=False) as old:
                    for left, right in ((beat, old["beat"]), (stats, old["stats"])):
                        if not np.array_equal(left, right):
                            raise ValueError("Single-template control did not preserve original features")
                    if available != bool(old["available"]):
                        raise ValueError("History availability changed")
                    queries, query_valid = current_queries(old["anchor"])
                templates, counts, dispersion = cluster_bank(bank["beats"])
                multi, weights = mix_templates(queries, query_valid, templates, counts, dispersion)
                relative = row["file"]
                dest = out / relative
                dest.parent.mkdir(exist_ok=True)
                with dest.open("wb") as f:
                    np.savez_compressed(f, single=beat, stats=stats, available=np.array(available),
                        templates=templates, counts=counts, dispersion=dispersion,
                        queries=queries, query_valid=query_valid, multi=multi, weights=weights)
                records.append({**row, "feature_sha256": sha(dest),
                    "active_templates": int((counts > 0).sum()),
                    "query_windows_valid": int(query_valid.sum()),
                    "history_beats": int(counts.sum())})
                print({"phase": "features", "split": split,
                       "completed": sum(r["split"] == split for r in records)}, flush=True)
        finally:
            ds.close()
    assert_disjoint(records)
    validation = [r for r in records if r["split"] == "validation"]
    donors = {str(seed): {subject: row["pair_id"] for subject, row in shuffled_donors(validation, seed).items()}
              for seed in SHUFFLES}
    for seed, mapping in donors.items():
        by_pair = {r["pair_id"]: r for r in validation}
        if any(by_pair[pair]["subject_id"] == subject for subject, pair in mapping.items()):
            raise AssertionError("A shuffled donor retained its recipient's patient")
    manifest = {"protocol": PROTOCOL, "completed": True, "records": records,
        "old_cache": str(oldroot), "old_cache_sha256": sha(oldroot / "cache.json"),
        "source_run": str(source_run), "initial_checkpoint": str(init), "initial_checkpoint_sha256": sha(init),
        "K": K, "temperature": TEMPERATURE, "shuffled_donors": donors,
        "grouping_and_matching": "visible historical I shapes and current I only; no beat-index alignment",
        "stats": "unchanged O16 record statistics", "history_filter": B_POLICY,
        "historical_reference_recheck_deferred": True}
    write(out / "features.json", manifest)
    csv_write(out / "feature_diagnostics.csv", [{k: r[k] for k in
        ("split", "pair_id", "active_templates", "query_windows_valid", "history_beats")} for r in records])


def load_features(root):
    manifest = read(root / "features.json")
    if manifest.get("protocol") != PROTOCOL or not manifest.get("completed"):
        raise ValueError("Incomplete historical feature cache")
    oldroot = Path(manifest["old_cache"])
    if sha(oldroot / "cache.json") != manifest["old_cache_sha256"]:
        raise ValueError("Base training cache changed")
    check_manifest(oldroot)
    assert_disjoint(manifest["records"])
    if sha(manifest["initial_checkpoint"]) != manifest["initial_checkpoint_sha256"]:
        raise ValueError("Initial C3 checkpoint changed")
    features, arrays = {}, {}
    for row in manifest["records"]:
        if sha(root / row["file"]) != row["feature_sha256"]:
            raise ValueError("Historical features changed")
        with np.load(root / row["file"], allow_pickle=False) as f:
            features[row["pair_id"]] = {k: f[k].copy() for k in f.files}
        with np.load(oldroot / row["file"], allow_pickle=False) as f:
            arrays[row["pair_id"]] = {k: f[k].copy() for k in f.files}
    return manifest, features, arrays


def visible_condition(row, features, manifest, representation, mode):
    """No targets enter feature choice. Shuffled donors are fixed by subject, not score."""
    recipient = features[row["pair_id"]]
    if mode == "disabled" or representation == "no_history":
        return np.zeros((12, 2, 300), np.float32), np.zeros(27, np.float32), False
    donor = recipient
    if mode.startswith("shuffled_"):
        donor = features[manifest["shuffled_donors"][mode.split("_")[1]][row["subject_id"]]]
    if representation == "single":
        beat = np.repeat(donor["single"][None], 12, axis=0)
    elif representation == "multi":
        if donor is recipient:
            beat = recipient["multi"]
        else:
            beat, _ = mix_templates(recipient["queries"], recipient["query_valid"],
                donor["templates"], donor["counts"], donor["dispersion"])
    else:
        raise ValueError("Unknown representation")
    return beat, donor["stats"], bool(donor["available"])


class Evaluator:
    def __init__(self, manifest, features, arrays, device):
        from baselines.B5_best_context.evaluate import load_baseline
        source = Path(manifest["source_run"])
        self.loaded = load_baseline(source / "base_inference")
        self.item = self.loaded[1]["task2"]
        self.lookup = {self.item["rows"][ids[0]]["pair_id"]: ids for ids in self.item["pairs"]}
        fixed = np.load(source / "study/C3_projection_five/task2/prediction_uV.npy", mmap_mode="r")
        self.fixed = np.array(fixed, copy=True)
        fixed._mmap.close()
        self.manifest, self.features, self.arrays, self.device = manifest, features, arrays, device

    def close(self):
        for item in self.loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()

    def run(self, model, representation, mode, output=None):
        import torch
        from baselines.B5_best_context.body_features import encode_windows, restore_ii
        from baselines.B5_final.body_scope import transform
        from baselines.B5_best_context.evaluate import _score, _record_metrics
        prediction = self.fixed.copy()
        model.eval()
        metrics = []
        with torch.no_grad():
            for row in self.manifest["records"]:
                if row["split"] != "validation":
                    continue
                a = self.arrays[row["pair_id"]]
                beat, stats, enabled = visible_condition(row, self.features, self.manifest, representation, mode)
                signal, base, scales = encode_windows(a["anchor"], a["base_ii"])
                x = [signal, base, beat, np.repeat(stats[None], 12, axis=0), np.full(12, enabled, np.float32)]
                value = model(*(torch.from_numpy(np.asarray(v, np.float32).copy()).to(self.device) for v in x))
                corrected = restore_ii(a["base_ii"], base, value.cpu().numpy(), scales)
                ids = self.lookup[row["pair_id"]]
                original = np.concatenate([self.item["prediction"][i] for i in ids], axis=-1)
                result = transform(original, a["anchor"], corrected, projection=True, scope="five")
                for j, index in enumerate(ids):
                    prediction[index] = result[:, j * 5000:(j + 1) * 5000]
                truth = np.concatenate([self.item["target"][i] for i in ids], axis=-1)
                metrics.append(_record_metrics(result, truth, self.item["quality"][ids[0]], self.item["rows"][ids[0]], "task2"))
        overall, details = _score(prediction, self.item, "task2")
        task1 = self.loaded[0]["score"]["task1_r_missing11"]
        score = {"task1_r": task1, "machine_r": overall["ecg_machine_d6_r_missing11"],
            "body_r": overall["body_scale_d6_r_missing11"], "task2_r": overall["r_missing11"],
            "mean_r": (task1 + overall["r_missing11"]) / 2}
        if output is not None:
            from ecg12gen.evaluate import write_report
            output.mkdir(parents=True, exist_ok=False)
            write_report(output, overall, details, title=f"C3 {representation} history {mode}")
            csv_write(output / "pair_metrics.csv", metrics)
            write(output / "score.json", score)
        return score


def train(args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from baselines.B5.runtime import seed_all, device_from_name
    from baselines.B5_best_context.body_model import BodyIIAdapter
    from baselines.B5_best_context.body_features import encode_windows
    root, out = Path(args.features).resolve(), Path(args.output).resolve()
    manifest, features, arrays = load_features(root)
    if out.exists():
        raise FileExistsError("Use new training output")
    if not 0 <= args.seed < 2**32:
        raise ValueError("Training seed must be a uint32")
    seed_all(args.seed, True)
    device = device_from_name(args.device)
    model = BodyIIAdapter().to(device)
    initial = torch.load(manifest["initial_checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(initial["model"], strict=True)
    records = [r for r in manifest["records"] if r["split"] == "train" and r["eligible"]]
    counts = Counter(r["target_record_id"] for r in records)
    values = []
    for row in records:
        a = arrays[row["pair_id"]]
        beat, stats, enabled = visible_condition(row, features, manifest, args.representation, "correct")
        signal, base, scales = encode_windows(a["anchor"], a["base_ii"])
        for j in range(12):
            if scales[j] <= 0:
                continue
            target = a["target_limb"][:, j * 5000:(j + 1) * 5000] / scales[j]
            valid = a["quality"][1:6] & (np.std(target.astype(float), axis=-1) > 0)
            if not valid.any():
                continue
            values.append((signal[j], base[j], a["projected"][:, j * 5000:(j + 1) * 5000] / scales[j],
                target, beat[j], stats, np.float32(enabled), valid,
                np.float32(1 / counts[row["target_record_id"]]), scales[j]))
    tensors = [torch.from_numpy(np.stack([v[k] for v in values]).copy()) for k in range(10)]
    data = TensorDataset(*tensors)
    loader = DataLoader(data, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    config = {"protocol": PROTOCOL, "representation": args.representation, "seed": args.seed,
        "epochs": 10, "batch_size": 16, "learning_rate": 2e-4, "features_sha256": sha(root / "features.json"),
        "initial_checkpoint_sha256": manifest["initial_checkpoint_sha256"], "training_windows": len(data),
        "loss": "unchanged C3 five-limb QC PCC + .01 residual L2", "FM_updates": 0, "slow_updates": 0,
        "K": K, "temperature": TEMPERATURE, "selection": "full120s correct-history mean_r including epoch0",
        "history_preprocessing": manifest["history_filter"], "source_run": manifest["source_run"]}
    out.mkdir(parents=True)
    write(out / "config.json", config)
    evaluator = Evaluator(manifest, features, arrays, device)
    try:
        first = evaluator.run(model, args.representation, "correct", out / "validation/epoch_000")
        expected = read(Path(manifest["initial_checkpoint"]).parent / "completed.json")["best_mean_r"]
        if args.representation == "single" and abs(first["mean_r"] - expected) > 1e-10:
            raise ValueError("Single-template initialization does not exactly reproduce trained C3")
        best, best_epoch, updates = first["mean_r"], 0, 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
        def save(name, epoch):
            path = out / name
            temp = path.with_suffix(".pending")
            torch.save({"protocol": PROTOCOL, "config": config, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "epoch": epoch, "updates": updates,
                "best_epoch": best_epoch, "best_mean_r": best}, temp)
            temp.replace(path)
        save("best.pt", 0)
        curve = [{"epoch": 0, "updates": 0, "train_loss": None, **first,
            "best_mean_r": best, "best_epoch": 0}]
        print({"phase": "training", "representation": args.representation, **curve[0]}, flush=True)
        for epoch in range(1, 11):
            model.train()
            total, count = 0., 0
            for batch in loader:
                x = [v.to(device) for v in batch]
                value = model(x[0], x[1], x[4], x[5], x[6])
                predicted = joint_prediction(value, x[1], x[2], x[9])
                loss = correlation_loss(predicted, x[3], x[7], x[8], value - x[1])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                total += float(loss.detach()) * len(value); count += len(value)
            result = evaluator.run(model, args.representation, "correct", out / f"validation/epoch_{epoch:03d}")
            if result["mean_r"] > best:
                best, best_epoch = result["mean_r"], epoch
                save("best.pt", epoch)
            save("last.pt", epoch)
            row = {"epoch": epoch, "updates": updates, "train_loss": total / count,
                **result, "best_mean_r": best, "best_epoch": best_epoch}
            curve.append(row)
            csv_write(out / "learning_curve.csv", curve)
            print({"phase": "training", "representation": args.representation, **row}, flush=True)
        # Evaluate all prespecified history controls on BOTH best and fixed-last checkpoints.
        controls = []
        modes = ("correct",) if args.representation == "no_history" else (
            "correct", "disabled", *[f"shuffled_{s}" for s in SHUFFLES])
        for name in ("best", "last"):
            payload = torch.load(out / (name + ".pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            for mode in modes:
                score = evaluator.run(model, args.representation, mode, out / "controls" / name / mode)
                controls.append({"checkpoint": name, "epoch": payload["epoch"],
                    "representation": args.representation, "history_mode": mode, **score})
        csv_write(out / "history_controls.csv", controls)
        write(out / "completed.json", {"completed": True, "representation": args.representation,
            "best_epoch": best_epoch, "best_mean_r": best, "epoch0_mean_r": first["mean_r"],
            "fixed_C3_mean_r": expected, "delta_vs_fixed_C3": best - expected,
            "last_mean_r": curve[-1]["mean_r"], "updates": updates, "controls": controls,
            "best_checkpoint_sha256": sha(out / "best.pt"), "FM_updates": 0, "slow_updates": 0})
    finally:
        evaluator.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--c3-run", required=True)
    prep.add_argument("--output", required=True)
    tr = subs.add_parser("train")
    tr.add_argument("--features", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--representation", choices=("single", "multi", "no_history"), required=True)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--seed", type=int, default=42, help="Training/shuffle seed only; frozen FM inference stays seed42")
    pipe = subs.add_parser("pipeline")
    pipe.add_argument("--c3-run", required=True)
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
            raise FileExistsError("Use new pipeline output")
        out.mkdir(parents=True)
        command = [sys.executable, "-u", "-m", "baselines.B5_final.body_history_study"]
        subprocess.run(command + ["prepare", "--c3-run", args.c3_run, "--output", str(out / "features")], check=True)
        for representation in ("single", "multi", "no_history"):
            subprocess.run(command + ["train", "--features", str(out / "features"),
                "--output", str(out / representation), "--representation", representation,
                "--device", args.device], check=True)
        summaries = [read(out / group / "completed.json") for group in ("single", "multi", "no_history")]
        csv_write(out / "summary.csv", [{k: r[k] for k in ("representation", "best_epoch", "best_mean_r",
            "epoch0_mean_r", "fixed_C3_mean_r", "delta_vs_fixed_C3", "last_mean_r")} for r in summaries])
        csv_write(out / "history_controls.csv", [row for r in summaries for row in r["controls"]])
        write(out / "completed.json", {"completed": True, "groups": summaries,
            "interpretation": "Compare multi vs single plus correct/disabled/all shuffled/no-history; no target-selected template or donor."})


if __name__ == "__main__":
    main()
