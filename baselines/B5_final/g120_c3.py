"""Three-arm G120/C3 comparison with G120 frozen for every body experiment.

Arm A is the deployed G120 route with the original body adapter.  Arm B applies
an existing C3 checkpoint to the same G120 base.  Arm C trains a fresh C3
adapter on G120-corrected training predictions and selects it by the complete
Task1/Task2 mean-r score.  No FM/P/G120 parameter is updated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from baselines.B5.scoring import scores_from_summaries, write_score_report
from baselines.B5_best.common import csv_write, read, sha, write
from baselines.B5_best_context.evaluate import load_baseline, pairs, _record_metrics, _score
from baselines.B5_final.body_scope import transform
from baselines.B5_final.c3_joint import assert_disjoint, correlation_loss, joint_prediction, project
from baselines.B5_final.profile import BASE_SHA, BODY_SHA

PROTOCOL = "B5-G120-C3-three-arm-v1"
C3_PROTOCOLS = {
    "B5-C3-five-limb-continuation-v1",
    "B5-C3-history-multitemplate-v1",
    PROTOCOL,
}


def _slow_record_indices(study):
    validation, training = {}, {}
    for row in study["validation"]:
        key = row["task"], row["pair_id"]
        if key in validation:
            raise ValueError("Duplicate slow120 validation pair: " + repr(key))
        validation[key] = row
    for row in study["train"]:
        key = row["target_record_id"]
        if key in training:
            raise ValueError("Duplicate slow120 training target: " + key)
        training[key] = row
    return validation, training


def _checkpoint_payload(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") not in C3_PROTOCOLS or not isinstance(payload.get("model"), dict):
        raise ValueError("Checkpoint is not a supported C3 adapter")
    return payload


def _load_slow(slow_run, checkpoint, device):
    from baselines.B5_slow120.common import load_correction, load_study
    root = Path(slow_run).resolve()
    study = load_study(root)
    model, payload = load_correction(checkpoint, study, root / "study.json", device)
    if payload.get("group") != "G120":
        raise ValueError("The combined study requires a G120 checkpoint")
    model.eval()
    return root, study, model, payload


def _corrected_base(data, model, scales, device):
    import torch
    from baselines.B5_slow120.evaluate import array_batch
    from baselines.B5_slow120.model import correct_base, inputs
    with torch.no_grad():
        delta = model(**inputs(array_batch(data, device)))[0].cpu().numpy()
    if not np.isfinite(delta).all():
        raise FloatingPointError("Nonfinite G120 correction")
    return correct_base(data["base_uV"], delta, scales)


def _group_a_files(folder):
    folder = Path(folder).resolve()
    report = read(folder / "evaluation.json")
    required = {task: folder / task / "prediction_uV.npy" for task in ("task1", "task2")}
    if (not report.get("completed") or report.get("summary", {}).get("group") != "G120"
            or not report.get("predictions_retained") or any(not p.is_file() for p in required.values())):
        raise ValueError("Arm A must be a completed G120 evaluation with retained predictions")
    return folder, report, required


def _frozen_manifest(study_root, study, checkpoint, correction, group_a):
    folder, report, predictions = _group_a_files(group_a)
    if (report.get("population") != study["population"]
            or report["summary"].get("epoch") != correction.get("epoch")):
        raise ValueError("Arm A does not match the G120 checkpoint/study population")
    return {
        "slow_run": str(study_root),
        "slow_study_sha256": sha(study_root / "study.json"),
        "slow_checkpoint": str(Path(checkpoint).resolve()),
        "slow_checkpoint_sha256": sha(checkpoint),
        "slow_group": correction["group"],
        "slow_epoch": correction["epoch"],
        "base_sha256": study["base_sha256"],
        "body_sha256": study["body_sha256"],
        "body_checkpoint": study["body_checkpoint"],
        "config": study["config"],
        "config_sha256": study["config_sha256"],
        "baseline_evaluation": study["baseline_evaluation"],
        "population": study["population"],
        "scale_uV": study["scale_uV"],
        "group_a_evaluation": str(folder),
        "group_a_report_sha256": sha(folder / "evaluation.json"),
        "group_a_prediction_sha256": {task: sha(path) for task, path in predictions.items()},
        "group_a_summary": report["summary"],
    }


def check_cache(root):
    root = Path(root).resolve()
    manifest = read(root / "cache.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("completed") is not True:
        raise ValueError("Incomplete or incompatible G120/C3 cache")
    assert_disjoint(manifest["records"])
    frozen = manifest["frozen"]
    checks = {
        Path(frozen["slow_run"]) / "study.json": frozen["slow_study_sha256"],
        Path(frozen["slow_checkpoint"]): frozen["slow_checkpoint_sha256"],
        Path(frozen["group_a_evaluation"]) / "evaluation.json": frozen["group_a_report_sha256"],
    }
    checks.update({Path(frozen["group_a_evaluation"]) / task / "prediction_uV.npy": digest
                   for task, digest in frozen["group_a_prediction_sha256"].items()})
    for path, digest in checks.items():
        if sha(path) != digest:
            raise ValueError("Frozen G120 source changed: " + str(path))
    for row in manifest["records"]:
        path = root / row["file"]
        if not path.resolve().is_relative_to(root) or sha(path) != row["sha256"]:
            raise ValueError("Changed G120/C3 cache record: " + row["file"])
    return manifest


def prepare(args):
    from baselines.B5.config import load_config
    from baselines.B5.data import common_config
    from baselines.B5.runtime import device_from_name
    from baselines.B5_best_context.adapters import history_from_sample, _visible_digest
    from baselines.B5_best_context.body_features import feature_encoding, relation_features
    from baselines.B5_best_context.body_history import history_policy, process_history
    from baselines.B5_best_context.body_model import BodyAdapter
    from ecg12gen.dataset import JointAnchorDataset

    out = Path(args.output).resolve()
    device = device_from_name(args.device)
    slow_root, study, correction_model, correction = _load_slow(
        args.slow_run, args.slow_checkpoint, device)
    frozen = _frozen_manifest(slow_root, study, args.slow_checkpoint, correction,
                              args.group_a_evaluation)
    if study["base_sha256"] != BASE_SHA or study["body_sha256"] != BODY_SHA:
        raise ValueError("G120 study does not use the pinned B5-final assets")
    if sha(study["body_checkpoint"]) != BODY_SHA:
        raise ValueError("Pinned body checkpoint content changed")
    baseline = load_baseline(study["baseline_evaluation"])
    if baseline[2] != study["population"]:
        raise ValueError("Plain baseline population changed since slow120 training")
    body = BodyAdapter(study["body_checkpoint"], BASE_SHA, device, expected_group="context")
    policy, encoding = history_policy(body.payload), feature_encoding(body.payload)
    frozen.update(history_preprocessing=policy, feature_encoding=encoding)
    validation_entries, train_entries = _slow_record_indices(study)
    ledger = out / "cache.json"
    if ledger.exists():
        manifest = read(ledger)
        if not args.resume or manifest.get("frozen") != frozen:
            raise ValueError("Resume changed a frozen source or --resume was omitted")
    else:
        if out.exists():
            raise FileExistsError("Use a new cache output")
        out.mkdir(parents=True)
        manifest = {"protocol": PROTOCOL, "completed": False, "frozen": frozen, "records": []}
        write(ledger, manifest)
    existing = {(row["split"], row["pair_id"]): row for row in manifest["records"]}
    cfg = load_config(study["config"], data_only=True)
    common = common_config(cfg)
    datasets = []
    try:
        for split in ("validation", "train"):
            dataset = JointAnchorDataset(common, "task2", split)
            datasets.append(dataset)
            rows = [dataset._rows[i] for i in dataset._indices]
            for indices in pairs(rows):
                row = rows[indices[0]]
                if row["input_type"] != "body_scale_d6":
                    continue
                key = split, row["pair_id"]
                if key in existing:
                    cached = existing[key]
                    if sha(out / cached["file"]) != cached["sha256"]:
                        raise ValueError("Changed partial cache record")
                    continue
                source = (validation_entries.get(("task2", row["pair_id"])) if split == "validation"
                          else train_entries.get(row["target_record_id"]))
                if source is None:
                    raise ValueError(f"No slow120 {split} record for {row['pair_id']}")
                samples = [dataset[i] for i in indices]
                anchor = np.concatenate([sample.anchor_i_ecg for sample in samples], axis=-1).astype(np.float32)
                target = np.concatenate([sample.Y_12lead[1:6] for sample in samples], axis=-1).astype(np.float32)
                quality = np.asarray(samples[0].target_quality_mask, bool)
                if any(not np.array_equal(sample.target_quality_mask, quality) for sample in samples):
                    raise ValueError("QC changed within a physical record")
                with np.load(slow_root / source["file"], allow_pickle=False) as slow:
                    if (not np.array_equal(anchor, slow["anchor_uV"])
                            or not np.array_equal(target, slow["target_uV"][1:6])
                            or not np.array_equal(quality, slow["mask"])):
                        raise ValueError("C3 data differ from the frozen slow120 record")
                    base = _corrected_base(slow, correction_model, study["scale_uV"], device)
                history = history_from_sample(samples[0])
                processed = process_history(history, policy)
                beat, stats, available = relation_features(processed, encoding=encoding)
                relative = split + "/" + hashlib.sha256(row["pair_id"].encode()).hexdigest()[:24] + ".npz"
                path = out / relative
                path.parent.mkdir(exist_ok=True)
                pending = path.with_suffix(".pending")
                with pending.open("wb") as handle:
                    np.savez_compressed(handle, anchor=anchor[0], base_uV=base, base_ii=base[1],
                        projected=project(base, anchor), target_limb=target, quality=quality,
                        beat=beat, stats=stats, available=np.array(available))
                pending.replace(path)
                manifest["records"].append({"split": split, "pair_id": row["pair_id"],
                    "subject_id": row["subject_id"], "target_record_id": row["target_record_id"],
                    "history_visible_sha256": _visible_digest(history), "file": relative,
                    "sha256": sha(path), "history_available": bool(available),
                    "eligible": bool(quality[0] and quality[1:6].any())})
                assert_disjoint(manifest["records"])
                write(ledger, manifest)
                print({"phase": "prepare_g120_c3", "split": split,
                       "cached_records": sum(x["split"] == split for x in manifest["records"])}, flush=True)
        if any(not any(row["split"] == split and row["eligible"] for row in manifest["records"])
               for split in ("train", "validation")):
            raise ValueError("No eligible G120/C3 records in one split")
        manifest["completed"] = True
        write(ledger, manifest)
        check_cache(out)
        write(out / "completed.json", {"completed": True,
            "counts": dict(Counter(row["split"] for row in manifest["records"])),
            "G120_updates": 0, "FM_updates": 0, "P_updates": 0})
    finally:
        for dataset in datasets:
            dataset.close()
        for item in baseline[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()


def _adapter(cache, device, checkpoint=None):
    from baselines.B5_best_context.body_model import BodyAdapter
    engine = BodyAdapter(cache["frozen"]["body_checkpoint"], BASE_SHA, device, expected_group="context")
    if checkpoint is not None:
        payload = _checkpoint_payload(checkpoint)
        engine.model.load_state_dict(payload["model"], strict=True)
        engine.model.eval()
        engine.model.requires_grad_(False)
        return engine, payload
    return engine, {"protocol": engine.payload["protocol"], "epoch": engine.payload["epoch"]}


class Evaluator:
    def __init__(self, cache_root, cache, device):
        self.root, self.cache, self.device = Path(cache_root), cache, device
        self.loaded = load_baseline(cache["frozen"]["baseline_evaluation"])
        if self.loaded[2] != cache["frozen"]["population"]:
            raise ValueError("Validation population changed")
        group_a = Path(cache["frozen"]["group_a_evaluation"])
        self.fixed = {task: np.load(group_a / task / "prediction_uV.npy", mmap_mode="r")
                      for task in ("task1", "task2")}
        for task, array in self.fixed.items():
            if array.shape != self.loaded[1][task]["prediction"].shape:
                raise ValueError("Arm A prediction shape changed: " + task)
        self.lookup = {self.loaded[1]["task2"]["rows"][ids[0]]["pair_id"]: ids
                       for ids in self.loaded[1]["task2"]["pairs"]}
        self.task1_overall, self.task1_details = _score(self.fixed["task1"], self.loaded[1]["task1"], "task1")

    def close(self):
        for array in self.fixed.values():
            array._mmap.close()
        for item in self.loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()

    def run(self, engine, label, output=None, *, keep_predictions=False):
        item = self.loaded[1]["task2"]
        prediction = np.array(self.fixed["task2"], copy=True)
        metrics = []
        for row in self.cache["records"]:
            if row["split"] != "validation":
                continue
            ids = self.lookup[row["pair_id"]]
            with np.load(self.root / row["file"], allow_pickle=False) as arrays:
                corrected, _, _ = engine.predict_ii_features(arrays["anchor"], arrays["base_ii"],
                    arrays["beat"], arrays["stats"], arrays["available"])
                result = transform(arrays["base_uV"], arrays["anchor"], corrected,
                                   projection=True, scope="five")
                if not np.array_equal(result[[0, *range(6, 12)]], arrays["base_uV"][[0, *range(6, 12)]]):
                    raise AssertionError("C3 changed I or chest leads")
                for j, index in enumerate(ids):
                    prediction[index] = result[:, j * 5000:(j + 1) * 5000]
                truth = np.concatenate([item["target"][i] for i in ids], axis=-1)
                metrics.append(_record_metrics(result, truth, item["quality"][ids[0]],
                                                item["rows"][ids[0]], "task2"))
        task2_overall, task2_details = _score(prediction, item, "task2")
        score = scores_from_summaries({"task1": self.task1_overall, "task2": task2_overall})
        row = {"group": label, "task1_r": score["task1_r_missing11"],
            "machine_r": task2_overall["ecg_machine_d6_r_missing11"],
            "body_r": task2_overall["body_scale_d6_r_missing11"],
            "task2_r": score["task2_r_missing11"], "mean_r": score["main_correlation"]}
        if output is not None:
            from ecg12gen.evaluate import write_report
            output = Path(output)
            output.mkdir(parents=True, exist_ok=False)
            write_report(output / "task1", self.task1_overall, self.task1_details,
                         title=f"{label} Task1 (fixed G120)")
            write_report(output / "task2", task2_overall, task2_details, title=f"{label} Task2")
            write_score_report(output, score)
            csv_write(output / "pair_metrics.csv", metrics)
            write(output / "score.json", row)
            if keep_predictions:
                np.save(output / "task1/prediction_uV.npy", np.asarray(self.fixed["task1"]), allow_pickle=False)
                np.save(output / "task2/prediction_uV.npy", prediction, allow_pickle=False)
        return row


def evaluate(args):
    from baselines.B5.runtime import device_from_name
    root = Path(args.cache).resolve()
    cache = check_cache(root)
    device = device_from_name(args.device)
    engine, payload = _adapter(cache, device, args.checkpoint)
    evaluator = Evaluator(root, cache, device)
    try:
        row = evaluator.run(engine, args.label, args.output, keep_predictions=args.keep_predictions)
        write(Path(args.output) / "checkpoint.json", {"path": str(Path(args.checkpoint).resolve()),
            "sha256": sha(args.checkpoint), "protocol": payload["protocol"], "epoch": payload.get("epoch")})
        print(row, flush=True)
    finally:
        evaluator.close()


def train(args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from baselines.B5.runtime import device_from_name, seed_all
    from baselines.B5_best_context.body_features import encode_windows

    root, out = Path(args.cache).resolve(), Path(args.output).resolve()
    cache = check_cache(root)
    if out.exists():
        raise FileExistsError("Use a new training directory")
    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Training settings must be positive and seed must be uint32")
    seed_all(args.seed, True)
    device = device_from_name(args.device)
    engine, _ = _adapter(cache, device)
    model = engine.model
    model.requires_grad_(True)
    records = [row for row in cache["records"] if row["split"] == "train" and row["eligible"]]
    counts = Counter(row["target_record_id"] for row in records)
    values = []
    for row in records:
        with np.load(root / row["file"], allow_pickle=False) as arrays:
            signal, base, scales = encode_windows(arrays["anchor"], arrays["base_ii"])
            for j in range(12):
                if scales[j] <= 0:
                    continue
                target = arrays["target_limb"][:, j * 5000:(j + 1) * 5000] / scales[j]
                valid = arrays["quality"][1:6] & (np.std(target.astype(float), axis=-1) > 0)
                if not valid.any():
                    continue
                values.append((signal[j], base[j],
                    arrays["projected"][:, j * 5000:(j + 1) * 5000] / scales[j], target,
                    arrays["beat"], arrays["stats"], np.float32(arrays["available"]), valid,
                    np.float32(1 / counts[row["target_record_id"]]), np.float32(scales[j])))
    if not values:
        raise ValueError("No usable G120/C3 training windows")
    tensors = [torch.from_numpy(np.stack([row[k] for row in values]).copy()) for k in range(10)]
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    out.mkdir(parents=True)
    config = {"protocol": PROTOCOL, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seed": args.seed, "cache_sha256": sha(root / "cache.json"),
        "slow_checkpoint_sha256": cache["frozen"]["slow_checkpoint_sha256"],
        "initial_body_sha256": cache["frozen"]["body_sha256"],
        "initialization": "pinned original body.pt; G120 base frozen",
        "loss": "five-limb QC PCC + .01 residual L2; inverse repeated-target weight",
        "selection": "complete G120 Task1/Task2 mean_r including epoch0",
        "FM_updates": 0, "P_updates": 0, "G120_updates": 0, "training_windows": len(values)}
    write(out / "config.json", config)
    evaluator = Evaluator(root, cache, device)
    try:
        model.eval()
        initial = evaluator.run(engine, "G120+C3_epoch0", out / "validation/epoch_000")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
        best, best_epoch, updates = initial["mean_r"], 0, 0
        history = [{"epoch": 0, "updates": 0, "train_loss": None, **initial,
                    "best_epoch": 0, "best_mean_r": best}]

        def save(name, epoch):
            path = out / name
            pending = path.with_suffix(".pending")
            torch.save({"protocol": PROTOCOL, "config": config, "epoch": epoch,
                "updates": updates, "best_epoch": best_epoch, "best_mean_r": best,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "cache": str(root), "inference": "G120 base -> fixed-I projection -> five-limb C3"}, pending)
            pending.replace(path)

        save("best.pt", 0)
        print({"phase": "train_g120_c3", **history[0]}, flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = count = 0
            for batch in loader:
                x = [value.to(device) for value in batch]
                output = model(x[0], x[1], x[4], x[5], x[6])
                predicted = joint_prediction(output, x[1], x[2], x[9])
                loss = correlation_loss(predicted, x[3], x[7], x[8], output - x[1])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite G120/C3 loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                total += float(loss.detach()) * len(output)
                count += len(output)
            model.eval()
            result = evaluator.run(engine, f"G120+new_C3_epoch{epoch}",
                                   out / f"validation/epoch_{epoch:03d}")
            if result["mean_r"] > best:
                best, best_epoch = result["mean_r"], epoch
                save("best.pt", epoch)
            save("last.pt", epoch)
            row = {"epoch": epoch, "updates": updates, "train_loss": total / count, **result,
                   "best_epoch": best_epoch, "best_mean_r": best,
                   "delta_mean_r_vs_epoch0": result["mean_r"] - initial["mean_r"]}
            history.append(row)
            csv_write(out / "learning_curve.csv", history)
            print({"phase": "train_g120_c3", **row}, flush=True)
        selected = _checkpoint_payload(out / "best.pt")
        model.load_state_dict(selected["model"], strict=True)
        model.eval()
        final = evaluator.run(engine, "G120+new_C3", out / "selected_best", keep_predictions=args.keep_predictions)
        if abs(final["mean_r"] - best) > 1e-10:
            raise ValueError("Selected G120/C3 checkpoint did not reproduce its score")
        write(out / "completed.json", {"completed": True, "best_epoch": best_epoch,
            "best_mean_r": best, "epoch0_mean_r": initial["mean_r"],
            "gain_over_epoch0": best - initial["mean_r"], "last_mean_r": history[-1]["mean_r"],
            "adapter_updates": updates, "FM_updates": 0, "P_updates": 0, "G120_updates": 0,
            "best_checkpoint_sha256": sha(out / "best.pt"), "selected_score": final})
    finally:
        evaluator.close()


def comparison_rows(run):
    run = Path(run)
    arm_a = read(run / "group_A_g120_original_body/evaluation.json")["summary"]
    arm_b = read(run / "group_B_g120_old_c3/score.json")
    arm_c = read(run / "group_C_train/selected_best/score.json")
    rows = []
    for name, value in (("A_G120_original_body", arm_a),
                        ("B_G120_old_C3", arm_b),
                        ("C_G120_new_C3", arm_c)):
        row = {"group": name, **{key: value[key] for key in
            ("task1_r", "machine_r", "body_r", "task2_r", "mean_r")}}
        rows.append(row)
    baseline = rows[0]["mean_r"]
    for row in rows:
        row["delta_mean_r_vs_A"] = row["mean_r"] - baseline
    return rows


def summarize(run, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new comparison output")
    output.mkdir(parents=True)
    rows = comparison_rows(run)
    csv_write(output / "summary.csv", rows)
    winner = max(rows, key=lambda row: row["mean_r"])
    write(output / "summary.json", {"completed": True, "groups": rows,
        "winner": winner["group"], "winner_mean_r": winner["mean_r"],
        "selection": "maximum complete Task1/Task2 mean_r; no seed cherry-picking"})
    lines = ["# B5-1022：G120 与 C3 三组对照", "",
        "|组别|Task1 r|机器 r|体脂秤 r|Task2 r|最终 R|ΔR vs A|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("|{group}|{task1_r:.8f}|{machine_r:.8f}|{body_r:.8f}|{task2_r:.8f}|"
                     "{mean_r:.8f}|{delta_mean_r_vs_A:+.8f}|".format(**row))
    lines += ["", f"按完整最终 R 选出的组别：**{winner['group']}**。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Use a new three-arm run directory")
    out.mkdir(parents=True)
    python = [sys.executable, "-u", "-m"]
    subprocess.run(python + ["baselines.B5_slow120.evaluate", "--run", args.slow_run,
        "--checkpoint", args.slow_checkpoint, "--output", str(out / "group_A_g120_original_body"),
        "--device", args.device, "--keep-predictions"], check=True)
    base = python + ["baselines.B5_final.g120_c3"]
    subprocess.run(base + ["prepare", "--slow-run", args.slow_run,
        "--slow-checkpoint", args.slow_checkpoint,
        "--group-a-evaluation", str(out / "group_A_g120_original_body"),
        "--output", str(out / "cache"), "--device", args.device], check=True)
    subprocess.run(base + ["evaluate", "--cache", str(out / "cache"),
        "--checkpoint", args.old_c3_checkpoint, "--output", str(out / "group_B_g120_old_c3"),
        "--label", "G120+old_C3", "--device", args.device], check=True)
    subprocess.run(base + ["train", "--cache", str(out / "cache"),
        "--output", str(out / "group_C_train"), "--device", args.device,
        "--seed", str(args.seed), "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate)], check=True)
    summarize(out, out / "comparison")
    write(out / "completed.json", {"completed": True, "protocol": PROTOCOL,
        "slow_checkpoint": str(Path(args.slow_checkpoint).resolve()),
        "slow_checkpoint_sha256": sha(args.slow_checkpoint),
        "old_c3_checkpoint": str(Path(args.old_c3_checkpoint).resolve()),
        "old_c3_checkpoint_sha256": sha(args.old_c3_checkpoint),
        "comparison": read(out / "comparison/summary.json")})
    print("Completed: " + str(out / "comparison/summary.md"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--slow-run", required=True)
    prep.add_argument("--slow-checkpoint", required=True)
    prep.add_argument("--group-a-evaluation", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--resume", action="store_true")
    prep.add_argument("--device", default="cuda")
    ev = subs.add_parser("evaluate")
    ev.add_argument("--cache", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--label", default="G120+C3")
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--keep-predictions", action="store_true")
    tr = subs.add_parser("train")
    tr.add_argument("--cache", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--learning-rate", type=float, default=2e-4)
    tr.add_argument("--keep-predictions", action="store_true")
    summary = subs.add_parser("summarize")
    summary.add_argument("--run", required=True)
    summary.add_argument("--output", required=True)
    pipe = subs.add_parser("pipeline")
    pipe.add_argument("--slow-run", required=True)
    pipe.add_argument("--slow-checkpoint", required=True)
    pipe.add_argument("--old-c3-checkpoint", required=True)
    pipe.add_argument("--output", required=True)
    pipe.add_argument("--device", default="cuda")
    pipe.add_argument("--seed", type=int, default=42)
    pipe.add_argument("--epochs", type=int, default=10)
    pipe.add_argument("--batch-size", type=int, default=16)
    pipe.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "train":
        train(args)
    elif args.command == "summarize":
        summarize(args.run, args.output)
    else:
        pipeline(args)


if __name__ == "__main__":
    main()
