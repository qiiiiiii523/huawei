"""Two FM x three slow-CNN crossover, without changing frozen studies."""
import argparse
from contextlib import ExitStack
import hashlib
from pathlib import Path

import numpy as np
import torch

from baselines.B5.checkpoint import checkpoint_preprocessor, model_from_checkpoint
from baselines.B5.data import preprocessing_config
from baselines.B5.runtime import build_validation_datasets, device_from_name, seed_all
from baselines.B5_best import SAMPLING
from baselines.B5_best.common import read, write, csv_write, sha
from baselines.B5_best.combine import combine, upsample_native
from baselines.B5_best.diagnose import csv_read, align, target_file
from baselines.B5_best.scoring import diagnostic_summary, scores_from_summaries, write_score_report
from baselines.B5_best.validate import pair_indices
from baselines.B5_slow_pcc import CACHE_PROTOCOL
from baselines.B5_slow_pcc.common import load_study, load_checkpoint, slow_model, validation_key
from baselines.B5_slow_pcc.core import generate, model_inputs, device_batch
from baselines.B5_slow_pcc.cache import generate_fm
from ecg12gen.evaluate import evaluate_record_predictions, write_report
from . import PROTOCOL
from .components import control_payload, candidate, save_candidate

SLOW_NAMES = ("original", "control20", "pcc20")
FM_NAMES = ("original", "control100")


def validation_manifest(cache, study_path):
    """Only validation is read; the 156 training archives are not needed."""
    manifest = read(cache / "cache.json")
    if (manifest.get("protocol") != CACHE_PROTOCOL or not manifest.get("completed")
            or manifest.get("study_sha256") != sha(study_path) or manifest.get("sampling") != SAMPLING):
        raise ValueError("Need the completed repaired SlowPCC cache from this study")
    entries = manifest["validation"]
    keys = [e["cache_key"] for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate validation pair cache key")
    lookup = {e["cache_key"]: e for e in entries}
    for entry in entries:
        path = (cache / entry["file"]).resolve()
        if not path.is_relative_to(cache.resolve()) or sha(path) != entry["sha256"]:
            raise ValueError("Validation cache changed: " + entry["file"])
    for task in ("task1", "task2"):
        rows = manifest["validation_rows"][task]
        for ids in pair_indices(rows):
            row = rows[ids[0]]
            entry = lookup[validation_key(task, row["pair_id"])]
            if (entry["target_record_id"] != row["target_record_id"]
                    or entry["subject_id"] != row["subject_id"]):
                raise ValueError("Validation cache identity changed")
    return manifest


class SavedControl:
    def __init__(self, root, rows_by_task):
        self.stack = ExitStack()
        self.arrays, self.orders, self.audit = {}, {}, {}
        try:
            for task, rows in rows_by_task.items():
                folder = root / task
                paths = {"prediction": folder / "prediction_uV.npy", "target": target_file(folder),
                         "mask": folder / "target_quality_mask.npy", "metadata": folder / "window_metadata.csv"}
                other = csv_read(paths["metadata"])
                self.orders[task] = align(rows, other)
                arrays = {}
                for key in ("prediction", "target", "mask"):
                    value = np.load(paths[key], mmap_mode="r", allow_pickle=False)
                    self.stack.callback(value._mmap.close)
                    expected = (len(other), 12) if key == "mask" else (len(other), 12, 5000)
                    if value.shape != expected or (key == "mask" and value.dtype != bool):
                        raise ValueError("Wrong saved control array shape/type: " + str(paths[key]))
                    arrays[key] = value
                self.arrays[task] = arrays
                self.audit[task] = {k: {"path": str(p.resolve()), "sha256": sha(p)} for k, p in paths.items()}
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stack.close()

    def record(self, task, indices, target, mask):
        ids = self.orders[task][indices]
        arrays = self.arrays[task]
        if (not np.array_equal(np.concatenate(arrays["target"][ids], axis=-1), target)
                or not np.array_equal(arrays["mask"][ids], np.broadcast_to(mask, (len(ids), 12)))):
            raise ValueError("Control100 targets/QC differ from the frozen SlowPCC validation")
        result = np.concatenate(arrays["prediction"][ids], axis=-1)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite saved control prediction")
        return result


@torch.no_grad()
def predict_slow(cache, entries, model, scales, device):
    """Once per physical target; labels and cached FM cannot enter the CNN."""
    learned, digests = {}, {}
    for entry in entries:
        key = entry["target_record_id"]
        with np.load(cache / entry["file"], allow_pickle=False) as archive:
            inputs = {k: torch.from_numpy(np.asarray(archive[k])[None].copy())
                      for k in model_inputs({k: None for k in archive.files})}
            digest = hashlib.sha256()
            for field, value in sorted(inputs.items()):
                digest.update(field.encode()); digest.update(value.numpy().tobytes())
            if key in learned:
                if digest.hexdigest() != digests[key]:
                    raise ValueError("Repeated target has different observed I/demographics")
                continue
            native = generate(model, device_batch(inputs, device))[0].cpu().numpy()
            learned[key] = upsample_native(native.astype(np.float64) * scales[1:, None].astype(np.float64))
            digests[key] = digest.hexdigest()
    return learned


def noise_inputs(study, bundle, rows_by_task):
    """Seed43 generation reads observed I from the validation adapter, not Y."""
    pre = checkpoint_preprocessor(bundle["fm"], preprocessing_config(study["data_config"]))
    datasets = build_validation_datasets(study["data_config"], pre)
    observed, references = {}, {}
    for task, rows in rows_by_task.items():
        ds = datasets[task]
        if ds.manifest_digest != study["manifests"][task]:
            raise ValueError("Validation population changed")
        order = align(rows, ds.rows)
        for ids in pair_indices(rows):
            row = rows[ids[0]]; key = row["target_record_id"]
            anchor = np.concatenate([ds.base[i].anchor_i_ecg for i in order[ids]], axis=-1).astype(np.float32)
            meta = ds.demographics.get(row["subject_id"])
            digest = hashlib.sha256(anchor.tobytes())
            for field, value in sorted(meta.items()):
                digest.update(field.encode()); digest.update(np.asarray(value).tobytes())
            if key in observed and references[key] != digest.hexdigest():
                raise ValueError("Repeated target has different test-visible conditions")
            observed[key] = (anchor, meta)
            references[key] = digest.hexdigest()
    return observed


def score_combo(fm_name, slow_name, learned, cache, manifest, saved, generated, output, sampling):
    output.mkdir(parents=True)
    lookup = {entry["cache_key"]: entry for entry in manifest["validation"]}
    summaries, audit = {}, {}
    for task, rows in manifest["validation_rows"].items():
        folder = output / task; folder.mkdir()
        with ExitStack() as stack:
            arrays = {}
            for name, dtype, shape in (("prediction_uV", np.float32, (len(rows), 12, 5000)),
                                      ("target_uV", np.float32, (len(rows), 12, 5000)),
                                      ("target_quality_mask", bool, (len(rows), 12))):
                value = np.lib.format.open_memmap(folder / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
                stack.callback(value._mmap.close)
                arrays[name] = value
            for ids in pair_indices(rows):
                first = rows[ids[0]]; key = first["target_record_id"]
                entry = lookup[validation_key(task, first["pair_id"])]
                with np.load(cache / entry["file"], allow_pickle=False) as record:
                    target, mask = record["target_uV"], record["mask"]
                    if generated is not None:
                        fm = generated[key]
                    elif fm_name == "original":
                        fm = record["fm_uV"]  # Preserve each historical task/pair ensemble.
                    else:
                        fm = saved.record(task, ids, target, mask)
                    pred = combine(fm, learned[key], .75)
                    for pos, i in enumerate(ids):
                        window = slice(pos * 5000, (pos + 1) * 5000)
                        arrays["prediction_uV"][i] = pred[:, window]
                        arrays["target_uV"][i] = target[:, window]
                        arrays["target_quality_mask"][i] = mask
            for value in arrays.values():
                value.flush()
            summary, details = evaluate_record_predictions(arrays["prediction_uV"], arrays["target_uV"], task,
                                      rows, target_quality_mask=arrays["target_quality_mask"])
            summary = diagnostic_summary(summary)
            summary.update(beta=.75, sampling=sampling, target_used_at_inference=False,
                           model="B5-component-crossover", fm_component=fm_name, slow_component=slow_name)
            write_report(folder, summary, details, title="Component crossover: full120s target-lead QC r")
            csv_write(folder / "window_metadata.csv", rows)
            summaries[task] = summary
        audit[task] = {name: sha(folder / name) for name in
                       ("window_metadata.csv", "target_uV.npy", "target_quality_mask.npy")}
    score = scores_from_summaries(summaries)
    write_score_report(output, score)
    write(output / "evaluation.json", {"completed": True, "protocol": PROTOCOL, "score": score,
          "sampling": sampling, "beta": .75, "population": audit,
          "training_started": False, "target_used_at_inference": False})
    return score


def check_reference(score, expected, label, tolerance=2e-5):
    for key in ("task1_r_missing11", "task2_r_missing11"):
        if abs(score[key] - expected[key]) > tolerance:
            raise ValueError(f"Reference {label} did not reproduce {key}: {score[key]} vs {expected[key]}; keep reports")


def run(args):
    output, root = Path(args.output).resolve(), Path(args.slow_run).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output; keep tee log BESIDE it")
    study_path, cache = root / "shared/study.json", root / "cache"
    print("Preflight: checking frozen study, validation cache and control100 identities (CPU/disk)", flush=True)
    study, original = load_study(study_path)
    if study["beta"] != .75:
        raise ValueError("This crossover fixes beta=0.75")
    manifest = validation_manifest(cache, study_path)
    fm_path, fm_evaluation = Path(args.control_checkpoint).resolve(), Path(args.control_evaluation).resolve()
    fm100, _ = control_payload(fm_path, fm_evaluation, original)
    for source in (root, fm_evaluation, fm_path.parent):
        if output == source or source.is_relative_to(output):
            raise ValueError("Output must not contain a source experiment")
    output.mkdir(parents=True, exist_ok=True)
    device = device_from_name(args.device)
    seed_all(args.seed, True)
    sampling = dict(SAMPLING, seed=args.seed)
    provenance = {"study_sha256": sha(study_path), "cache_sha256": sha(cache / "cache.json"),
                  "control100_checkpoint": str(fm_path), "control100_checkpoint_sha256": sha(fm_path),
                  "control100_evaluation": str(fm_evaluation),
                  "control100_evaluation_sha256": sha(fm_evaluation / "evaluation.json")}
    slow_states = {"original": original["slow"]["model"]}
    expected = {"original": original["reference_qc_r"]}
    for name, group in (("control20", "control"), ("pcc20", "pcc")):
        cp_path = root / group / "best.pt"
        cp, _, _, model = load_checkpoint(cp_path, study_path, torch.device("cpu"))
        if cp["group"] != group or cp["cache_sha256"] != sha(cache / "cache.json"):
            raise ValueError("Updated slow checkpoint/group/cache mismatch")
        slow_states[name] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        expected[name] = cp["validation_score"]
        provenance[name + "_checkpoint"] = str(cp_path)
        provenance[name + "_sha256"] = sha(cp_path)
        provenance[name + "_epoch"] = cp["epoch"]
        del model
    status = {"protocol": PROTOCOL, "completed": False, "training_started": False, "optimizer_updates": 0,
              "sampling": sampling, "beta": .75, "FM_inference_required": args.seed != 42,
              "provenance": provenance, "frozen_study_modified": False}
    write(output / "run.json", status)
    rows, winner, best = [], None, -float("inf")
    scales = np.asarray(original["fm"]["scales"]["d12"], np.float32)
    observed = noise_inputs(study, original, manifest["validation_rows"]) if args.seed != 42 else None
    if observed is not None:
        for entry in manifest["validation"]:
            anchor, meta = observed[entry["target_record_id"]]
            with np.load(cache / entry["file"], allow_pickle=False) as record:
                if not np.array_equal(anchor.reshape(1, 6000, 10).mean(-1) / scales[0], record["anchor"]):
                    raise ValueError("Seed43 observed I differs from frozen seed42 conditions")
                if any(not np.array_equal(record[k], v) for k, v in meta.items()):
                    raise ValueError("Seed43 demographics differ from frozen seed42 conditions")
    fm_payloads = {"original": original["fm"], "control100": fm100}
    learned_by_name = {}
    for slow_name in SLOW_NAMES:
        slow_bundle = candidate(original, original["fm"], slow_states[slow_name], "original", slow_name, provenance)
        model = slow_model(slow_bundle, device, trainable=False).eval()
        learned_by_name[slow_name] = predict_slow(cache, manifest["validation"], model, scales, device)
        del model
        print(f"Slow predictions ready: {slow_name}; reused for both FM components", flush=True)
    with ExitStack() as stack:
        saved = SavedControl(fm_evaluation, manifest["validation_rows"])
        stack.callback(saved.close)
        write(output / "source_arrays.json", saved.audit)
        lookup = {e["cache_key"]: e for e in manifest["validation"]}
        for task, metadata in manifest["validation_rows"].items():
            for ids in pair_indices(metadata):
                entry = lookup[validation_key(task, metadata[ids[0]]["pair_id"])]
                with np.load(cache / entry["file"], allow_pickle=False) as record:
                    saved.record(task, ids, record["target_uV"], record["mask"])
        for fm_name in FM_NAMES:
            generated = None
            if observed is not None:
                generated = {}
                fm_model = model_from_checkpoint(fm_payloads[fm_name], device).requires_grad_(False).eval()
                for n, (key, (anchor, meta)) in enumerate(observed.items(), 1):
                    generated[key] = generate_fm(fm_model, anchor, meta, key, scales, sampling, device)
                    print(f"seed{args.seed} {fm_name} FM {n}/{len(observed)}: {key}", flush=True)
                del fm_model
            for slow_name in SLOW_NAMES:
                label = fm_name + "__" + slow_name
                bundle = candidate(original, fm_payloads[fm_name], slow_states[slow_name], fm_name, slow_name, provenance)
                print(f"Scoring {label}: beta=.75, seed{args.seed}", flush=True)
                score = score_combo(fm_name, slow_name, learned_by_name[slow_name], cache, manifest,
                                    saved, generated, output / label, sampling)
                if args.seed == 42 and fm_name == "original":
                    check_reference(score, expected[slow_name], label)
                if args.seed == 42 and fm_name == "control100" and slow_name == "original":
                    check_reference(score, {"task1_r_missing11": .697867, "task2_r_missing11": .693206}, label)
                save_candidate(output / (label + ".pt"), bundle, score, args.seed)
                rows.append({"fm": fm_name, "slow": slow_name, "seed": args.seed, "beta": .75, **score})
                csv_write(output / "summary.csv", rows)
                print({"combination": label, **score}, flush=True)
                if score["main_correlation"] > best:
                    winner, best = label, score["main_correlation"]
    write(output / "result.json", {**status, "completed": True, "scores": rows,
          "best_combination": winner, "best_mean_r": best,
          "portable_bundle": str(output / (winner + ".pt")),
          "seed43_bundle_deployment_seed_is_42": args.seed != 42})
    write(output / "run.json", {**status, "completed": True})
    table = ["# B5组件交叉组合", "", "固定beta=0.75；平均r=(r1+r2)/2；RMSE不选模。", "",
             "|FM|慢CNN|seed|Task1 r|Task2 r|平均r|", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"|{row['fm']}|{row['slow']}|{args.seed}|{row['task1_r_missing11']:.6f}|"
                     f"{row['task2_r_missing11']:.6f}|{row['main_correlation']:.6f}|")
    table += ["", f"本次验证最高组合：{winner}，R={best:.8f}。",
              "这是同一验证集的候选筛选，不是独立测试成绩；不把两个单项最大值合成虚构结果。",
              "seed43仅用于噪声稳定性复核，导出的部署包仍保持seed42。"]
    (output / "report.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"Completed: {output / 'summary.csv'}; best={winner}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path("/home/qht/huawei")
    parser.add_argument("--slow-run", default=str(repo / "results/B5-best-exp-SlowPCC/combo_pcc_v1"))
    parser.add_argument("--control-checkpoint", default=str(repo / "results/B5-scratch/early_public_v1/control/checkpoints/last.pt"))
    parser.add_argument("--control-evaluation", default=str(repo / "results/B5-scratch/early_public_v1/control/evaluations/last"))
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
