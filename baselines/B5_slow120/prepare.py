"""Build an immutable full-record cache. FM/P frozen, Huawei train patients only."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from baselines.B5.checkpoint import checkpoint_preprocessor
from baselines.B5.config import load_config
from baselines.B5.data import HuaweiTrainDataset, demographics_table, preprocessing_config, common_config, demographics_path
from baselines.B5.metadata import read_csv, subject_key
from baselines.B5.runtime import build_validation_datasets, device_from_name, seed_all
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_best.combine import slow_curve
from baselines.B5_best.predict import Predictor
from baselines.B5_best.slow import predict_native
from baselines.B5_best.validate import verify_reference_inputs
from baselines.B5_best_context.evaluate import load_baseline
from baselines.B5_final.profile import add_asset_arguments, assets_from_args, check_payloads, BASE_SHA, BODY_SHA, SAMPLING
from baselines.B5_final.validate import check_baseline
from . import PROTOCOL, FIELDS
from .common import code_sha
from .model import visible_inputs, SlowCorrection, inputs


def group_train_rows(rows):
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["target_record_id"]].append(i)
    for indices in grouped.values():
        indices.sort(key=lambda i: int(rows[i]["start_sample_500hz"]))
        if [int(rows[i]["start_sample_500hz"]) for i in indices] != list(range(0, 60000, 5000)):
            raise ValueError("Train needs twelve unique real consecutive windows per physical record")
        if len({subject_key(rows[i]["subject_id"]) for i in indices}) != 1:
            raise ValueError("Train record mixes patients")
    return [grouped[key] for key in sorted(grouped)]


@torch.no_grad()
def native_prediction(engine, anchor, metadata):
    fields = {k: torch.from_numpy(np.asarray(metadata[k])[None].copy()).to(engine.device) for k in FIELDS}
    anchor50 = np.asarray(anchor, np.float32).reshape(1, 6000, 10).mean(-1) / engine.scale[0]
    return predict_native(engine.slow, {"anchor": torch.from_numpy(anchor50[None]).to(engine.device),
                                        **fields})[0].cpu().numpy()


def prepare(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Use a new shared cache directory; interrupted caches are not reused")
    assets = assets_from_args(args)
    g0 = read(args.g0_report)
    if (not g0.get("completed") or not g0.get("reference_check", {}).get("matched")
            or g0.get("base_sha256") != BASE_SHA or g0.get("body_sha256") != BODY_SHA):
        raise ValueError("First reproduce B5-final G0 using B5_final.validate --verify-reference")
    baseline_dir = Path(args.baseline_evaluation or assets.baseline_evaluation).resolve()
    if str(baseline_dir) != g0["baseline_evaluation"]:
        raise ValueError("G0 report belongs to another baseline cache")
    config_path = Path(args.config or assets.validation_config).resolve()
    # Final's report names the profile-default config even when --config is
    # overridden. The launcher passes the same explicit config to both phases;
    # verify the bound data below rather than rejecting a relocated config.
    cfg = load_config(config_path, data_only=True)
    bundle = load_bundle(assets.base)
    check_payloads(bundle, torch.load(assets.body, map_location="cpu", weights_only=True))
    verify_reference_inputs(bundle, cfg)
    preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(cfg))
    train = HuaweiTrainDataset(cfg, preprocessor, demographics_table(cfg))
    validation = build_validation_datasets(cfg, preprocessor)
    loaded = load_baseline(baseline_dir)
    check_baseline(loaded[0])
    if loaded[2] != g0["population"]:
        raise ValueError("Baseline population/predictions changed since G0 reproduction")
    train_subjects = {subject_key(r["subject_id"]) for r in train.rows}
    val_subjects = {subject_key(r["subject_id"]) for item in loaded[1].values() for r in item["rows"]}
    if train_subjects & val_subjects:
        raise ValueError("Train/validation patient leakage")
    splits = {subject_key(r["subject_id"]): r["split"] for r in read_csv(common_config(cfg).path("subject_split_csv"))}
    if any(splits.get(s) != "validation" for s in val_subjects):
        raise ValueError("Validation population includes a non-validation patient")
    device = device_from_name(args.device)
    seed_all(42, True)
    engine = Predictor(bundle, device)
    out.mkdir(parents=True)
    entries = {"train": [], "validation": []}
    scale = engine.scale

    def save(split, raw, mask, anchor, metadata, base, row, task):
        if raw.shape != (12, 60000) or mask.shape != (12,) or not np.isfinite(raw).all():
            raise ValueError("Need full finite raw-uV targets and per-record QC")
        native = native_prediction(engine, anchor, metadata)
        target = slow_curve(raw[1:]).reshape(11, 6000, 10).mean(-1) / scale[1:, None]
        record = {**visible_inputs(anchor, scale[0], metadata), "base_slow": native.astype(np.float32),
                  "slow_target": target.astype(np.float32), "base_uV": np.asarray(base, np.float32),
                  "target_uV": np.asarray(raw, np.float32), "mask": np.asarray(mask, bool),
                  "anchor_uV": np.asarray(anchor, np.float32)}
        relative = f"{split}/{len(entries[split]):05d}.npz"
        path = out / relative
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **record)
        entries[split].append({"file": relative, "sha256": sha(path), "task": task,
            "target_record_id": row["target_record_id"], "subject_id": row["subject_id"],
            "pair_id": row.get("pair_id", ""), "input_type": row.get("input_type", ""),
            "window_indices": row.get("window_indices", [])})
        print({"cache": split, "records": len(entries[split]), "record": row["target_record_id"]}, flush=True)

    try:
        # Validation anchors are explicitly read from live observed-input fields,
        # never extracted from hidden Y. Repeated targets must have equal inputs.
        seen = {}
        for task, item in loaded[1].items():
            ds = validation[task]
            live = {(r["pair_id"], int(r["start_sample_500hz"])): i for i, r in enumerate(ds.rows)}
            if len(live) != len(ds.rows) or len(live) != len(item["rows"]):
                raise ValueError("Live/cache validation size mismatch")
            for ids in item["pairs"]:
                row = item["rows"][ids[0]]
                samples = []
                for i in ids:
                    r = item["rows"][i]
                    j = live[(r["pair_id"], int(r["start_sample_500hz"]))]
                    sample = ds.base[j]
                    if (sample.subject_id != r["subject_id"] or sample.target_record_id != r["target_record_id"]
                            or sample.context_source_type != r["input_type"]
                            or not np.array_equal(sample.Y_12lead, item["target"][i])
                            or not np.array_equal(sample.target_quality_mask, item["quality"][i])):
                        raise ValueError("Live validation data/identity/QC differs from pinned cache")
                    samples.append(sample)
                anchor = np.concatenate([s.anchor_i_ecg for s in samples], -1)
                raw = np.concatenate([item["target"][i] for i in ids], -1)
                masks = np.asarray([item["quality"][i] for i in ids])
                if not np.all(masks == masks[0]):
                    raise ValueError("Record QC must be stable across windows")
                meta = ds.demographics.get(row["subject_id"])
                digest = sha_visible(anchor, meta)
                if seen.setdefault(row["target_record_id"], digest) != digest:
                    raise ValueError("Repeated target has different visible inputs")
                base = np.concatenate([item["prediction"][i] for i in ids], -1)
                save("validation", raw, masks[0], anchor, meta, base, {**row, "window_indices": ids}, task)
        for ids in group_train_rows(train.rows):
            samples = [train.base[i] for i in ids]
            raw = np.concatenate([s.Y_12lead for s in samples], -1)
            mask = np.asarray([s.target_quality_mask for s in samples])
            if not np.all(mask == mask[0]) or not mask[0, 0] or not mask[0, 1:].any():
                raise ValueError("Train requires reliable I and at least one target, stable record QC")
            row = train.rows[ids[0]]
            meta = train.demographics.get(row["subject_id"])
            anchor = raw[:1].copy()  # train within-d12 synchronous observed I
            base = engine.predict_encoded(anchor, meta, row["target_record_id"], args.fm_batch_size)
            save("train", raw, mask[0], anchor, meta, base, row, "train")
        torch.manual_seed(args.training_seed)
        g50 = SlowCorrection("G50").eval()
        torch.manual_seed(args.training_seed)
        g120 = SlowCorrection("G120").eval()
        if state_sha(g50.state_dict()) != state_sha(g120.state_dict()):
            raise AssertionError("Matched arms must have identical initial weights")
        with np.load(out / entries["train"][0]["file"]) as data:
            batch = {k: torch.from_numpy(np.asarray(v)[None].copy()) for k, v in data.items()}
        with torch.no_grad():
            for model in (g50, g120):
                if torch.count_nonzero(model(**inputs(batch))).item() != 0:
                    raise AssertionError("Correction must be exactly zero at initialization")
        study = {"protocol": PROTOCOL, "completed": True, "score_protocol": SCORE_PROTOCOL,
            "base_checkpoint": str(assets.base), "base_sha256": BASE_SHA,
            "body_checkpoint": str(assets.body), "body_sha256": BODY_SHA,
            "config": str(config_path), "config_sha256": sha(config_path), "baseline_evaluation": str(baseline_dir),
            "population": loaded[2], "sampling": SAMPLING, "slow_beta": .75, "scale_uV": scale.tolist(),
            "model": {"channels": 32, "layers": 2, "heads": 4},
            "training": {"epochs": args.epochs, "seed": args.training_seed, "batch_records": args.batch_records,
                "learning_rate": args.learning_rate, "weight_decay": .01, "gradient_clip": 1.,
                "validate_every": args.validate_every}, "code_sha256": code_sha(),
            "initial_weights_sha256": state_sha(g50.state_dict()), "reference": g0["summary"][-1],
            "g0_report": str(Path(args.g0_report).resolve()), "g0_report_sha256": sha(args.g0_report),
            "train": entries["train"], "validation": entries["validation"],
            "validation_rows": {t: i["rows"] for t, i in loaded[1].items()},
            "train_subjects": len(train_subjects), "validation_subjects": len(val_subjects),
            "data_sources": {"config": sha(config_path), "demographics": sha(demographics_path(cfg)),
                "split": sha(common_config(cfg).path("subject_split_csv")),
                "strict_index": sha(common_config(cfg).repository_root / "metadata/d12_strict_pretrain_index.csv")},
            "FM_optimizer_updates": 0, "P_optimizer_updates": 0,
            "loss": {"slow_huber": 1., "slow_pcc": .1, "F_residual": 2.5890171538829843,
                "F_width_native": 51, "combo_pcc": .1, "extra_multiscale": 0.},
            "note": "Training loss uses fixed plain base; selection uses unchanged Final d6/body after slow correction."}
        write(out / "study.json", study)
        write(out / "preflight.json", {"ready": True, "training_started": False, "train_records": len(entries["train"]),
            "validation_pairs": len(entries["validation"]), "zero_init_max_abs_difference": 0.,
            "same_initial_weights": True, "parameters_per_arm": sum(p.numel() for p in g50.parameters()),
            "sampling": SAMPLING, "reference": study["reference"]})
    finally:
        for ds in validation.values():
            ds.base.close()
        for array in train.base.targets.values():
            array._mmap.close()
        for item in loaded[1].values():
            for k in ("prediction", "target", "quality"):
                item[k]._mmap.close()


def sha_visible(anchor, metadata):
    import hashlib
    digest = hashlib.sha256(np.ascontiguousarray(anchor).tobytes())
    for k in FIELDS:
        digest.update(np.asarray(metadata[k]).tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_asset_arguments(p)
    p.add_argument("--config")
    p.add_argument("--baseline-evaluation")
    p.add_argument("--g0-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fm-batch-size", type=int, default=8)
    p.add_argument("--training-seed", type=int, default=43)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--validate-every", type=int, default=5)
    p.add_argument("--batch-records", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    args = p.parse_args()
    if (min(args.epochs, args.validate_every, args.batch_records, args.fm_batch_size) < 1
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not 0 <= args.training_seed < 2**32):
        p.error("Need positive budgets/LR and uint32 training seed")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    prepare(args)


if __name__ == "__main__":
    main()
