"""Server-only preparation: immutable observed I, labels and frozen P per record."""
import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
import torch

from baselines.B5.checkpoint import checkpoint_preprocessor
from baselines.B5.config import load_config, ModelConfig
from baselines.B5.data import HuaweiTrainDataset, demographics_table, preprocessing_config, common_config, demographics_path
from baselines.B5.metadata import read_csv, subject_key
from baselines.B5.runtime import build_validation_datasets, device_from_name, seed_all
from baselines.B5.objective import validate_mainline_loss
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.predict import Predictor
from baselines.B5_best.combine import slow_curve, upsample_native
from baselines.B5_best.slow import predict_native
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_best.validate import verify_reference_inputs
from baselines.B5_best_context.evaluate import load_baseline
from baselines.B5_final.profile import add_asset_arguments, assets_from_args, check_payloads, BASE_SHA, BODY_SHA, SAMPLING
from baselines.B5_final.validate import check_baseline
from . import PROTOCOL, FIELDS
from .common import DEFAULT_CONFIG, load_experiment, code_sha
from .model import visible_windows


def group_train_rows(rows):
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["target_record_id"]].append(i)
    for ids in grouped.values():
        ids.sort(key=lambda i: int(rows[i]["start_sample_500hz"]))
        if ([int(rows[i]["start_sample_500hz"]) for i in ids] != list(range(0, 60000, 5000))
                or len({subject_key(rows[i]["subject_id"]) for i in ids}) != 1):
            raise ValueError("Need 12 unique consecutive windows per train patient record")
    return [grouped[key] for key in sorted(grouped)]


def visible_digest(anchor, metadata):
    digest = hashlib.sha256(np.ascontiguousarray(anchor).tobytes())
    for key in FIELDS:
        digest.update(np.asarray(metadata[key]).tobytes())
    return digest.hexdigest()


@torch.no_grad()
def frozen_components(engine, anchor, meta):
    fields = {k: torch.from_numpy(np.asarray(meta[k])[None].copy()).to(engine.device) for k in FIELDS}
    raw50 = np.asarray(anchor, np.float32).reshape(1, 6000, 10).mean(-1) / engine.scale[0]
    native = predict_native(engine.slow, {"anchor": torch.from_numpy(raw50[None]).to(engine.device), **fields})
    slow = upsample_native(native[0].cpu().numpy().astype(float) * engine.scale[1:, None].astype(float))
    inputs = visible_windows(anchor, meta, engine.scale[0], engine.device)
    condition = engine.fm.encode_conditions(**inputs)
    original_i = (condition.anchor_prediction[:, 0].reshape(1, 60000).cpu().numpy() * engine.scale[0])
    return slow.astype(np.float32), original_i.astype(np.float32)


def prepare(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Use a new shared cache directory")
    experiment = load_experiment(args.experiment_config)
    assets = assets_from_args(args)
    g0 = read(args.g0_report)
    if (g0.get("completed") is not True or not g0.get("reference_check", {}).get("matched")
            or g0.get("base_sha256") != BASE_SHA or g0.get("body_sha256") != BODY_SHA):
        raise ValueError("First reproduce original B5_final.validate --verify-reference")
    baseline = Path(args.baseline_evaluation or assets.baseline_evaluation).resolve()
    if str(baseline) != g0["baseline_evaluation"]:
        raise ValueError("G0 belongs to a different baseline")
    config_path = Path(args.config or assets.validation_config).resolve()
    cfg = load_config(config_path, data_only=True)
    bundle = load_bundle(assets.base)
    if ModelConfig.from_dict(cfg["model"]).fingerprint != bundle["fm"]["architecture_hash"]:
        raise ValueError("Historical data/model configuration differs from selected Final")
    check_payloads(bundle, torch.load(assets.body, map_location="cpu", weights_only=True))
    verify_reference_inputs(bundle, cfg)
    # Portable bundles intentionally omit training settings. The pinned historical
    # data/run config is the default audited objective, not invented loss weights.
    fm_loss = cfg["loss"]
    loss_source = {"kind": "historical_Final_run_config", "path": str(config_path), "sha256": sha(config_path)}
    if args.fm_source_checkpoint:
        from baselines.B5.checkpoint import load_checkpoint
        if sha(args.fm_source_checkpoint) != bundle["provenance"]["fm_sha256"]:
            raise ValueError("Raw FM source must match the bundle's original source digest")
        source = load_checkpoint(args.fm_source_checkpoint)
        fm_loss = source.get("loss_config", source.get("config", {}).get("loss"))
        if fm_loss is None:
            raise ValueError("Raw FM source has no loss configuration")
        loss_source = {"kind": "matching_raw_FM_source", "path": str(Path(args.fm_source_checkpoint).resolve()),
                       "sha256": sha(args.fm_source_checkpoint)}
    validate_mainline_loss(fm_loss)
    preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(cfg))
    train = HuaweiTrainDataset(cfg, preprocessor, demographics_table(cfg))
    validation = build_validation_datasets(cfg, preprocessor)
    loaded = load_baseline(baseline)
    check_baseline(loaded[0])
    if loaded[2] != g0["population"]:
        raise ValueError("Baseline cache changed since G0")
    train_subjects = {subject_key(r["subject_id"]) for r in train.rows}
    val_subjects = {subject_key(r["subject_id"]) for item in loaded[1].values() for r in item["rows"]}
    if train_subjects & val_subjects:
        raise ValueError("Train/validation patients overlap")
    splits = {subject_key(r["subject_id"]): r["split"] for r in read_csv(common_config(cfg).path("subject_split_csv"))}
    if any(splits.get(s) != "validation" for s in val_subjects):
        raise ValueError("Validation patient split changed")
    device = device_from_name(args.device)
    seed_all(42, True)
    engine = Predictor(bundle, device)
    engine.fm.requires_grad_(False).eval()
    out.mkdir(parents=True)
    entries = {"train": [], "validation": []}

    def save(split, raw, mask, anchor, meta, row, task, base=None):
        if raw.shape != (12, 60000) or mask.shape != (12,) or not np.isfinite(raw).all():
            raise ValueError("Need a finite full record with original per-lead QC")
        if split == "train" and (not mask[0] or not mask[1:].any()):
            raise ValueError("Training needs reliable I and valid target leads")
        # Keep ALL original validation rows, including bad-I pairs. The existing
        # scorer applies the original input/target QC; preparation must not alter
        # population fingerprints or silently redefine the scored population.
        slow, original_i = frozen_components(engine, anchor, meta)
        record = {k: np.asarray(meta[k]) for k in FIELDS}
        record.update(anchor_uV=np.asarray(anchor, np.float32), target_uV=raw.astype(np.float32),
                      mask=mask.astype(bool), slow_uV=slow, original_i_uV=original_i,
                      fast_target=(raw[1:].astype(float) - slow_curve(raw[1:])).astype(np.float32))
        if base is not None:
            # Preserve exactly the original cached I; recomputation only checks the visible path.
            if not np.allclose(original_i, base[:1], rtol=2e-5, atol=2e-3):
                raise ValueError("Visible I head differs from original baseline")
            record["original_i_uV"] = base[:1].copy()
            record["base_uV"] = np.asarray(base, np.float32)
        relative = f"{split}/{len(entries[split]):05d}.npz"
        path = out / relative
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **record)
        entries[split].append({"file": relative, "sha256": sha(path), "task": task,
            "target_record_id": row["target_record_id"], "subject_id": row["subject_id"],
            "pair_id": row.get("pair_id", ""), "input_type": row.get("input_type", ""),
            "window_indices": row.get("window_indices", [])})
        print({"cache": split, "records": len(entries[split])}, flush=True)

    try:
        seen = {}
        for task, item in loaded[1].items():
            ds = validation[task]
            live = {(r["pair_id"], int(r["start_sample_500hz"])): i for i, r in enumerate(ds.rows)}
            if len(live) != len(ds.rows) or len(live) != len(item["rows"]):
                raise ValueError("Live validation/cache population mismatch")
            for ids in item["pairs"]:
                row = item["rows"][ids[0]]
                samples = []
                for i in ids:
                    r = item["rows"][i]
                    sample = ds.base[live[(r["pair_id"], int(r["start_sample_500hz"]))]]
                    if (sample.subject_id != r["subject_id"] or sample.target_record_id != r["target_record_id"]
                            or sample.context_source_type != r["input_type"]
                            or not np.array_equal(sample.Y_12lead, item["target"][i])
                            or not np.array_equal(sample.target_quality_mask, item["quality"][i])):
                        raise ValueError("Validation identity, waveform or QC changed")
                    samples.append(sample)
                anchor = np.concatenate([s.anchor_i_ecg for s in samples], -1)
                raw = np.concatenate([item["target"][i] for i in ids], -1)
                masks = np.asarray([item["quality"][i] for i in ids])
                if not np.all(masks == masks[0]):
                    raise ValueError("Record QC must be constant across windows")
                meta = ds.demographics.get(row["subject_id"])
                digest = visible_digest(anchor, meta)
                if seen.setdefault(row["target_record_id"], digest) != digest:
                    raise ValueError("Repeated target has different observed I or metadata")
                save("validation", raw, masks[0], anchor, meta, {**row, "window_indices": ids}, task,
                     np.concatenate([item["prediction"][i] for i in ids], -1))
        for ids in group_train_rows(train.rows):
            samples = [train.base[i] for i in ids]
            masks = np.asarray([s.target_quality_mask for s in samples])
            if not np.all(masks == masks[0]):
                raise ValueError("Train record QC must be stable")
            raw = np.concatenate([s.Y_12lead for s in samples], -1)
            row = train.rows[ids[0]]
            save("train", raw, masks[0], raw[:1].copy(), train.demographics.get(row["subject_id"]), row, "train")
        if not entries["train"] or not entries["validation"]:
            raise ValueError("No complete train/validation records")
        study = {"protocol": PROTOCOL, "completed": True, "code_sha256": code_sha(),
            "base_checkpoint": str(assets.base), "base_sha256": BASE_SHA,
            "body_checkpoint": str(assets.body), "body_sha256": BODY_SHA,
            "config": str(config_path), "config_sha256": sha(config_path),
            "baseline_evaluation": str(baseline), "population": loaded[2], "sampling": SAMPLING,
            "experiment": experiment, "model_config": bundle["fm"]["model_config"],
            "scale_uV": engine.scale.tolist(), "fm_loss": fm_loss, "fm_loss_source": loss_source,
            "initial_weights_sha256": state_sha(bundle["fm"]["ema"]),
            "train": entries["train"], "validation": entries["validation"],
            "reference": g0["summary"][-1], "g0_summary": g0["summary"],
            "g0_report_sha256": sha(args.g0_report), "train_subjects": len(train_subjects),
            "validation_subjects": len(val_subjects), "data_sources": {
                "demographics": sha(demographics_path(cfg)), "split": sha(common_config(cfg).path("subject_split_csv")),
                "strict_index": sha(common_config(cfg).repository_root / "metadata/d12_strict_pretrain_index.csv")}}
        write(out / "study.json", study)
        write(out / "preflight.json", {"ready": True, "training_started": False,
            "train_records": len(entries["train"]), "validation_pairs": len(entries["validation"]),
            "original_weights": "read_only", "train_FM_ODE_calls": 0,
            "initial_weights_sha256": study["initial_weights_sha256"]})
    finally:
        for ds in validation.values():
            ds.base.close()
        for array in train.base.targets.values():
            array._mmap.close()
        for item in loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_asset_arguments(parser)
    parser.add_argument("--config", help="Original Huawei data configuration")
    parser.add_argument("--experiment-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--baseline-evaluation")
    parser.add_argument("--fm-source-checkpoint", help="Optional original raw FM checkpoint, verified against bundle provenance")
    parser.add_argument("--g0-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    prepare(args)


if __name__ == "__main__":
    main()
