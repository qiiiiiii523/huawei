"""Build immutable, calibrated public/Huawei record caches and freeze slow partners."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from baselines.B5.checkpoint import load_checkpoint
from baselines.B5.config import load_config, ModelConfig
from baselines.B5.data import common_config, demographics_table, demographics_path
from baselines.B5.runtime import device_from_name
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import read, write, sha, csv_write
from baselines.B5_best.slow import load_slow, predict_native
from baselines.B5_best_context.adapters import history_from_sample
from baselines.B5_slow120.common import load_study as source_study, load_correction
from baselines.B5_slow120.model import visible_inputs, inputs
from baselines.B5_slow120.evaluate import array_batch
from ecg12gen.dataset import JointAnchorDataset
from . import PROTOCOL
from .audit import audit, make_pairs
from .ptb_qc import calibrated_record
from .common import code_sha
from .data import FIELDS
from .sampling import SAMPLING


def prepare(a):
    out = Path(a.output).resolve()
    if out.exists(): raise FileExistsError("Use new shared output")
    src = Path(a.slow_run).resolve(); s = source_study(src)
    device = device_from_name(a.device)
    g, gc = load_correction(a.slow_checkpoint, s, src / "study.json", device)
    if gc["group"] != "G120": raise ValueError("Need actual G120 checkpoint")
    report = read(Path(a.stable_evaluation) / "evaluation.json")
    if (not report.get("completed") or report.get("summary", {}).get("group") != "G120"
            or report["summary"]["epoch"] != gc["epoch"] or report.get("population") != s["population"]):
        raise ValueError("Stable evaluation must be this G120/study's complete original-route evaluation")
    bundle = load_bundle(s["base_checkpoint"])
    cp = load_checkpoint(a.public_checkpoint)
    if cp.get("stage") != "public": raise ValueError("Use a genuine public-pretrained B5 checkpoint")
    if cp["scales"] != bundle["fm"]["scales"] or cp["model_config"] != bundle["fm"]["model_config"]:
        raise ValueError("Public FM must share the existing Final architecture and frozen scales")
    cfg = load_config(s["config"], data_only=True)
    if sha(s["config"]) != s["config_sha256"]:
        raise ValueError("Original G120 data configuration changed")
    if sha(demographics_path(cfg)) != s["data_sources"]["demographics"]:
        raise ValueError("Demographics differ from the fixed G120 experiment")
    meta = demographics_table(cfg)
    scale = np.asarray(cp["scales"]["d12"], np.float32)
    p = load_slow(bundle["slow"], device, bundle["provenance"]["slow_variant"])
    g.eval()
    public_root = Path(a.public_root).resolve()
    if out.is_relative_to(public_root): raise ValueError("Output cannot be inside public source")
    out.mkdir(parents=True)
    candidate = audit(public_root, out / "public_audit")
    records = {k: [] for k in ("public_train", "public_validation", "huawei_train", "huawei_validation")}
    assets = {str(Path(path).resolve()): sha(path) for path in
        (a.public_checkpoint, a.slow_checkpoint, src / "study.json", s["base_checkpoint"], s["config"],
         Path(a.stable_evaluation) / "evaluation.json")}

    def save(split, row, values):
        path = out / split / f"{len(records[split]):05d}.npz"; path.parent.mkdir(exist_ok=True)
        np.savez(path, **values)
        records[split].append(dict(row, file=path.relative_to(out).as_posix(), sha256=sha(path)))
        print({"cache": split, "records": len(records[split]), "target": row["target_record_id"]}, flush=True)

    # Rebuild one-history-per-target AFTER technical QC, keeping patient split unchanged.
    eligible, audit_rows = [], []
    public_meta = {}
    qc_dir = out / "public_qc"; qc_dir.mkdir()
    for row in candidate["records"]:
        raw, mask, reasons, demographics, _ = calibrated_record(public_root / row["record_id"])
        good = bool(mask[0] and mask[1:].any())
        audit_rows.append(dict(record=row["record_id"], usable=good, mask="".join(map(lambda v: str(int(v)), mask)),
                               reasons="|".join(reasons)))
        if not good: continue
        qfile = qc_dir / (row["record_id"].replace('/', '_') + '.npz')
        np.savez(qfile, target_uV=raw, mask=mask, **demographics)
        public_meta[row["record_id"]] = qfile
        eligible.append(row)
    csv_write(out / "public_QC.csv", audit_rows)
    pairs = make_pairs(eligible, candidate["patient_split"])
    for split in ("train", "validation"):
        subset = [r for r in pairs if r["split"] == split]
        patients = len({r["subject_id"] for r in subset})
        if patients < (a.min_public_train if split == "train" else a.min_public_validation):
            raise ValueError("Too few post-QC cross-record public patients; inspect public_QC.csv")
        for row in subset:
            with np.load(public_meta[row["target_record_id"]]) as t, np.load(public_meta[row["history_record_id"]]) as h:
                # 120s cap is real data only. One random target window per record exposure.
                n = min(60000, t["target_uV"].shape[-1] // 5000 * 5000)
                hist = h["target_uV"][:6, :60000].copy()
                hm = np.broadcast_to(h["mask"][:6, None], hist.shape).copy()
                hist[~hm] = 0
                save("public_" + split, row, dict(target_uV=t["target_uV"][:, :n], anchor_uV=t["target_uV"][:1, :n],
                    mask=t["mask"], history_uV=hist, history_mask=hm, **{k: t[k] for k in FIELDS}))
    assets.update({str(public_root / name): digest for name, digest in candidate["sources_sha256"].items()})

    subjects = {}
    expected_validation = {e["pair_id"]: e for e in s["validation"]
                           if e["task"] == "task2" and e["input_type"] == "ecg_machine_d6"}
    actual_validation = set()
    for split in ("train", "validation"):
        ds = JointAnchorDataset(common_config(cfg), "task2", split)
        try:
            rows = [ds._rows[i] for i in ds._indices]
            grouped = defaultdict(list)
            for i, r in enumerate(rows):
                if r["input_type"] == "ecg_machine_d6": grouped[r["pair_id"]].append(i)
            # De-duplicate target exposure in training, never remove validation pairs.
            seen = set(); subjects[split] = set()
            for pair in sorted(grouped):
                ids = sorted(grouped[pair], key=lambda i: int(rows[i]["start_sample_500hz"]))
                if [int(rows[i]["start_sample_500hz"]) for i in ids] != list(range(0, 60000, 5000)):
                    raise ValueError("Need all twelve consecutive target windows")
                row = rows[ids[0]]
                if split == "train" and row["target_record_id"] in seen: continue
                seen.add(row["target_record_id"])
                samples = [ds[i] for i in ids]
                raw = np.concatenate([x.Y_12lead for x in samples], -1)
                anchor = np.concatenate([x.anchor_i_ecg for x in samples], -1)
                mask = samples[0].target_quality_mask
                if split == "validation":
                    actual_validation.add(row["pair_id"])
                    if row["pair_id"] not in expected_validation:
                        raise ValueError("Machine validation pair absent from fixed G120 population")
                    with np.load(src / expected_validation[row["pair_id"]]["file"]) as original:
                        if (not np.array_equal(raw, original["target_uV"]) or not np.array_equal(anchor, original["anchor_uV"])
                                or not np.array_equal(mask, original["mask"])):
                            raise ValueError("Validation I/target/QC changed from fixed G120")
                if any(not np.array_equal(mask, x.target_quality_mask) for x in samples):
                    raise ValueError("Window QC changes within record")
                if split == "train" and (not mask[0] or not mask[1:].any()): continue
                history = history_from_sample(samples[0])
                hu = history.raw_uV if history.raw_uV is not None else np.zeros((6, 1), np.float32)
                hm = history.valid_mask if history.valid_mask is not None else np.zeros_like(hu, bool)
                demographics = meta.get(row["subject_id"])
                fields = {k: torch.from_numpy(np.asarray(demographics[k])[None].copy()).to(device) for k in FIELDS}
                with torch.no_grad():
                    native = predict_native(p, dict(anchor=torch.from_numpy(
                        (anchor.reshape(1, 6000, 10).mean(-1) / scale[0])[None]).to(device), **fields))[0].cpu().numpy()
                    glob = g(**inputs(array_batch(visible_inputs(anchor, scale[0], demographics), device)))[0].cpu().numpy()
                save("huawei_" + split, {k: row[k] for k in ("subject_id", "pair_id", "target_record_id", "input_record_id", "input_type")},
                    dict(target_uV=raw, anchor_uV=anchor, mask=mask, history_uV=hu, history_mask=hm,
                         fixed_slow=(native.astype(np.float64) + glob.astype(np.float64)) * scale[1:, None], **demographics))
                subjects[split].add(row["subject_id"])
        finally: ds.close()
    if subjects["train"] & subjects["validation"]: raise ValueError("Huawei patient leakage")
    if actual_validation != set(expected_validation): raise ValueError("Incomplete fixed machine validation population")
    if not all(records.values()): raise ValueError("Empty prepared split")
    for path in (common_config(cfg).path("subject_split_csv"), demographics_path(cfg),
                 common_config(cfg).path("device_interpretation_qc_csv")):
        assets[str(path.resolve())] = sha(path)
    study = dict(protocol=PROTOCOL, completed=True, code_sha256=code_sha(), sampling=SAMPLING,
        assets=assets, records=records, public_checkpoint=str(Path(a.public_checkpoint).resolve()),
        stable_bundle=s["base_checkpoint"], slow_run=str(src), slow_checkpoint=str(Path(a.slow_checkpoint).resolve()),
        scale_uV=scale.tolist(), model_config=cp["model_config"],
        loss=cfg["loss"], fixed_routes=dict(task1_r=report["summary"]["task1_r"], body_r=report["summary"]["body_r"]),
        stable_source_report=str(Path(a.stable_evaluation).resolve()),
        training=dict(public_epochs=a.public_epochs, huawei_epochs=a.huawei_epochs, validate_every=a.validate_every,
                      seed=43, batch_records=2, learning_rate=1e-4, weight_decay=.01, ema_decay=.999,
                      gradient_clip=1., target_sampling="one_random_window_per_target_per_epoch"),
        waveform_QC_completed=True, physical_calibration_checked=True, training_ready=True,
        context_scaling="valid_record_median / frozen_target_d12_limb_scales", slow_beta=.75)
    write(out / "study.json", study)
    write(out / "preflight.json", dict(ready=True, training_started=False,
        counts={k: len(v) for k, v in records.items()}, public_patients={s: len({r["subject_id"] for r in pairs if r["split"] == s}) for s in ("train", "validation")}))
    print({"completed": True, "training_ready": True, "shared": str(out)}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("public-checkpoint", "public-root", "slow-run", "slow-checkpoint", "stable-evaluation", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--public-epochs", type=int, default=20)
    p.add_argument("--huawei-epochs", type=int, default=20)
    p.add_argument("--validate-every", type=int, default=5)
    p.add_argument("--min-public-train", type=int, default=20)
    p.add_argument("--min-public-validation", type=int, default=5)
    a = p.parse_args()
    if min(a.public_epochs, a.huawei_epochs, a.validate_every, a.min_public_train, a.min_public_validation) < 1:
        p.error("Positive budgets and patient thresholds required")
    torch.set_num_threads(2)
    prepare(a)


if __name__ == "__main__": main()
