"""Bind an existing control100 cache and PTB data to one immutable CMP study."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch

from baselines.B5_best.bundle import DEFAULT_PROFILE, load_inference_bundle, load_bundle, bundle_slow_beta
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_best import SAMPLING
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best.validate import pair_indices
from baselines.B5_best.slow import predict_native
from baselines.B5.metadata import encode_demographics
from . import PROTOCOL
from .common import empty_output, verify_entries, read_record, validate_record, code_digest, model_from_bundle
from .core import generate, FIELDS, upsample


def source_study(cache, manifest, explicit=None):
    candidates = ([Path(explicit)] if explicit else
                  [cache.parent / "shared/study.json", cache.parent / "study.json",
                   cache.parent.parent / "shared/study.json"])
    for path in candidates:
        if path.is_file() and sha(path) == manifest.get("study_sha256"):
            return path.resolve(), read(path)
    raise ValueError("Cannot prove cache identity. Pass --cache-study PATH to the study.json whose SHA is in cache.json")


def check_populations(huawei, public):
    train_subjects = {e["subject_id"] for e in huawei["train"]}
    val_subjects = {e["subject_id"] for e in huawei["validation"]}
    if train_subjects & val_subjects:
        raise ValueError("Huawei patient leakage")
    if len({e["target_record_id"] for e in huawei["train"]}) != len(huawei["train"]):
        raise ValueError("Huawei training must contain unique physical targets")
    if {e["subject_id"] for e in public["train"]} & {e["subject_id"] for e in public["validation"]}:
        raise ValueError("Public patient leakage")
    if ({e["raw_wave_sha256"] for e in public["train"]} &
            {e["raw_wave_sha256"] for e in public["validation"]}):
        raise ValueError("Public duplicate waveform crosses splits")
    lookup = {e["cache_key"]: e for e in huawei["validation"]}
    if len(lookup) != len(huawei["validation"]) or set(huawei["validation_rows"]) != {"task1", "task2"}:
        raise ValueError("Validation task/pair identity mismatch")
    for task, rows in huawei["validation_rows"].items():
        for indices in pair_indices(rows):
            row = rows[indices[0]]
            entry = lookup[task + ":" + str(row["pair_id"])]
            if (entry["target_record_id"] != row["target_record_id"] or
                    entry["subject_id"] != row["subject_id"]):
                raise ValueError("Validation cache and row identities differ")


def compatibility(bundle):
    model = model_from_bundle(bundle, torch.device("cpu")).eval()
    fields = {k: torch.from_numpy(v[None]) for k, v in encode_demographics({"age": 40}).items()}
    inputs = {"anchor": torch.sin(torch.arange(6000) / 30).reshape(1, 1, 6000), **fields}
    with torch.no_grad():
        expected = predict_native(model.requires_grad_(False), inputs)
        actual = generate(model, inputs)
        difference = float((expected - actual).abs().max())
    if difference != 0:
        raise ValueError("Trainable S50 forward differs from deployed path")
    return {"native_max_abs_difference": difference, "initial_slow_state_sha256": state_sha(model.state_dict()),
            "parameters": sum(p.numel() for p in model.parameters())}


def prepare(bundle_path, cache, public_root, output, cache_study=None, u1=1560, u2=1560, validate_every=390):
    if min(u1, u2, validate_every) < 1:
        raise ValueError("Positive update budgets and validation interval required")
    bundle, initial_path = load_inference_bundle(bundle_path)
    profile = read(DEFAULT_PROFILE)
    if (bundle_slow_beta(bundle) != .75 or bundle["sampling"] != SAMPLING or
            bundle["provenance"].get("slow_variant") != "slow_f" or
            bundle["provenance"].get("fm_sha256") != profile["components"]["fm_sha256"]):
        raise ValueError("Need the current control100 + compatible S50-F bundle, beta=.75")
    cache, public_root = Path(cache).resolve(), Path(public_root).resolve()
    manifest, public = read(cache / "cache.json"), read(public_root / "public.json")
    if (not manifest.get("completed") or not public.get("completed") or
            manifest.get("sampling") != SAMPLING or public.get("protocol") != PROTOCOL):
        raise ValueError("Incomplete cache/public manifest or wrong Heun16/K16")
    source_path, source = source_study(cache, manifest, cache_study)
    source_bundle_path = source_path.parent / "components.pt"
    if sha(source_bundle_path) != source["bundle_sha256"]:
        raise ValueError("Source cache component package changed")
    source_bundle = load_bundle(source_bundle_path)
    if (source_bundle["fm_state_sha256"] != bundle["fm_state_sha256"] or
            source_bundle["fm"]["scales_sha256"] != bundle["fm"]["scales_sha256"] or
            public["bundle_scales_sha256"] != bundle["fm"]["scales_sha256"]):
        raise ValueError("The cache uses another FM/scales: original-FM cache is NOT control100")
    # Original data/QC changes must not be silently paired with stale cached labels.
    for key, item in source.get("source_checks", {}).items():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Huawei data/QC changed; rebuild its cache: " + key)
    check_populations(manifest, public)
    scales = np.asarray(bundle["fm"]["scales"]["d12"], np.float32)
    for root, values, is_huawei in ((cache, manifest, True), (public_root, public, False)):
        verify_entries(root, values["train"] + values["validation"])
        for part in ("train", "validation"):
            for entry in values[part]:
                validate_record(read_record(root, entry), scales, huawei=is_huawei, training=(part == "train"))
    validation_inputs = {}
    for entry in manifest["validation"]:
        record = read_record(cache, entry)
        key = entry["target_record_id"]
        if key in validation_inputs:
            previous = read_record(cache, validation_inputs[key])
            for field in ("anchor", "numeric", "sex", "field_mask", "age_topcoded", "target_uV", "slow_target", "mask"):
                if not np.array_equal(record[field], previous[field]):
                    raise ValueError("Repeated target changes visible inputs/labels/QC: " + key)
        else:
            validation_inputs[key] = entry
    output = empty_output(output)
    shutil.copyfile(initial_path, output / "components.pt")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        checked = compatibility(bundle)
    finally:
        torch.set_num_threads(threads)
    study = {"protocol": PROTOCOL, "score_protocol": SCORE_PROTOCOL, "code_sha256": code_digest(),
        "bundle_sha256": sha(output / "components.pt"), "initial_bundle_source": str(initial_path),
        "huawei": {"root": str(cache), "manifest": str(cache / "cache.json"), "sha256": sha(cache / "cache.json")},
        "public": {"root": str(public_root), "manifest": str(public_root / "public.json"), "sha256": sha(public_root / "public.json")},
        "external_checks": [{"path": str(source_path), "sha256": sha(source_path)},
                            {"path": str(source_bundle_path), "sha256": sha(source_bundle_path)}],
        "sampling": dict(SAMPLING), "beta": .75, "residual_weight": 2.5890171538829843,
        "F_width_native": 51, "combo_weight": .1, "mean_weight": 0.,
        "training": {"stage1_updates": u1, "stage2_updates": u2, "validate_every": validate_every,
                     "batch_records": 2, "seed": 42, "learning_rate": 1e-4, "weight_decay": .01, "gradient_clip": 1.},
        "selection": "initial_bundle_or_stage2_initial_or_stage2_periodic_mean_r",
        "stage1_handoff": "fixed_final_update_not_validation_best",
        "public_patient_balanced": True, "M_batch": ["huawei", "public"],
        "source_cache_study": str(source_path), "source_F_weight": source.get("residual_weight"), **checked}
    if source.get("residual_weight") != study["residual_weight"]:
        raise ValueError("Source S50-F coefficient differs; do not guess a new F weight")
    write(output / "study.json", study)
    write(output / "preflight.json", {"ready": True, "training_started": False, "FM_optimizer_updates": 0,
          "score_protocol": SCORE_PROTOCOL, "baseline_must_be_recomputed_in_current_QC_scope": True,
          "train_records": len(manifest["train"]), "public_train_records": len(public["train"]), **checked})
    print({"ready": True, "study": str(output / "study.json"), "training_started": False}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", help="Portable bundle; omitted uses the pinned B5-best default")
    p.add_argument("--huawei-cache", default="/home/qht/huawei/results/B5-best-exp-SlowPCC/control100_pcc20_cache_v1/cache")
    p.add_argument("--cache-study", help="Source study.json proving the reused FM cache identity")
    p.add_argument("--public", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage1-updates", type=int, default=1560)
    p.add_argument("--stage2-updates", type=int, default=1560)
    p.add_argument("--validate-every", type=int, default=390)
    a = p.parse_args()
    prepare(a.bundle, a.huawei_cache, a.public, a.output, a.cache_study,
            a.stage1_updates, a.stage2_updates, a.validate_every)


if __name__ == "__main__":
    main()
