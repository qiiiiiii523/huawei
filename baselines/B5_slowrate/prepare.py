"""Derive only low-rate visible inputs; reuse all frozen FM/P/target caches."""
import argparse
from pathlib import Path
import numpy as np
import torch

from baselines.B5.runtime import seed_all
from baselines.B5_best.common import write, sha, state_sha
from baselines.B5_slow120 import FIELDS
from baselines.B5_slow120.common import load_study as load_parent
from . import PROTOCOL, RATES
from .common import code_sha, within
from .model import RateCorrection, visible_inputs, inputs, array_batch


def prepare(args):
    source, output = Path(args.source_shared).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new shared directory; incomplete sidecars are not reused")
    parent = load_parent(source)
    cfg = dict(parent["training"])
    for name in ("epochs", "validate_every", "batch_records", "learning_rate"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    if args.training_seed is not None:
        cfg["seed"] = args.training_seed
    if (min(cfg[k] for k in ("epochs", "validate_every", "batch_records")) < 1
            or not np.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0
            or not 0 <= cfg["seed"] < 2**32):
        raise ValueError("Need positive training budgets/LR and uint32 seed")
    output.mkdir(parents=True)
    entries = {"train": [], "validation": []}
    for split in entries:
        for entry in parent[split]:
            with np.load(within(source, entry["file"]), allow_pickle=False) as raw:
                meta = {k: raw[k] for k in FIELDS}
                rates = {group: visible_inputs(raw["anchor_uV"], parent["scale_uV"][0], meta, group)["global_i"]
                         for group in RATES}
                if not np.array_equal(rates["R5"], raw["global_i"]):
                    raise AssertionError("5Hz preprocessing differs from original G120")
            path = output / split / f"{len(entries[split]):05d}.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez(path, **rates)
            entries[split].append({**entry, "source_file": entry["file"], "source_sha256": entry["sha256"],
                                  "file": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha(path)})
            print({"phase": "rate_inputs", "split": split, "records": len(entries[split]),
                   "record": entry["target_record_id"]}, flush=True)
    initial = []
    model_cfg = dict(parent["model"])
    for group in RATES:
        seed_all(cfg["seed"], True)
        model = RateCorrection(group, **model_cfg).eval()
        initial.append(state_sha(model.state_dict()))
        with np.load(within(source, parent["train"][0]["file"]), allow_pickle=False) as data:
            fields = visible_inputs(data["anchor_uV"], parent["scale_uV"][0], data, group)
        with torch.no_grad():
            delta = model(**inputs(array_batch(fields, "cpu")))
        if delta.shape != (1, 11, 6000) or torch.count_nonzero(delta).item():
            raise AssertionError("All rate arms must start with exactly zero full120 correction")
    if len(set(initial)) != 1:
        raise AssertionError("All three rate arms must have identical initial parameter bytes")
    keys = ("score_protocol", "base_checkpoint", "base_sha256", "body_checkpoint", "body_sha256", "config",
            "config_sha256", "baseline_evaluation", "population", "sampling", "slow_beta", "scale_uV",
            "reference", "validation_rows", "train_subjects", "validation_subjects", "data_sources", "loss")
    study = {**{k: parent[k] for k in keys}, "protocol": PROTOCOL, "completed": True,
        "code_sha256": code_sha(), "source_shared": str(source), "source_study_sha256": sha(source / "study.json"),
        "rates_hz": RATES, "model": model_cfg, "training": cfg, "initial_weights_sha256": initial[0], **entries,
        "position_units": "original_5Hz_ticks_at_equal_physical_time",
        "initialization": "same_fresh_G120_parameters_zero_output_head_not_trained_G120_continuation",
        "comparison": "same_patients_parameters_batches_updates_loss_selection_different_rate_tokens_and_compute",
        "FM_optimizer_updates": 0, "P_optimizer_updates": 0, "body_optimizer_updates": 0}
    write(output / "study.json", study)
    write(output / "preflight.json", {"ready": True, "same_initial_weights": True,
        "original_5Hz_inputs_bit_equal": True, "zero_initial_output": True,
        "parameters_per_arm": sum(p.numel() for p in model.parameters()),
        "tokens_per_arm": {g: int(120 * h) for g, h in RATES.items()}, "training": cfg,
        "parent_FM_P_caches_reused": True, "training_started": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-shared", required=True, help="Existing unmodified B5_slow120 shared directory")
    p.add_argument("--output", required=True)
    p.add_argument("--training-seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--validate-every", type=int)
    p.add_argument("--batch-records", type=int)
    p.add_argument("--learning-rate", type=float)
    args = p.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    prepare(args)


if __name__ == "__main__":
    main()
