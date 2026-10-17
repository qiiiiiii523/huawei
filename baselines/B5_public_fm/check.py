"""CPU functional/data preflight. No sampling loop, backward pass or optimizer."""
import argparse

import numpy as np
import torch

from baselines.B5.checkpoint import load_checkpoint, checkpoint_preprocessor, model_from_checkpoint
from baselines.B5.data import preprocessing_config
from baselines.B5.runtime import build_datasets, INPUT_FIELDS, collate, to_device, seed_all
from baselines.B5.flow import linear_path
from baselines.B5_best.bundle import load_bundle
from .common import read, write, sha, state_sha, load_study
from .public_data import PublicDataset


def check(study_path, report):
    path, study = load_study(study_path)
    root = path.parent
    torch.set_num_threads(2)
    seed_all(42, True)
    checkpoint = load_checkpoint(root / "public_initializer.pt")
    fixed = load_bundle(root / "fixed_slow_bundle.pt")
    config = read(root / "data_config.json")
    preprocessor = checkpoint_preprocessor(checkpoint, preprocessing_config(config))
    train, validation = build_datasets(config, preprocessor)
    items = [PublicDataset(root / "public", source, "train", preprocessor)[0] for source in ("ptbxl", "chapman")]
    items.append(train[0])
    for item in items:
        if item["anchor"].shape != (1, 5000) or item["target"].shape != (12, 5000):
            raise ValueError("Input/target shape mismatch")
        if not np.array_equal(item["target"][:1], item["anchor"]):
            raise ValueError("Target I and visible anchor must share scale without centering")
    batch = to_device(collate(items[:1]), torch.device("cpu"))
    model = model_from_checkpoint(checkpoint, torch.device("cpu"))
    before = state_sha(model.state_dict())
    time = torch.full((1,), 0.5)
    state, _ = linear_path(batch["target"][:, 1:], batch["quality_mask"][:, 1:], torch.zeros_like(batch["target"][:, 1:]), time)
    with torch.no_grad():
        velocity, anchor = model(state, time, **{k: batch[k] for k in INPUT_FIELDS})
    if not torch.isfinite(velocity).all() or velocity.shape != (1, 11, 5000) or anchor.shape != (1, 1, 5000):
        raise ValueError("CPU full-window forward failed")
    if before != state_sha(model.state_dict()) or before != study["public_source"]["ema_sha256"]:
        raise ValueError("Preflight changed weights or initializer mismatch")
    result = {"ready": True, "training_started": False, "optimizer_updates": 0,
              "CPU_full_window_velocity_shape": list(velocity.shape), "same_public_initial_weights": True,
              "initial_EMA_sha256": before, "fixed_slow_state_sha256": fixed["slow_state_sha256"],
              "train_windows": len(train), "validation_windows": {t: len(ds) for t, ds in validation.items()},
              "sampling": study["settings"]["sampling"], "slow_beta": study["settings"]["slow_beta"],
              "expanded_chapman_probability": study["expanded_chapman_probability"], "score_protocol": study["score_protocol"],
              "note": "Functional preflight, not an ECG performance result"}
    write(report, result)
    print(result, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    check(args.study, args.report)


if __name__ == "__main__":
    main()
