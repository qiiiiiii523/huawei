"""Experimental hybrid inference: observed 120s I and demographics only."""
import argparse
from pathlib import Path

import numpy as np

from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best.common import read, write, sha
from .common import load_experiment_checkpoint
from .evaluation import ExperimentalPredictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--anchor-npy", required=True, help="uV, 500Hz, 120s, [1,60000] or [60000]")
    p.add_argument("--metadata-json")
    p.add_argument("--record-key", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty inference output")
    device = device_from_name(args.device)
    cp, study, bundle, _, model = load_experiment_checkpoint(args.checkpoint, args.study, device,
                                                            verify_data=False)
    seed_all(42, True)
    prediction = ExperimentalPredictor(bundle, device, model).predict(
        np.load(args.anchor_npy, allow_pickle=False),
        read(args.metadata_json) if args.metadata_json else {}, args.record_key)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "prediction_uV.npy", prediction, allow_pickle=False)
    write(output / "inference.json", {"checkpoint_sha256": sha(args.checkpoint),
          "group": cp["group"], "epoch": cp["epoch"], "sampling": study["sampling"],
          "beta": study["beta"], "shape": list(prediction.shape), "target_used_at_inference": False,
          "training_or_validation_target_files_read": False})


if __name__ == "__main__":
    main()
