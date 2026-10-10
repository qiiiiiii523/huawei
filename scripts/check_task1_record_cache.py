"""Audit an independent Task1 cache; does not rebuild data or train."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.evaluate import evaluate_record_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/common.yaml"))
    args = parser.parse_args()
    config = ECGDataConfig.from_yaml(args.config)
    datasets = [JointAnchorDataset(config, "task1", split) for split in ("train", "validation")]
    try:
        for ds in datasets:
            for sample in ds:
                sample.validate()
                assert sample.context_time_mask.shape == sample.context_ecg.shape
        validation = datasets[1]
        rows = [validation._rows[i] for i in validation._indices]
        if not rows:
            raise AssertionError("No validation targets")
        target = np.stack([validation[i].Y_12lead for i in range(len(validation))])
        identity, _ = evaluate_record_predictions(target, target, "task1", rows)
        assert np.isclose(identity["r_missing11"], 1) and identity["missing11_mean_rmse_uV"] == 0
        print("PASS: independent Task1 cache; 120-second targets, separate full contexts, subject split and identity score;",
              "train_windows=", len(datasets[0]), "validation_windows=", len(validation), "validation_pairs=", identity["n_records"])
    finally:
        for ds in datasets:
            ds.close()


if __name__ == "__main__":
    main()
