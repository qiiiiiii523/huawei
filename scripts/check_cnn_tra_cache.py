"""Small real-data integration check for 120-s grouping and seven-lead cache."""
from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.cnn_tra_data import HuaweiSevenLeadData, TASKS
from ecg12gen.cnn_tra_model import CNNTransformerSevenLead
from scripts.train_cnn_tra import validate_and_cache


def main() -> None:
    data = HuaweiSevenLeadData("/root/autodl-tmp/huawei_data")
    for task in TASKS:
        rows = data.validation_rows[task]
        # One complete pair for Task1; one of each device for Task2.
        first_pairs = {}
        for row in rows:
            first_pairs.setdefault(row["input_type"], row["pair_id"])
        selected = set(first_pairs.values())
        indices = [i for i, row in enumerate(rows) if row["pair_id"] in selected]
        data.validation_arrays[task] = np.asarray(data.validation_arrays[task][indices])
        data.validation_rows[task] = [rows[i] for i in indices]
        data.validation_masks[task] = data.validation_masks[task][indices]
        assert len(indices) == (12 if task == "task1" else 24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransformerSevenLead().to(device)
    with TemporaryDirectory(prefix="cnn_tra_cache_check_", dir="/root/autodl-tmp") as root:
        output = Path(root)
        summary = validate_and_cache(model, data, device, output, 0, 4, "synthetic_model")
        assert np.isfinite(summary["r_7lead"])
        assert np.isfinite(summary["rmse_7lead_uV"])
        for task in TASKS:
            task_dir = output / "step_000000" / task
            prediction = np.load(task_dir / "prediction_7_uV.npy")
            target = np.load(task_dir / "target_7_uV.npy")
            anchor = np.load(task_dir / "input_i_uV.npy")
            assert prediction.shape == target.shape == (len(data.validation_rows[task]), 7, 5000)
            assert anchor.shape == (len(prediction), 1, 5000)
            assert np.isfinite(prediction).all()
        assert (output / "step_000000" / "COMPLETE").exists()
    print("CACHE_CHECK_OK")


if __name__ == "__main__":
    main()
