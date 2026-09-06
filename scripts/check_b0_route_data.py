from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from ecg12gen.dataset import ECGDataConfig
from ecg12gen.b0_route_data import (
    B0RouteDataset,
    load_frozen_preprocessor,
)


def main():
    config = ECGDataConfig.from_yaml(
        ROOT / "configs/common.yaml"
    )

    preprocessor = load_frozen_preprocessor(
        config,
        ROOT / "outputs/b0/preprocessing_scales.npz",
    )

    for task_id, channels in (("task1", 1), ("task2", 6)):
        for mode, split in (
            ("strict", "train"),
            ("weak", "train"),
            ("weak", "validation"),
        ):
            dataset = B0RouteDataset(
                config,
                preprocessor,
                task_id=task_id,
                split=split,
                mode=mode,
            )

            batch = next(iter(DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
            )))

            assert batch["inputs"].shape[1:] == (channels, 5000)
            assert batch["target"].shape[1:] == (12, 5000)
            assert batch["observed_input"].shape[1:] == (12, 5000)

            for key in ("inputs", "target", "observed_input"):
                assert torch.isfinite(batch[key]).all(), key

            if mode == "strict":
                assert torch.equal(
                    batch["inputs"],
                    batch["target"][:, :channels],
                )
                assert batch["pointwise_loss_allowed"].all()
            else:
                assert not batch["pointwise_loss_allowed"].any()

            print(
                f"PASS: {task_id}, {mode}, {split}, "
                f"windows={len(dataset)}"
            )

    print("All B0 data checks passed.")


if __name__ == "__main__":
    main()