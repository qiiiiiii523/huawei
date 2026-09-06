from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from ecg12gen.dataset import ECGDataConfig
from ecg12gen.models.b0_linear import B0Linear
from ecg12gen.b0_route_loss import B0RouteLoss
from ecg12gen.b0_route_data import (
    B0RouteDataset,
    load_frozen_preprocessor,
)


def first_batch(dataset):
    return next(iter(DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )))


def main():
    torch.manual_seed(42)

    config = ECGDataConfig.from_yaml(
        ROOT / "configs/common.yaml"
    )
    preprocessor = load_frozen_preprocessor(
        config,
        ROOT / "outputs/b0/preprocessing_scales.npz",
    )
    loss_fn = B0RouteLoss(ROOT / "configs/losses.yaml")

    scale = torch.tensor(
        preprocessor.scale_uV_by_source["d12"],
        dtype=torch.float32,
    )

    for task_id, channels in (("task1", 1), ("task2", 6)):
        strict_dataset = B0RouteDataset(
            config, preprocessor, task_id, "train", "strict"
        )
        weak_dataset = B0RouteDataset(
            config, preprocessor, task_id, "train", "weak"
        )

        strict_batch = first_batch(strict_dataset)
        weak_batch = first_batch(weak_dataset)

        # 从严格训练D12数据独立抽样，不按弱配对目标取reference。
        reference_loader = DataLoader(
            strict_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )
        reference = next(iter(reference_loader))["target"]

        for profile in ("strict_sync", "L0", "L1"):
            model = B0Linear(channels)
            batch = (
                strict_batch
                if profile == "strict_sync"
                else weak_batch
            )

            assert all(s == "train" for s in batch["split"])
            alignment = batch["alignment_mode"]
            assert len(set(alignment)) == 1

            flags = batch["pointwise_loss_allowed"]
            assert (flags == flags[0]).all()

            prediction = model(batch["inputs"])

            loss, logs = loss_fn(
                prediction,
                profile=profile,
                lead_mask=batch["lead_mask"],
                observed_input=batch["observed_input"],
                d12_scale_uV=scale,
                alignment_mode=alignment[0],
                pointwise_loss_allowed=bool(flags[0].item()),
                # L0不传入当前弱配对目标。
                target=(
                    None if profile == "L0"
                    else batch["target"]
                ),
                strict_reference=(
                    None if profile == "strict_sync"
                    else reference
                ),
            )

            loss.backward()
            grad = model.mapping.weight.grad

            assert grad is not None
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0

            print(f"PASS: {task_id}, {profile}")
            print({k: round(v, 6) for k, v in logs.items()})

    print("All real-batch checks passed.")


if __name__ == "__main__":
    main()