"""Public pretraining and Huawei fine-tuning for physiology-decomposed regression."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from baselines.B5.config import ModelConfig, load_config
from baselines.B5.data import HuaweiTrainDataset, demographics_table, load_preprocessor
from baselines.B5.runtime import collate, device_from_name, to_device
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from . import PROTOCOL
from .checkpoint import load as load_checkpoint, save as save_checkpoint
from .losses import aurora_loss
from .model import AuroraRegression
from .public_data import PublicDataset


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_epoch(model, loader, optimizer, device, weights, d12_scale, accumulation: int,
              gradient_clip: float) -> dict[str, float]:
    model.train()
    names = ("loss", "generation", "pcc", "source_reconstruction", "representation_consistency")
    totals = {name: 0.0 for name in names}
    optimizer.zero_grad(set_to_none=True)
    count = 0
    for number, raw in enumerate(loader, 1):
        batch = to_device(raw, device)
        loss, parts = aurora_loss(model, batch, weights, d12_scale)
        (loss / accumulation).backward()
        if number % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        totals["loss"] += float(loss.detach())
        for key, value in parts.items():
            totals[key] += float(value.detach())
        count += 1
    if count % accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if not count:
        raise ValueError("Training loader produced no batches")
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("public", "huawei"), required=True)
    parser.add_argument("--experiment-config", default="configs/b5_regression_aurora.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--study", help="Prepared public study, required for public stage")
    parser.add_argument("--huawei-config", help="Existing B5 Huawei data config")
    parser.add_argument("--init-checkpoint", help="Public AURORA checkpoint required for Huawei stage")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    experiment = read_json(args.experiment_config)
    settings = experiment[args.stage]
    if args.stage == "public" and not args.study:
        parser.error("--study is required for public pretraining")
    if args.stage == "huawei" and (not args.huawei_config or not args.init_checkpoint):
        parser.error("--huawei-config and --init-checkpoint are required for Huawei fine-tuning")
    seed_all(int(settings["seed"]))
    device = device_from_name(args.device)
    model = AuroraRegression(ModelConfig.from_dict(experiment["model"])).to(device)
    if args.stage == "huawei":
        huawei = load_config(args.huawei_config, data_only=True)
        preprocessor = load_preprocessor(huawei)
        dataset = HuaweiTrainDataset(huawei, preprocessor, demographics_table(huawei))
        payload = load_checkpoint(args.init_checkpoint, model=model, expected_stage="public")
        for key, expected in preprocessor.scale_uV_by_source.items():
            if key not in payload["scales"] or not np.array_equal(expected, payload["scales"][key]):
                raise ValueError("Public checkpoint and Huawei stage use different frozen scales")
    else:
        study = Path(args.study)
        metadata = read_json(study / "study.json")
        if metadata.get("completed") is not True:
            raise ValueError("Run prepare successfully before public training")
        if metadata.get("protocol") != PROTOCOL:
            raise ValueError(f"Study protocol must be {PROTOCOL!r}; prepare a new AURORA study")
        preprocessor = ECGPreprocessor.load(
            PreprocessingConfig.from_yaml(experiment["preprocessing_config"]), study / "scales.npz")
        dataset = ConcatDataset([
            PublicDataset(study / "public", "ptbxl", "train", preprocessor),
            PublicDataset(study / "public", "chapman", "train", preprocessor),
        ])
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=True,
                        num_workers=int(settings["workers"]), pin_memory=device.type == "cuda",
                        collate_fn=collate, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    scale = torch.as_tensor(preprocessor.scale_uV_by_source["d12"], dtype=torch.float32, device=device)
    output = Path(args.output)
    if output.exists() and any(path.name != "train.log" for path in output.iterdir()):
        raise FileExistsError("Use a new empty --output directory; runs are immutable")
    output.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs or settings["epochs"])
    for epoch in range(1, epochs + 1):
        summary = run_epoch(model, loader, optimizer, device, experiment["loss"], scale,
                            int(settings["gradient_accumulation"]), float(settings["gradient_clip"]))
        summary.update({"epoch": epoch, "stage": args.stage})
        (output / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        save_checkpoint(output / "last.pt", model=model, optimizer=optimizer, config=experiment,
                        stage=args.stage, preprocessor=preprocessor, epoch=epoch)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
