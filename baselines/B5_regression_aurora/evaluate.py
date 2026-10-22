"""Official Huawei raw-uV record/QC evaluation for the baseline I-only route."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.B5.config import ModelConfig, load_config
from baselines.B5.data import HuaweiValidationDataset, demographics_table, load_preprocessor
from baselines.B5.runtime import collate, device_from_name, to_device
from baselines.B5.scoring import diagnostic_summary, scores_from_summaries, write_score_report
from ecg12gen.evaluate import evaluate_record_predictions, write_report
from . import ARCHITECTURE
from .checkpoint import load as load_checkpoint
from .model import AuroraRegression


@torch.no_grad()
def evaluate(model, config, preprocessor, device, output_root: Path, batch_size: int) -> dict:
    model.eval()
    demographics = demographics_table(config)
    scale_numpy = preprocessor.scale_uV_by_source["d12"][None, :, None]
    scale_tensor = torch.as_tensor(preprocessor.scale_uV_by_source["d12"],
                                   dtype=torch.float32, device=device)
    summaries = {}
    for task in ("task1", "task2"):
        dataset = HuaweiValidationDataset(config, task, preprocessor, demographics)
        task_output = output_root / task
        task_output.mkdir(parents=True, exist_ok=True)
        shape = (len(dataset), 12, 5000)
        prediction = np.lib.format.open_memmap(task_output / "prediction_uV.npy", mode="w+",
                                                dtype=np.float32, shape=shape)
        target = np.lib.format.open_memmap(task_output / "target_uV.npy", mode="w+",
                                           dtype=np.float32, shape=shape)
        quality = np.lib.format.open_memmap(task_output / "target_quality_mask.npy", mode="w+",
                                            dtype=bool, shape=(len(dataset), 12))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        rows, offset = [], 0
        for batch in loader:
            inputs = to_device({key: batch[key] for key in
                                ("anchor", "numeric", "sex", "field_mask", "age_topcoded")}, device)
            generated = model(inputs["anchor"], inputs["numeric"], inputs["sex"], inputs["field_mask"],
                              inputs["age_topcoded"], scale_tensor).full12
            size = len(generated)
            prediction[offset:offset + size] = generated.cpu().numpy() * scale_numpy
            target[offset:offset + size] = batch["target_uV"]
            quality[offset:offset + size] = batch["quality_mask"].numpy()
            rows.extend(batch["evaluation_metadata"]); offset += size
        prediction.flush(); target.flush(); quality.flush()
        overall, details = evaluate_record_predictions(
            prediction, target, task, rows, target_quality_mask=quality,
            input_quality_mask=quality[:, 0], require_reliable_input=True,
            input_quality_source="same_record_target_I")
        overall = diagnostic_summary(overall)
        overall.update({"route": "baseline", "source_domain": "Huawei", "architecture": ARCHITECTURE,
                        "prediction_mode": "single_forward_physiology_two_stage"})
        write_report(task_output, overall, details,
                     title=f"B5 AURORA-inspired {task} raw-uV record evaluation")
        with (task_output / "window_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader(); writer.writerows(rows)
        summaries[task] = overall
    score = scores_from_summaries(summaries)
    score.update({"route": "baseline", "architecture": ARCHITECTURE,
                  "prediction_mode": "single_forward_physiology_two_stage"})
    write_score_report(output_root, score)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-config", default="configs/b5_regression_aurora.json")
    parser.add_argument("--huawei-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    experiment = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    config = load_config(args.huawei_config, data_only=True)
    preprocessor = load_preprocessor(config)
    device = device_from_name(args.device)
    model = AuroraRegression(ModelConfig.from_dict(experiment["model"])).to(device)
    payload = load_checkpoint(args.checkpoint, model=model, expected_stage="huawei")
    for key, expected in preprocessor.scale_uV_by_source.items():
        if key not in payload["scales"] or not np.array_equal(expected, payload["scales"][key]):
            raise ValueError("Checkpoint and evaluation use different frozen scales")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty --output directory")
    output.mkdir(parents=True, exist_ok=True)
    score = evaluate(model, config, preprocessor, device, output, args.batch_size)
    print(json.dumps(score, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
