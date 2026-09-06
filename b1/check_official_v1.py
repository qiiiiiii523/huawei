"""Read-only acceptance checks for the B1 official-v1 runner."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from b1.model import LightweightUNet
from b1.official_v1 import (
    ExperimentSpec,
    fit_preprocessor,
    load_pseudo,
    load_strict_raw,
    load_weak,
    pseudo_loss,
    read_matrix,
    strict_loss,
    transform_bundle,
    transform_strict,
    weak_loss,
)
from ecg12gen.training import load_training_protocol

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    specs = read_matrix()
    assert len(specs) == 14
    assert sum(spec.task == "task1" for spec in specs) == 6
    assert sum(spec.task == "task2" for spec in specs) == 8
    assert {(spec.route, spec.loss_variant) for spec in specs if spec.task == "task1"} == {
        ("A", "L0"), ("A", "L1"), ("B", "L0"), ("B", "L1"), ("C1", None), ("C2", None)}

    fixed = load_training_protocol(ROOT / "configs" / "training_protocol_v1.yaml")["fair_baseline_comparison"]
    assert fixed["seed"] == 42 and fixed["batch_size"] == 16 and fixed["epochs_per_stage"] == 100
    assert fixed["optimizer"] == "AdamW" and fixed["learning_rate"] == 0.001
    assert fixed["weight_decay"] == 0.0001 and fixed["scheduler"] == "none"
    assert fixed["gradient_clip_norm"] is None and fixed["early_stopping"]["enabled"] is False
    assert fixed["adapter"] == "disabled" and fixed["baseline_head"] == "disabled_for_first_comparison"

    strict1 = load_strict_raw("task1")
    train1, val1 = load_weak("task1", "train", "A"), load_weak("task1", "validation", "A")
    assert (len(train1.x), len(val1.x), len(strict1.x)) == (882, 213, 962)
    assert len(load_pseudo("C1").x) == 18 and len(load_pseudo("C2").x) == 33

    task2_counts = {}
    for variant in ("A", "B"):
        train2, val2 = load_weak("task2", "train", variant), load_weak("task2", "validation", variant)
        task2_counts[variant] = {
            "train": {source: sum(row["input_type"] == source for row in train2.metadata)
                      for source in ("ecg_machine_d6", "body_scale_d6")},
            "validation": {source: sum(row["input_type"] == source for row in val2.metadata)
                           for source in ("ecg_machine_d6", "body_scale_d6")}}
    assert task2_counts["A"] == task2_counts["B"]

    pre = fit_preprocessor("task1", train1, strict1)
    strict = transform_strict(strict1, pre, "task1")
    weak = transform_bundle(train1, pre, "task1")
    pseudo = transform_bundle(load_pseudo("C1"), pre, "task1")
    model = LightweightUNet(1)
    pred = model(torch.from_numpy(weak.x[:2]))
    s_total, _ = strict_loss(model(torch.from_numpy(strict.x[:2])), torch.from_numpy(strict.y[:2]),
                             torch.from_numpy(strict.observed[:2]), 1)
    w_total, _ = weak_loss(pred, torch.from_numpy(weak.y[:2]), torch.from_numpy(weak.observed[:2]),
                           torch.from_numpy(strict.y[:2]), 1, "L1")
    p_pred = model(torch.from_numpy(pseudo.x[:2]))
    p_total, _ = pseudo_loss(p_pred, torch.from_numpy(pseudo.y[:2]), torch.from_numpy(pseudo.observed[:2]),
                             torch.from_numpy(pseudo.quality[:2]), 1)
    assert all(torch.isfinite(value) for value in (s_total, w_total, p_total))
    assert sum(parameter.numel() for parameter in model.parameters()) == 163004
    print(json.dumps({"status": "PASS", "experiments": len(specs), "task2_counts": task2_counts,
                      "strict_windows": len(strict1.x), "c1_windows": 18, "c2_windows": 33,
                      "model_parameters_task1": 163004}, ensure_ascii=False))


if __name__ == "__main__":
    main()
