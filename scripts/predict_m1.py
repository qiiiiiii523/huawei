"""Formal M1 inference: explicit context plus explicit machine-I anchor only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.contracts import ContractError, prepare_joint_anchor_inference
from ecg12gen.m1_data import transform_context_window
from ecg12gen.m1_model import M1Model
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def _npy_batch(path: str, name: str) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[-1] != 5000:
        raise ContractError(f"{name} must have shape [N,C,5000]")
    return values


def _indices(value: str | None) -> tuple[int, ...]:
    if value is None:
        return tuple(range(6))
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(set(result)) != len(result) or any(item not in range(6) for item in result):
        raise ContractError("context-lead-indices must be unique canonical d6 indices")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--anchor-npy", required=True, help="Explicit machine-I anchor [N,1,5000]")
    parser.add_argument("--watch-npy", default=None, help="Task 1 watch context [N,1,5000]")
    parser.add_argument("--d6-npy", default=None, help="Task 2 one d6 context [N,6/5,5000]")
    parser.add_argument("--context-source-type", choices=("ecg_machine_d6", "body_scale_d6"), default=None)
    parser.add_argument("--context-lead-indices", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.task_id == "task1" and (args.watch_npy is None or args.d6_npy is not None):
        parser.error("task1 requires --watch-npy and forbids --d6-npy")
    if args.task_id == "task2" and (args.d6_npy is None or args.context_source_type is None or args.watch_npy is not None):
        parser.error("task2 requires one --d6-npy and --context-source-type, and forbids watch context")

    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else checkpoint_path.parent
    config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    import json
    scale_values = json.loads((run_dir / "preprocessing_scales.json").read_text(encoding="utf-8"))
    preprocessor = ECGPreprocessor(config, {key: np.asarray(value, dtype=np.float32) for key, value in scale_values.items()})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("task_id") != args.task_id:
        raise ContractError("checkpoint task_id does not match inference task")
    model = M1Model(fusion_mode=str(checkpoint.get("fusion_mode", "none")),
                    transformer_layers=int(checkpoint.get("transformer_layers", 4))).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    anchor_raw = _npy_batch(args.anchor_npy, "anchor-npy")
    if anchor_raw.shape[1] != 1:
        raise ContractError("--anchor-npy must contain exactly one machine-I channel")
    anchor_model = preprocessor.transform_batch(anchor_raw, "ecg_machine_i")[0]
    if args.task_id == "task1":
        context_raw = _npy_batch(args.watch_npy, "watch-npy")
        context_model = preprocessor.transform_batch(context_raw, "watch_ecg")[0]
        source = "watch_ecg"
        context_mask = None
        indices = None
    else:
        context_raw = _npy_batch(args.d6_npy, "d6-npy")
        indices = _indices(args.context_lead_indices)
        if context_raw.shape[1] != len(indices):
            raise ContractError("d6-npy channel count does not match context-lead-indices")
        context_mask_np = np.zeros((len(context_raw), 6), dtype=bool)
        context_mask_np[:, list(indices)] = True
        context_model = np.stack([transform_context_window(preprocessor, row, args.context_source_type, context_mask_np[i])
                                   for i, row in enumerate(context_raw)])
        context_mask = torch.from_numpy(context_mask_np)
        source = args.context_source_type
    if context_raw.shape[0] != anchor_raw.shape[0]:
        raise ContractError("context and anchor batch sizes must match")
    # This public contract deliberately has no target argument and is checked
    # per item before entering the model.
    for row_context, row_anchor in zip(context_raw, anchor_raw):
        prepare_joint_anchor_inference(row_context, context_source_type=source, task_id=args.task_id,
                                       anchor_i_ecg=row_anchor, context_channel_indices=indices)
    with torch.no_grad():
        anchor_tensor = torch.from_numpy(anchor_model).to(args.device)
        context_tensor = torch.from_numpy(context_model).to(args.device)
        prediction_model = model(anchor_tensor, context_ecg=context_tensor,
                                 context_source_type=source, context_lead_mask=context_mask.to(args.device) if context_mask is not None else None)
        baseline = model.predict_baseline(anchor_tensor).cpu().numpy()
    prediction_raw = prediction_model.cpu().numpy() * preprocessor.scale_uV_by_source["d12"][None, :, None] + baseline[:, :, None]
    prediction_submit = prediction_raw.copy()
    prediction_submit[:, :1] = anchor_raw
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "prediction_raw.npy", prediction_raw.astype(np.float32))
    np.save(output / "prediction_submit.npy", prediction_submit.astype(np.float32))
    print(f"Wrote M1 raw and submit predictions to {output}")


if __name__ == "__main__":
    main()
