"""Selected two-network B5 plus fixed per-device context, with target-free inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write
from baselines.B5_best_context.adapters import HistoryRecord
from .profile import (BASE_SHA, add_asset_arguments, assets_from_args, resolve_assets, check_payloads)


class Predictor:
    def __init__(self, device, *, assets=None):
        from baselines.B5_best.bundle import load_bundle
        from baselines.B5_best_context.body_model import BodyAdapter
        from baselines.B5_best_context.predict import Predictor as ContextPredictor
        from baselines.B5.runtime import seed_all
        self.assets = assets or resolve_assets()
        bundle = load_bundle(self.assets.base)
        body = BodyAdapter(self.assets.body, BASE_SHA, device, expected_group="context")
        check_payloads(bundle, body.payload)
        seed_all(bundle["sampling"]["seed"], True)
        self.engine = ContextPredictor(bundle, device, machine_mode="fir25", body_adapter=body)

    def predict(self, anchor_uV, metadata, record_key, *, task_id, input_type, history=None, batch_size=8):
        # Keep the already evaluated FM I head and all existing fallbacks unchanged.
        return self.engine.predict(anchor_uV, metadata, record_key, task_id=task_id,
                                   input_type=input_type, history=history, batch_size=batch_size)

    def predict_encoded(self, anchor_uV, demographics, record_key, *, task_id, input_type,
                        history=None, batch_size=8):
        return self.engine.predict_encoded(anchor_uV, demographics, record_key, task_id=task_id,
                                           input_type=input_type, history=history, batch_size=batch_size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_asset_arguments(p)
    p.add_argument("--anchor-npy", required=True, help="Current raw I[1,60000] or [60000], 500Hz uV")
    p.add_argument("--metadata-json")
    p.add_argument("--record-key", required=True, help="Stable key: changing it changes the fixed-seed FM noise")
    p.add_argument("--task-id", required=True, choices=("task1", "task2"))
    p.add_argument("--input-type", required=True, choices=("watch_ecg", "ecg_machine_d6", "body_scale_d6"))
    p.add_argument("--history-npy", help="RAW unfiltered physical historical six leads [6,T] in uV")
    p.add_argument("--history-mask-npy", help="Boolean validity [6,T], including lead QC and gaps")
    p.add_argument("--history-record-id", default="provided_history")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if bool(args.history_npy) != bool(args.history_mask_npy):
        p.error("Supply both raw history and its validity mask, or neither")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a new inference output directory; keep tee logs outside it")
    assets = assets_from_args(args)
    from baselines.B5.runtime import device_from_name
    engine = Predictor(device_from_name(args.device), assets=assets)
    history = None if not args.history_npy else HistoryRecord(args.input_type, args.history_record_id,
        np.load(args.history_npy, allow_pickle=False), np.load(args.history_mask_npy, allow_pickle=False))
    prediction, status = engine.predict(np.load(args.anchor_npy, allow_pickle=False),
        read(args.metadata_json) if args.metadata_json else {}, args.record_key,
        task_id=args.task_id, input_type=args.input_type, history=history, batch_size=args.batch_size)
    out.mkdir(parents=True)
    np.save(out / "prediction_uV.npy", prediction, allow_pickle=False)
    write(out / "inference.json", {**assets.report(), "shape": list(prediction.shape),
        "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "unit": "uV", "sampling_rate_hz": 500, "task_id": args.task_id, "input_type": args.input_type,
        "record_key": args.record_key, "context": status})
    print({"prediction": str(out / "prediction_uV.npy"), "context": status}, flush=True)


if __name__ == "__main__":
    main()
