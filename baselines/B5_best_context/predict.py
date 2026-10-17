"""Frozen B5-best inference followed by explicit task/device routing; no hidden target input."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .adapters import ContextRouter, HistoryRecord, DEFAULT_CONFIG, MODES, load_settings


class Predictor:
    def __init__(self, bundle, device, *, settings=None, machine_mode=None, body_adapter=None):
        from baselines.B5_best.predict import Predictor as BasePredictor
        self.base = BasePredictor(bundle, device)
        self.router = ContextRouter(settings, machine_mode=machine_mode, body_adapter=body_adapter)

    def predict(self, anchor_uV, metadata, record_key, *, task_id, input_type,
                history=None, batch_size=8):
        base = self.base.predict(anchor_uV, metadata, record_key, batch_size)
        return self.router.apply(base, anchor_uV, task_id=task_id, input_type=input_type, history=history)

    def predict_encoded(self, anchor_uV, demographics, record_key, *, task_id, input_type,
                        history=None, batch_size=8):
        base = self.base.predict_encoded(anchor_uV, demographics, record_key, batch_size)
        return self.router.apply(base, anchor_uV, task_id=task_id, input_type=input_type, history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--profile", default=str(Path(__file__).resolve().parents[2] / "configs/b5_best.json"))
    parser.add_argument("--context-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--machine-mode", choices=MODES)
    parser.add_argument("--anchor-npy", required=True, help="Raw current I[1,60000], 500Hz uV")
    parser.add_argument("--metadata-json")
    parser.add_argument("--record-key", required=True)
    parser.add_argument("--task-id", required=True, choices=("task1", "task2"))
    parser.add_argument("--input-type", required=True, choices=("watch_ecg", "ecg_machine_d6", "body_scale_d6"))
    parser.add_argument("--history-npy", help="Raw physical history[6,T] in uV, no centering/scaling")
    parser.add_argument("--history-mask-npy", help="Boolean validity[6,T], including input QC and gaps")
    parser.add_argument("--history-record-id", default="provided_history")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if bool(args.history_npy) != bool(args.history_mask_npy):
        parser.error("Supply both history and validity mask, or neither")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory; keep shell tee logs outside it")
    from baselines.B5_best.bundle import load_inference_bundle
    from baselines.B5_best.common import read, write, sha
    from baselines.B5.runtime import device_from_name, seed_all
    bundle, checkpoint = load_inference_bundle(args.checkpoint, args.profile)
    seed_all(bundle["sampling"]["seed"], True)
    history = None if not args.history_npy else HistoryRecord(
        args.input_type, args.history_record_id, np.load(args.history_npy, allow_pickle=False),
        np.load(args.history_mask_npy, allow_pickle=False))
    settings = load_settings(args.context_config)
    engine = Predictor(bundle, device_from_name(args.device), settings=settings, machine_mode=args.machine_mode)
    prediction, status = engine.predict(np.load(args.anchor_npy, allow_pickle=False),
        read(args.metadata_json) if args.metadata_json else {}, args.record_key,
        task_id=args.task_id, input_type=args.input_type, history=history, batch_size=args.batch_size)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "prediction_uV.npy", prediction, allow_pickle=False)
    write(out / "inference.json", {"checkpoint_sha256": sha(checkpoint), "sampling": bundle["sampling"],
        "slow_beta": engine.base.slow_beta, "task_id": args.task_id, "input_type": args.input_type,
        "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "unit": "uV", "sampling_rate_hz": 500, "context": status, "settings": settings,
        "target_used_at_inference": False, "training_started": False})
    print(status, flush=True)


if __name__ == "__main__":
    main()
