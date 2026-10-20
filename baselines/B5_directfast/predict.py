"""Real raw-I inference from a portable package; no cache, labels or train files."""
import argparse
from pathlib import Path
import numpy as np
import torch
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5.metadata import encode_demographics
from baselines.B5_best.bundle import models
from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.adapters import HistoryRecord
from .package import load_package
from .model import build_model
from .context_transfer import routers
from .inference import predict_frozen, predict_base


class Predictor:
    def __init__(self, package, device):
        self.payload, self.device = load_package(package), device
        value = self.payload
        candidate = value["candidate"]
        self.group = candidate["group"]
        self.model = build_model(self.group, candidate["model_config"], candidate["model"], device).eval()
        self.frozen_fm, self.frozen_p = models(value["original_bundle"], device)
        self.frozen_fm.requires_grad_(False).eval()
        self.frozen_p.requires_grad_(False).eval()
        self.scale = np.asarray(candidate["scale_uV"], np.float32)
        self.routes = routers(None, device, group=self.group, predictor_sha=value["fast_state_sha256"],
                              body_payload=value["body"], settings=value["context_settings"])

    @torch.no_grad()
    def predict_base_encoded(self, anchor_uV, demographics, record_key, batch_size=4):
        slow, original_i = predict_frozen(self.frozen_fm, self.frozen_p, self.scale, anchor_uV,
                                         demographics, self.device, batch_size)
        return predict_base(self.model, self.group, anchor_uV, demographics, record_key, self.scale,
                            slow, original_i, self.device, batch_size)

    @torch.no_grad()
    def predict(self, anchor_uV, metadata, record_key, *, task_id, input_type, history=None, route="both", batch_size=4):
        if route not in self.routes or batch_size < 1:
            raise ValueError("Invalid route or batch size")
        raw = np.asarray(anchor_uV, np.float32)
        if raw.shape == (60000,):
            raw = raw[None]
        meta = encode_demographics(metadata)
        base = self.predict_base_encoded(raw, meta, record_key, batch_size)
        return self.routes[route].apply(base, raw, task_id=task_id, input_type=input_type, history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--anchor-npy", required=True)
    parser.add_argument("--metadata-json")
    parser.add_argument("--record-key", required=True)
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--input-type", choices=("watch_ecg", "ecg_machine_d6", "body_scale_d6"), required=True)
    parser.add_argument("--route", choices=("baseline", "d6_only", "both"), default="both")
    parser.add_argument("--history-npy")
    parser.add_argument("--history-mask-npy")
    parser.add_argument("--history-record-id", default="provided_history")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if bool(args.history_npy) != bool(args.history_mask_npy) or args.batch_size < 1:
        parser.error("Supply history AND mask, and positive batch size")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new output directory")
    seed_all(42, True)
    engine = Predictor(args.package, device_from_name(args.device))
    history = (HistoryRecord(args.input_type, args.history_record_id, np.load(args.history_npy, allow_pickle=False),
                            np.load(args.history_mask_npy, allow_pickle=False)) if args.history_npy else None)
    prediction, status = engine.predict(np.load(args.anchor_npy, allow_pickle=False),
        read(args.metadata_json) if args.metadata_json else {}, args.record_key,
        task_id=args.task_id, input_type=args.input_type, history=history, route=args.route, batch_size=args.batch_size)
    output.mkdir(parents=True)
    np.save(output / "prediction_uV.npy", prediction, allow_pickle=False)
    write(output / "inference.json", {"package_sha256": sha(args.package), "group": engine.group,
        "route": args.route, "shape": list(prediction.shape), "unit": "uV", "sampling_rate_hz": 500,
        "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "context": status, "target_used_at_inference": False, "prepared_data_required_at_inference": False})


if __name__ == "__main__":
    main()
