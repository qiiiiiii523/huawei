"""Real inference: raw visible I -> frozen FM/P + slow correction -> fixed context."""
import argparse
from pathlib import Path
import numpy as np
import torch
from baselines.B5.runtime import seed_all, device_from_name
from baselines.B5.metadata import encode_demographics
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.predict import Predictor as BasePredictor
from baselines.B5_best.slow import predict_native
from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.adapters import ContextRouter, load_settings, HistoryRecord
from baselines.B5_best_context.body_model import BodyAdapter
from baselines.B5_final.profile import add_asset_arguments, assets_from_args, check_payloads, BASE_SHA, BODY_SHA
from . import PROTOCOL
from .common import load_correction, load_study
from .model import visible_inputs, inputs, correct_base
from .evaluate import array_batch


class Predictor:
    def __init__(self, assets, correction, study, study_path, device):
        self.device = device
        seed_all(42, True)
        bundle = load_bundle(assets.base)
        self.base = BasePredictor(bundle, device)
        self.model, self.payload = load_correction(correction, study, study_path, device)
        self.model.eval()
        self.body = BodyAdapter(assets.body, BASE_SHA, device, expected_group="context")
        check_payloads(bundle, self.body.payload)
        self.router = ContextRouter(load_settings(), machine_mode="fir25", body_adapter=self.body)

    @torch.no_grad()
    def predict(self, anchor, metadata, record_key, task_id, input_type, history=None, batch_size=8):
        # No target, QC label, cache or future target enters either model.
        meta = encode_demographics(metadata)
        base = self.base.predict_encoded(anchor, meta, record_key, batch_size)
        fields = visible_inputs(anchor, self.base.scale[0], meta)
        batch = array_batch(fields, self.device)
        batch["base_slow"] = predict_native(self.base.slow, {"anchor": batch["anchor50"],
            "numeric": batch["numeric"], "sex": batch["sex"], "field_mask": batch["field_mask"],
            "age_topcoded": batch["age_topcoded"]})
        delta = self.model(**inputs(batch))[0].cpu().numpy()
        corrected = correct_base(base, delta, self.base.scale)
        return self.router.apply(corrected, anchor, task_id=task_id, input_type=input_type, history=history)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_asset_arguments(p)
    p.add_argument("--run", required=True, help="Prepared shared directory; only study.json read, no labels needed")
    p.add_argument("--correction", required=True)
    p.add_argument("--anchor-npy", required=True)
    p.add_argument("--metadata-json")
    p.add_argument("--record-key", required=True)
    p.add_argument("--task-id", choices=("task1", "task2"), required=True)
    p.add_argument("--input-type", choices=("watch_ecg", "ecg_machine_d6", "body_scale_d6"), required=True)
    p.add_argument("--history-npy")
    p.add_argument("--history-mask-npy")
    p.add_argument("--history-record-id", default="provided_history")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if bool(args.history_npy) != bool(args.history_mask_npy):
        p.error("Supply both raw history and mask or neither")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a new inference output")
    study = load_study(args.run, verify_records=False)
    assets = assets_from_args(args)
    engine = Predictor(assets, args.correction, study, Path(args.run) / "study.json", device_from_name(args.device))
    history = (HistoryRecord(args.input_type, args.history_record_id, np.load(args.history_npy, allow_pickle=False),
        np.load(args.history_mask_npy, allow_pickle=False)) if args.history_npy else None)
    prediction, status = engine.predict(np.load(args.anchor_npy, allow_pickle=False),
        read(args.metadata_json) if args.metadata_json else {}, args.record_key, args.task_id,
        args.input_type, history, args.batch_size)
    out.mkdir(parents=True)
    np.save(out / "prediction_uV.npy", prediction, allow_pickle=False)
    write(out / "inference.json", {"protocol": PROTOCOL, "group": engine.payload["group"],
        "epoch": engine.payload["epoch"], "base_sha256": BASE_SHA, "body_sha256": BODY_SHA,
        "correction_sha256": sha(args.correction), "shape": list(prediction.shape), "unit": "uV",
        "sampling": study["sampling"], "slow_beta": .75, "context": status, "target_used_at_inference": False})


if __name__ == "__main__":
    main()
