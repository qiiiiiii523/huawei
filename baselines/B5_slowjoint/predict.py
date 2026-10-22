"""New-record inference reads I, demographics and optional legal history, never labels."""
import argparse
from pathlib import Path
import numpy as np
import torch
from baselines.B5.runtime import seed_all, device_from_name
from baselines.B5.metadata import encode_demographics
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.predict import Predictor as BasePredictor
from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.adapters import ContextRouter, load_settings, HistoryRecord
from baselines.B5_best_context.body_model import BodyAdapter
from baselines.B5_final.profile import check_payloads, BASE_SHA, BODY_SHA, SAMPLING
from baselines.B5_slow120.prepare import native_prediction
from baselines.B5_slow120.model import visible_inputs, correct_base
from . import PROTOCOL
from .model import visible_batch
from .common import load_model


class Predictor:
    def __init__(self, run, checkpoint, device):
        # Inference reads only study metadata and weights, not caches, targets or QC.
        self.root, self.device = Path(run).resolve(), device
        self.study = read(self.root / "study.json")
        if self.study.get("protocol") != PROTOCOL or self.study.get("sampling") != SAMPLING:
            raise ValueError("Wrong joint model manifest")
        seed_all(42, True)
        for key, expected in (("base_checkpoint", BASE_SHA), ("body_checkpoint", BODY_SHA)):
            if sha(self.study[key]) != expected:
                raise ValueError("Selected frozen inference weight changed")
        bundle = load_bundle(self.study["base_checkpoint"])
        self.base = BasePredictor(bundle, device)
        self.model, self.payload = load_model(checkpoint, self.study, self.root / "study.json", device)
        self.model.eval()
        self.body = BodyAdapter(self.study["body_checkpoint"], BASE_SHA, device, expected_group="context")
        check_payloads(bundle, self.body.payload)
        self.router = ContextRouter(load_settings(), machine_mode="fir25", body_adapter=self.body)

    @torch.no_grad()
    def predict(self, anchor, metadata, record_key, task_id, input_type, history=None, batch_size=8):
        meta = encode_demographics(metadata)
        plain = self.base.predict_encoded(anchor, meta, record_key, batch_size)
        fields = {**visible_inputs(anchor, self.base.scale[0], meta), "anchor_uV": anchor}
        total = self.model(**visible_batch(fields, self.device, self.base.scale[0]))[0].cpu().numpy()
        original = native_prediction(self.base, anchor, meta)
        result = correct_base(plain, total - original, self.base.scale)
        return self.router.apply(result, anchor, task_id=task_id, input_type=input_type, history=history)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Only shared/study.json is needed, no prepared data")
    p.add_argument("--checkpoint", required=True)
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
    a = p.parse_args()
    if bool(a.history_npy) != bool(a.history_mask_npy):
        p.error("Supply history and history QC together")
    output = Path(a.output)
    if output.exists():
        raise FileExistsError("Choose a new inference output")
    engine = Predictor(a.run, a.checkpoint, device_from_name(a.device))
    history = (HistoryRecord(a.input_type, a.history_record_id, np.load(a.history_npy, allow_pickle=False),
        np.load(a.history_mask_npy, allow_pickle=False)) if a.history_npy else None)
    result, status = engine.predict(np.load(a.anchor_npy, allow_pickle=False),
        read(a.metadata_json) if a.metadata_json else {}, a.record_key, a.task_id, a.input_type, history)
    output.mkdir(parents=True)
    np.save(output / "prediction_uV.npy", result, allow_pickle=False)
    write(output / "inference.json", dict(protocol=PROTOCOL, checkpoint_sha256=sha(a.checkpoint),
        shape=list(result.shape), unit="uV", sampling=SAMPLING, slow_beta=.75, context=status,
        target_used_at_inference=False, prepared_data_required=False, group=engine.payload["group"],
        fusion_mode=engine.payload.get("fusion_mode", "add")))


if __name__ == "__main__":
    main()
