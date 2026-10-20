"""Portable candidate inference from current I/demographics only, without study caches."""
import argparse
from pathlib import Path
import copy
import numpy as np
import torch

from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best.bundle import load_bundle, models, validate_bundle
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5.checkpoint import atomic_save
from baselines.B5.metadata import encode_demographics
from baselines.B5_final.profile import BASE_SHA, SAMPLING
from . import PROTOCOL, PACKAGE_PROTOCOL
from .common import load_study, load_candidate, code_sha
from .inference import generate_record, frozen_slow, outputs


def export(run, candidate, output):
    root, output = Path(run).resolve(), Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new inference package path")
    study = load_study(root)
    _, value = load_candidate(candidate, study, root / "study.json", torch.device("cpu"))
    bundle = copy.deepcopy(load_bundle(study["base_checkpoint"]))
    bundle["fm"]["ema"] = value["ema"]
    bundle["fm_state_sha256"] = state_sha(value["ema"])
    bundle["reference_qc_r"] = None
    validate_bundle(bundle)
    payload = {"protocol": PACKAGE_PROTOCOL, "source_base_sha256": BASE_SHA,
        "window_seconds": value["window_seconds"], "update": value["update"], "context": "disabled",
        "study_sha256": sha(root / "study.json"), "candidate_sha256": sha(candidate),
        "code_sha256": code_sha(), "sampling": SAMPLING, "slow_beta": .75,
        "bundle": bundle, "target_used_at_inference": False, "training_cache_required": False}
    atomic_save(output, payload)
    write(output.with_suffix(".json"), {k: v for k, v in payload.items() if k != "bundle"} | {"package_sha256": sha(output)})


class Predictor:
    def __init__(self, package, device):
        self.payload = torch.load(package, map_location="cpu", weights_only=True)
        p = self.payload
        if (p.get("protocol") != PACKAGE_PROTOCOL or p.get("source_base_sha256") != BASE_SHA
                or p.get("window_seconds") not in (10, 40) or p.get("context") != "disabled"
                or p.get("sampling") != SAMPLING or p.get("slow_beta") != .75
                or p.get("code_sha256") != code_sha() or p.get("target_used_at_inference") is not False
                or p.get("training_cache_required") is not False):
            raise ValueError("Invalid FM40 portable inference package")
        self.fm, self.slow = models(p["bundle"], device)
        self.slow.requires_grad_(False)
        self.device, self.seconds = device, p["window_seconds"]
        self.scale = np.asarray(p["bundle"]["fm"]["scales"]["d12"], np.float32)

    def predict_encoded(self, anchor, meta, record_key, batch_size=1):
        raw = generate_record(self.fm, anchor, meta, record_key, self.scale, self.device, self.seconds, batch_size)
        slow = frozen_slow(self.slow, anchor, meta, self.scale, self.device)
        return outputs(raw, slow)

    def predict(self, anchor, meta, record_key, batch_size=1):
        return self.predict_encoded(anchor, encode_demographics(meta), record_key, batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("package")
    pack.add_argument("--run", required=True)
    pack.add_argument("--candidate", required=True)
    pack.add_argument("--output", required=True)
    pred = commands.add_parser("predict")
    pred.add_argument("--package", required=True)
    pred.add_argument("--anchor-npy", required=True)
    pred.add_argument("--metadata-json")
    pred.add_argument("--record-key", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--mode", choices=("raw", "hybrid", "both"), default="both")
    pred.add_argument("--device", default="cuda")
    pred.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.command == "package":
        export(args.run, args.candidate, args.output)
        return
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a new prediction directory")
    seed_all(42, True)
    engine = Predictor(args.package, device_from_name(args.device))
    values = engine.predict(np.load(args.anchor_npy, allow_pickle=False),
                            read(args.metadata_json) if args.metadata_json else {}, args.record_key, args.batch_size)
    out.mkdir(parents=True)
    for mode, array in values.items():
        if args.mode in (mode, "both"):
            np.save(out / (mode + "_prediction_uV.npy"), array, allow_pickle=False)
    write(out / "inference.json", {"package_sha256": sha(args.package), "window_seconds": engine.seconds,
        "shape": [12, 60000], "sampling_rate_hz": 500, "unit": "uV", "context": "disabled",
        "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "mode": args.mode, "sampling": SAMPLING, "record_key": args.record_key,
        "target_used_at_inference": False, "training_cache_required": False})


if __name__ == "__main__":
    main()
