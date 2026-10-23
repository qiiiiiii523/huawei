"""Extract and validate the pinned FM EMA only. No slow model imports or loading."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from baselines.B5 import ARCHITECTURE_ID, CONDITION_SCHEMA, PREPROCESSING_VERSION
from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet
from . import PROTOCOL
from .profile import (FM_STATE_SHA, SOURCE_BUNDLE_SHA, SOURCE_FM_SHA, contract, file_sha,
                      load_profile, resolve_checkpoint, write)

FM_FIELDS = {"architecture_id", "architecture_hash", "model_config", "condition_schema",
             "preprocessing_version", "scales", "scales_sha256", "ema"}
ARCHITECTURE_SHA = "637df1317e18638abc52330edb754308bebd802803ad5bb349a853d2f52afd46"
SCALES_SHA = "8780cc7efa0e27e6cbd307b69aa0650d3a4b03b8e84923fde434929e8778dfc3"


def scales_digest(scales):
    import hashlib
    h = hashlib.sha256()
    for key in sorted(scales):
        h.update(key.encode())
        h.update(np.asarray(scales[key], dtype=np.float32).tobytes())
    return h.hexdigest()


def frozen_model(fm, device):
    model = B5UNet(ModelConfig.from_dict(fm["model_config"]))
    model.load_state_dict(fm["ema"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def state_sha(state):
    import hashlib
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        v = tensor.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def validate_fm(fm):
    if set(fm) != FM_FIELDS:
        raise ValueError("FM-only payload must contain exactly the pinned inference fields")
    if (fm["architecture_id"] != ARCHITECTURE_ID or fm["condition_schema"] != CONDITION_SCHEMA
            or fm["preprocessing_version"] != PREPROCESSING_VERSION
            or fm["architecture_hash"] != ARCHITECTURE_SHA
            or ModelConfig.from_dict(fm["model_config"]).fingerprint != fm["architecture_hash"]):
        raise ValueError("FM architecture/schema/preprocessing mismatch")
    scales = fm["scales"]
    d12 = np.asarray(scales["d12"], dtype=np.float32)
    anchor = np.asarray(scales["ecg_machine_i"], dtype=np.float32)
    if (fm["scales_sha256"] != SCALES_SHA or fm["scales_sha256"] != scales_digest(scales) or d12.shape != (12,)
            or not np.isfinite(d12).all() or np.any(d12 <= 0)
            or anchor.shape != (1,) or not np.array_equal(anchor, d12[:1])):
        raise ValueError("Invalid frozen scale")
    if state_sha(fm["ema"]) != FM_STATE_SHA:
        raise ValueError("This is not the selected FM_P_best45 EMA")
    if any(not torch.isfinite(t).all() for t in fm["ema"].values()):
        raise ValueError("Nonfinite FM parameters")


def validate_package(value, profile):
    if set(value) != {"protocol", "contract", "fm", "provenance"}:
        raise ValueError("Expected a standalone FM-only package, no slow/context/optimizer payload")
    if value["protocol"] != PROTOCOL or value["contract"] != contract(profile):
        raise ValueError("Package and frozen B5-fast profile disagree")
    source = value["provenance"]
    if (source.get("source_bundle_sha256") != SOURCE_BUNDLE_SHA
            or source.get("source_fm_checkpoint_sha256") != SOURCE_FM_SHA
            or source.get("fm_state_sha256") != FM_STATE_SHA):
        raise ValueError("Package source identity mismatch")
    validate_fm(value["fm"])


def export(source, output, profile=None):
    profile = load_profile() if profile is None else profile
    source, output = Path(source), Path(output)
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Choose a NEW package path; existing assets are retained")
    if file_sha(source) != SOURCE_BUNDLE_SHA:
        raise ValueError("Source bytes differ from the selected B5-1022 base bundle")
    bundle = torch.load(source, map_location="cpu", weights_only=True)
    fm = {k: bundle["fm"][k] for k in FM_FIELDS}
    if (bundle["fm_state_sha256"] != FM_STATE_SHA
            or bundle["provenance"]["fm_sha256"] != SOURCE_FM_SHA):
        raise ValueError("Source bundle FM provenance mismatch")
    value = {"protocol": PROTOCOL, "contract": contract(profile), "fm": fm,
             "provenance": {"source_bundle_sha256": SOURCE_BUNDLE_SHA,
                 "source_fm_checkpoint_sha256": SOURCE_FM_SHA, "fm_state_sha256": FM_STATE_SHA,
                 "candidate": "FM_P_best45", "source_fm_epoch": 45,
                 "source_commit": "69bc40f", "optimizer_updates": 0}}
    validate_package(value, profile)
    # Strict loading validates every tensor, not just its shape/hash metadata.
    frozen_model(fm, torch.device("cpu"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(value, handle)
    write(sidecar, {"protocol": PROTOCOL, "package_sha256": file_sha(output),
                    "fm_state_sha256": FM_STATE_SHA, "contract": contract(profile),
                    "provenance": value["provenance"], "slow_model_included": False})
    return value


def load(path=None, profile=None):
    profile = load_profile() if profile is None else profile
    path = resolve_checkpoint(profile, path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    validate_package(value, profile)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="The pinned B5 FM+P base.pt container")
    p.add_argument("--output", default="models/b5_fast.pt")
    args = p.parse_args()
    export(args.source, args.output)
    print({"package": str(Path(args.output).resolve()), "fm_state_sha256": FM_STATE_SHA})


if __name__ == "__main__":
    main()
