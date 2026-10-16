"""CPU-only pair/data preflight. Does not generate FM cache or train."""
import argparse
from pathlib import Path

import numpy as np
import torch

from baselines.B5.checkpoint import checkpoint_preprocessor
from baselines.B5.data import preprocessing_config
from baselines.B5.runtime import build_datasets
from baselines.B5.metadata import encode_demographics
from baselines.B5_best import SAMPLING, SCORE_PROTOCOL
from baselines.B5_best.bundle import DEFAULT_PROFILE, export_bundle, bundle_slow_beta
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_best.slow import predict_native
from baselines.B5_best.combine import upsample_native
from baselines.B5_best.diagnose import target_file
from baselines.B5_best.validate import verify_reference_inputs
from . import PROTOCOL
from .common import code_digest, slow_model
from .core import generate, upsample


def compatibility(bundle):
    model = slow_model(bundle, torch.device("cpu"), trainable=False).eval()
    fields = {k: torch.from_numpy(v[None]) for k, v in encode_demographics({"age": 40}).items()}
    inputs = {"anchor": torch.sin(torch.arange(6000) / 30).reshape(1, 1, 6000), **fields}
    with torch.no_grad():
        old, new = predict_native(model, inputs), generate(model, inputs)
        difference = float((old - new).abs().max())
        restored = new.double() * torch.tensor(bundle["fm"]["scales"]["d12"][1:])[None, :, None].double()
        expected = upsample_native(restored[0].numpy())
        error = float(np.max(np.abs(upsample(restored)[0].numpy() - expected)))
    if difference != 0. or error > 1e-7:
        raise ValueError("Differentiable slow inference/interpolation differs from deployment")
    return {"native_prediction_max_abs_difference": difference,
            "interpolation_max_abs_difference_uV": error,
            "initial_slow_state_sha256": state_sha(model.state_dict()),
            "slow_parameters": sum(p.numel() for p in model.parameters())}


def prepare(root, output, epochs=20, combo_weight=.1):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use empty shared output; tee logs beside it")
    if epochs < 1 or not np.isfinite(combo_weight) or combo_weight <= 0:
        raise ValueError("Need positive epochs and a positive fixed combo PCC weight")
    output.mkdir(parents=True, exist_ok=True)
    profile = read(DEFAULT_PROFILE)
    bundle = export_bundle(profile["fm_checkpoint"], profile["slow_checkpoint"], profile["prepared_info"],
                           output / "components.pt", profile)
    old_slow = torch.load(profile["slow_checkpoint"], map_location="cpu", weights_only=True)
    source = old_slow["config"]
    expected = {"context_seconds": 50, "group": "F", "model_channels": 48,
                "huber_weight": 1., "pcc_weight": .1, "mean_weight": 0., "residual_width_native": 51}
    if any(source.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported source S50-F loss/network configuration")
    weight = float(source["residual_weight"])
    if not .01 <= weight <= 10 or weight != source["calibrated_weights"]["residual"]:
        raise ValueError("Invalid source fixed F coefficient")
    info = read(profile["prepared_info"])
    config = read(Path(profile["fm_checkpoint"]).parent / "resolved_config.json")
    config["repository_root"] = str(root)
    checks = info.get("source_checks", {})
    required = {"strict_index", "demographics_csv", "subject_split_csv", "device_interpretation_qc_csv",
                "task1_train_target", "task2_train_target", "task1_validation_target", "task2_validation_target"}
    if not required.issubset(checks):
        raise ValueError("Original data fingerprints are incomplete")
    for key, item in checks.items():
        print("Checking original data: " + key, flush=True)
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Original data changed: " + key)
    verify_reference_inputs(bundle, config)
    preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(config))
    train, validation = build_datasets(config, preprocessor)
    manifests = {"train": train.manifest_digest, **{k: ds.manifest_digest for k, ds in validation.items()}}
    if any(bundle["provenance"]["source_manifests"].get(k) != v for k, v in manifests.items()):
        raise ValueError("Train/validation selection differs from fixed B5 run")
    if {r["subject_id"] for r in train.rows} & {r["subject_id"] for ds in validation.values() for r in ds.rows}:
        raise ValueError("Subject leakage")
    # Reuse the known fixed FM validation ensemble, NOT as training samples.
    evaluation = Path(info["evaluation_dir"])
    settings = read(evaluation / "inference_sampling.json")
    if settings["sampling"] != SAMPLING or Path(settings["checkpoint"]).resolve() != Path(profile["fm_checkpoint"]).resolve():
        raise ValueError("Historical FM validation predictions use another checkpoint/sampling")
    evaluation_checks = {}
    for task in ("task1", "task2"):
        folder = evaluation / task
        for path in (folder / "prediction_uV.npy", target_file(folder), folder / "window_metadata.csv"):
            evaluation_checks[str(path)] = sha(path)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        checked = compatibility(bundle)
    finally:
        torch.set_num_threads(threads)
    training = {"epochs": epochs, "batch_records": int(source["batch_records"]),
                "learning_rate": float(source["learning_rate"]), "weight_decay": float(source["weight_decay"]),
                "seed": 42, "workers": 0, "gradient_clip": 1., "validate_every": 5}
    if training["batch_records"] < 1 or training["learning_rate"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("Invalid original slow training settings")
    study = {"protocol": PROTOCOL, "score_protocol": SCORE_PROTOCOL, "sampling": dict(SAMPLING),
             "beta": bundle_slow_beta(bundle), "combo_weight": combo_weight, "training": training,
             "residual_weight": weight, "residual_width_native": 51, "mean_weight": 0.,
             "bundle_sha256": sha(output / "components.pt"), "code_sha256": code_digest(),
             "source_checks": checks, "data_config": config, "manifests": manifests,
             "source_validation_evaluation": str(evaluation), "source_evaluation_checks": evaluation_checks,
             "source_slow_config": source, "source_provenance": bundle["provenance"],
             "train_windows": len(train), "train_records": len({r["target_record_id"] for r in train.rows}),
             "validation_unique_targets": len({r["target_record_id"] for ds in validation.values() for r in ds.rows})}
    write(output / "study.json", study)
    write(output / "preflight.json", {"ready": True, "training_started": False, "FM_optimizer_updates": 0,
          "FM_cache_generated": False, "sampling": dict(SAMPLING), "training": training,
          "fixed_beta_both_arms": bundle_slow_beta(bundle),
          "old_F_coefficient": weight, "extra_combo_PCC_weight": combo_weight, **checked})
    print({"ready": True, "study": str(output / "study.json"), **checked}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", default="/home/qht/huawei")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--combo-weight", type=float, default=.1)
    args = p.parse_args()
    prepare(args.repository_root, args.output, args.epochs, args.combo_weight)


if __name__ == "__main__":
    main()
