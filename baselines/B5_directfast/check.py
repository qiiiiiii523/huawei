"""Server preflight with a real prepared record; no optimizer updates or model saving."""
import argparse
from pathlib import Path
import numpy as np
import torch
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best.bundle import load_bundle, models
from baselines.B5_best.combine import slow_curve
from baselines.B5_best.common import write, state_sha
from .common import load_study, array_fields
from .model import build_model, highpass
from .data import TrainingWindows
from .losses import direct_objective
from .inference import predict_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new preflight report")
    study = load_study(args.run)
    device = device_from_name(args.device)
    seed_all(study["experiment"]["training"]["seed"], True)
    bundle = load_bundle(study["base_checkpoint"])
    candidate = build_model("D1", study["model_config"], bundle["fm"]["ema"], device)
    if state_sha(candidate.state_dict()) != study["initial_weights_sha256"]:
        raise ValueError("D1 initializer differs from selected FM EMA")
    dataset = TrainingWindows(args.run, study["train"], study["scale_uV"])
    batch = dataset[min(1, len(dataset) - 1)]  # centre with real neighbours when possible
    batch["inputs"] = {k: v.to(device) for k, v in batch["inputs"].items()}
    for key in ("target", "fast_target", "slow", "mask"):
        batch[key] = batch[key].to(device)
    candidate.eval()
    with torch.no_grad():
        first = candidate(**batch["inputs"])
        second = candidate(**batch["inputs"])
    if not torch.equal(first, second):
        raise ValueError("D1 eval is not deterministic")
    sequence = first.transpose(0, 1).reshape(1, 11, -1)
    actual = highpass(sequence)[0].cpu().numpy()
    # Reference H for arbitrary length with exactly the same 501/edge rule.
    x = sequence[0].double().cpu().numpy()
    padded = np.pad(x, ((0, 0), (250, 250)), mode="edge")
    sums = np.pad(padded.cumsum(-1), ((0, 0), (1, 0)))
    expected = x - (sums[:, 501:] - sums[:, :-501]) / 501
    error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=2e-6, atol=2e-6):
        raise ValueError("Differentiable H differs from deploy H")
    candidate.train()
    loss, _ = direct_objective(candidate, batch, study["experiment"]["direct_loss"])
    loss.backward()
    grads = [p.grad for p in candidate.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
        raise ValueError("No finite nonzero direct gradient")
    if any(p.grad is not None for p in candidate.anchor_head.parameters()):
        raise ValueError("Frozen I head received gradients")
    frozen_fm, frozen_p = models(bundle, device)
    frozen_fm.requires_grad_(False).eval()
    frozen_p.requires_grad_(False).eval()
    with np.load(Path(args.run) / study["train"][0]["file"], allow_pickle=False) as record:
        slow, _ = predict_frozen(frozen_fm, frozen_p, np.asarray(study["scale_uV"], np.float32),
                                record["anchor_uV"], array_fields(record), device)
        p_error = float(np.max(np.abs(slow - record["slow_uV"])))
        if not np.allclose(slow, record["slow_uV"], rtol=2e-5, atol=2e-3):
            raise ValueError("Cached P differs from real visible-input P inference")
        truth = record["target_uV"][1:]
        fast = truth.astype(float) - slow_curve(truth)
        if not np.allclose(record["fast_target"], fast, rtol=2e-5, atol=2e-3):
            raise ValueError("Cached fast target uses a different record boundary rule")
    if any(p.grad is not None for p in frozen_p.parameters()) or any(p.grad is not None for p in frozen_fm.parameters()):
        raise ValueError("Original P/I models unexpectedly received gradients")
    write(output, {"ready": True, "optimizer_updates": 0, "deterministic_D1": True,
        "nonzero_finite_gradients": True, "torch_vs_numpy_H_max_abs": error,
        "frozen_P_max_abs_uV": p_error, "training_started": False,
        "note": "Server preflight only; not a model-quality claim or exhaustive test suite."})


if __name__ == "__main__":
    main()
