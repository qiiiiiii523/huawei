"""Reuse immutable G120 caches; initialize H/P from the exact same P and G120."""
import argparse
from pathlib import Path
import numpy as np
import torch

from baselines.B5.checkpoint import atomic_save
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_slow120.common import load_study as load_source, load_correction
from . import PROTOCOL
from .common import code_sha
from .model import JointSlow
from .evaluate import Evaluator


def prepare(a):
    out, source = Path(a.output).resolve(), Path(a.source_run).resolve()
    if out.exists():
        raise FileExistsError("Use a NEW shared directory; existing source results are never modified")
    original = load_source(source)
    bundle = load_bundle(original["base_checkpoint"])
    device = device_from_name(a.device)
    seed_all(a.seed, True)
    g, cp = load_correction(a.g120_checkpoint, original, source / "study.json", device)
    if cp["group"] != "G120" or not cp.get("validation"):
        raise ValueError("Use the selected G120 best.pt with its measured validation scores")
    model = JointSlow(original["model"], a.window_batch).to(device)
    model.local.load_state_dict(bundle["slow"]["model"], strict=True)
    model.global_model.load_state_dict(g.state_dict(), strict=True)
    model.eval()
    public = None
    if a.public_cache:
        public_root = Path(a.public_cache).resolve()
        m = read(public_root / "public.json")
        if m.get("protocol") != PROTOCOL or m.get("completed") is not True or m.get("scales_uV") != original["scale_uV"]:
            raise ValueError("Prepare the real120 public cache with this exact Final bundle first")
        parts = [{e["subject_id"] for e in m[s]} for s in ("train", "validation")]
        if not all(parts) or parts[0] & parts[1]:
            raise ValueError("Public patients must have independent nonempty train/validation splits")
        public = {"root": str(public_root), "sha256": sha(public_root / "public.json")}
    out.mkdir(parents=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    atomic_save(out / "initializer.pt", dict(protocol=PROTOCOL, model=state))
    study = dict(original)
    study.update(protocol=PROTOCOL, completed=True, source_run=str(source), source_study_sha256=sha(source / "study.json"),
                 source_g120_checkpoint=str(Path(a.g120_checkpoint).resolve()), source_g120_sha256=sha(a.g120_checkpoint),
                 source_g120_epoch=cp["epoch"], reference=cp["validation"], public=public,
                 initializer_sha256=sha(out / "initializer.pt"), initial_weights_sha256=state_sha(state),
                 code_sha256=code_sha(), P_optimizer_updates="trainable in H/P", FM_optimizer_updates=0,
                 training=dict(epochs=a.epochs, public_epochs=a.public_epochs, seed=a.seed, batch_records=a.batch_records,
                    window_batch=a.window_batch, cnn_learning_rate=a.cnn_lr, global_learning_rate=a.global_lr,
                    weight_decay=.01, gradient_clip=1., validate_every=a.validate_every),
                 note="Local P and G120 jointly trainable. Source cache/P/FM/context/beta unchanged. Epoch0 retained.")
    # A full target-QC score reproduction, not just a synthetic shape check.
    evaluator = Evaluator(out, study, device)
    try:
        row = evaluator.evaluate(model, out / "initial_evaluation", "INITIAL", 0)
    finally:
        evaluator.close()
    differences = {k: row[k] - study["reference"][k] for k in ("task1_r", "task2_r", "machine_r", "body_r", "mean_r")}
    matched = all(abs(v) < 2e-6 for v in differences.values())
    write(out / "preflight.json", dict(ready=matched, training_started=False, optimizer_updates=0,
          source_g120_epoch=cp["epoch"], initial_score=row, source_score=study["reference"],
          score_differences=differences, tolerance=2e-6, same_initial_weights=True,
          CNN_parameters=sum(p.numel() for p in model.local.parameters()),
          G120_parameters=sum(p.numel() for p in model.global_model.parameters()),
          FM_fixed=True, context_fixed=True, slow_beta=.75,
          public_available=bool(public), original_cache_copied=False))
    if not matched:
        raise ValueError("Initial joint forward does not reproduce the source G120; inspect preflight.json")
    write(out / "study.json", study)
    print({"ready": True, "training_started": False, "reference_mean_r": row["mean_r"],
           "H_available": True, "P_available": bool(public), "shared": str(out)}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True, help="Existing G120 shared/ directory")
    p.add_argument("--g120-checkpoint", required=True)
    p.add_argument("--public-cache", help="New B5_slowjoint.public output, not the old 50..120s cache")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--public-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--batch-records", type=int, default=2)
    p.add_argument("--window-batch", type=int, default=2)
    p.add_argument("--cnn-lr", type=float, default=1e-5)
    p.add_argument("--global-lr", type=float, default=1e-4)
    p.add_argument("--validate-every", type=int, default=5)
    a = p.parse_args()
    if (min(a.epochs, a.public_epochs, a.batch_records, a.window_batch, a.validate_every) < 1
            or not 0 <= a.seed < 2**32 or not np.isfinite([a.cnn_lr, a.global_lr]).all()
            or min(a.cnn_lr, a.global_lr) <= 0):
        p.error("Need positive budgets and finite learning rates, uint32 seed")
    torch.set_num_threads(2)
    prepare(a)


if __name__ == "__main__":
    main()
