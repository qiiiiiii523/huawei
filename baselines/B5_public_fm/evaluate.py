"""Evaluate the frozen stable baseline or a trained FM with a chosen frozen slow model."""
import argparse
from pathlib import Path

import torch

from baselines.B5.checkpoint import load_checkpoint, model_from_checkpoint, atomic_save
from baselines.B5.runtime import device_from_name
from baselines.B5_best.bundle import load_bundle
from .common import load_study, read, sha, write
from .evaluation import HybridEvaluator, candidate_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--checkpoint", help="PublicFM or standard B5-U checkpoint; omitted evaluates frozen current stable FM")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    path, study = load_study(args.study)
    device = device_from_name(args.device)
    evaluator = HybridEvaluator(path, study, device)
    fixed = load_bundle(path.parent / "fixed_slow_bundle.pt")
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else fixed["fm"]
    if checkpoint["architecture_hash"] != fixed["fm"]["architecture_hash"] or checkpoint["scales_sha256"] != fixed["fm"]["scales_sha256"]:
        raise ValueError("Candidate architecture/scales must match the study")
    score = evaluator.evaluate(model_from_checkpoint(checkpoint, device), args.output)
    bundle = candidate_bundle(checkpoint, fixed, args.checkpoint, score, study["huawei_manifests"], study["huawei_sources"])
    atomic_save(Path(args.output) / "model_bundle.pt", bundle)
    write(Path(args.output) / "source.json", {"checkpoint": args.checkpoint or "frozen_stable_bundle_FM",
          "checkpoint_sha256": sha(args.checkpoint) if args.checkpoint else sha(path.parent / "fixed_slow_bundle.pt"),
          "study_sha256": sha(path), "score": score})
    print(score, flush=True)


if __name__ == "__main__":
    main()
