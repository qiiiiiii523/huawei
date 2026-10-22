"""Select immutable rate-best sources and define separate follow-up arms."""
import argparse
from pathlib import Path
import numpy as np
import torch

from baselines.B5.runtime import seed_all
from baselines.B5_best.common import write, sha, state_sha
from baselines.B5_slowrate.common import load_study as load_rates
from . import PROTOCOL, GROUPS
from .common import code_sha, source_checkpoints, initialize


def prepare(args):
    source, output = Path(args.rate_run).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new follow-up shared directory")
    parent = load_rates(source / "shared")
    sources = source_checkpoints(source, parent)
    reference = args.reference_group
    if reference == "best":
        reference = max(sources, key=lambda g: sources[g]["summary"]["mean_r"])
    groups = tuple(args.groups.split(","))
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError("Unknown/duplicate follow-up groups")
    seed = parent["training"]["seed"] if args.training_seed is None else args.training_seed
    rates = {g: args.fusion_learning_rate if g.startswith("FUSION") else args.learning_rate for g in groups}
    context = getattr(args, "p_context_seconds", 100)
    dilation = getattr(args, "p_extra_dilation", 256)
    rf_seconds = (2045 + 8 * dilation) / 50
    if context not in (60, 80, 100, 120) or dilation < 1 or rf_seconds + 10 > context:
        raise ValueError("Choose context >= expanded RF + central10s; e.g. S100/d256 or S120/d384")
    if (min(args.epochs, args.validate_every, args.batch_records) < 1 or not 0 <= seed < 2**32
            or any(not np.isfinite(v) or v <= 0 for v in rates.values())
            or not np.isfinite(args.p_preservation) or args.p_preservation < 0):
        raise ValueError("Invalid follow-up training budget/learning rate/preservation")
    study = {**parent, "protocol": PROTOCOL, "completed": True, "code_sha256": code_sha(),
        "rate_shared": str(source / "shared"), "rate_study_sha256": sha(source / "shared/study.json"),
        "sources": sources, "reference_group": reference, "reference_Final": parent["reference"],
        "reference": sources[reference]["summary"], "groups": groups, "tail_blocks": args.tail_blocks,
        "p_context_seconds": context, "p_extra_dilation": dilation,
        "p_preservation": args.p_preservation,
        "training": {**parent["training"], "seed": seed, "epochs": args.epochs,
                     "validate_every": args.validate_every, "batch_records": args.batch_records},
        "learning_rates": rates,
        "P_designs": {"PTAIL": {"context_seconds": 50, "rf_seconds": 40.9},
                      "PCTX": {"context_seconds": context, "rf_seconds": 40.9},
                      "PCTX_RF": {"context_seconds": context, "rf_seconds": rf_seconds, "extra_dilation": dilation}},
        "initialization_note": "PCTX/PCTX_RF may change epoch0 due to P-window execution; report actual scores",
        "initial_weights_sha256": {}}
    counts = {}
    for group in groups:
        seed_all(seed, True)
        model = initialize(group, study, "cpu")
        study["initial_weights_sha256"][group] = state_sha(model.state_dict())
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        counts[group] = {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                         "trainable_names": names, "learning_rate": rates[group]}
    output.mkdir(parents=True)
    write(output / "study.json", study)
    write(output / "preflight.json", {"ready": True, "training_started": False,
        "reference_group": reference, "reference": study["reference"], "groups": counts,
        "P_designs": study["P_designs"], "no_new_FM_P_cache": True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rate-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference-group", choices=("best", "R2p5", "R5", "R10"), default="best")
    p.add_argument("--groups", default=",".join(GROUPS))
    p.add_argument("--training-seed", type=int)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--validate-every", type=int, default=1)
    p.add_argument("--batch-records", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--fusion-learning-rate", type=float, default=1e-4)
    p.add_argument("--p-preservation", type=float, default=.1)
    p.add_argument("--tail-blocks", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--p-context-seconds", type=int, choices=(60, 80, 100, 120), default=100)
    p.add_argument("--p-extra-dilation", type=int, default=256)
    args = p.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    prepare(args)


if __name__ == "__main__":
    main()
