"""Equal-weight residual ensemble, before the single fixed context application."""
import argparse
from pathlib import Path
from baselines.B5.runtime import device_from_name
from baselines.B5_best.common import write, sha
from baselines.B5_slowrate.common import load_study as load_rates, load_correction
from baselines.B5_slowrate import RATES
from .common import source_checkpoints
from .model import FixedEnsemble
from .evaluate import Evaluator


def run(args):
    roots = [Path(p).resolve() for p in args.rate_runs]
    groups = args.groups
    if len(roots) == 1:
        roots *= len(groups)
    if len(roots) != len(groups) or not groups:
        raise ValueError("Use one rate-run for all groups, or one run per member")
    device = device_from_name(args.device)
    parent = load_rates(roots[0] / "shared")
    selected = source_checkpoints(roots[0], parent)
    reference = max(selected, key=lambda g: selected[g]["summary"]["mean_r"])
    models, members = [], []
    for root, group in zip(roots, groups):
        study = load_rates(root / "shared")
        for field in ("source_study_sha256", "population", "scale_uV", "sampling", "base_sha256", "body_sha256"):
            if study[field] != parent[field]:
                raise ValueError("Ensemble members must share the same immutable observed-data/base study")
        path = root / group / "best.pt"
        model, cp = load_correction(path, study, root / "shared/study.json", device)
        if cp["group"] != group:
            raise ValueError("Ensemble member group mismatch")
        models.append(model.eval())
        members.append({"path": str(path), "sha256": sha(path), "group": group,
                        "seed": study["training"]["seed"], "weight": 1. / len(groups)})
    study = {**parent, "rate_shared": str(roots[0] / "shared"), "reference_group": reference,
             "reference_Final": parent["reference"], "reference": selected[reference]["summary"]}
    evaluator = Evaluator(roots[0] / "shared", study, device)
    try:
        row = evaluator.evaluate(FixedEnsemble(models).to(device), args.output,
                                 "MEAN_" + "_".join(groups), 0, keep_predictions=args.keep_predictions)
    finally:
        evaluator.close()
    write(Path(args.output) / "ensemble.json", {"completed": True, "members": members,
        "summary": row, "reference_best_single": study["reference"],
        "equal_weights_fixed_before_validation": True, "context_applied_once": True,
        "no_patient_or_validation_fitted_weights": True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rate-runs", nargs="+", required=True)
    p.add_argument("--groups", nargs="+", choices=tuple(RATES), default=list(RATES))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--keep-predictions", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
