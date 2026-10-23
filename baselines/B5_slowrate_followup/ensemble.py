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
    if len(roots) != len(groups) or len(groups) < 2:
        raise ValueError("Use >=2 members, with one shared rate-run or one run per member")
    device = device_from_name(args.device)
    parent = load_rates(roots[0] / "shared")
    studies = {root: load_rates(root / "shared") for root in dict.fromkeys(roots)}
    selected = {root: source_checkpoints(root, study) for root, study in studies.items()}
    # Cross-seed ensembles must beat the strongest verified single from ANY
    # contributing run, not just the first run on the command line.
    reference_root, reference = max(((root, g) for root in studies for g in RATES),
        key=lambda item: selected[item[0]][item[1]]["summary"]["mean_r"])
    models, members = [], []
    for root, group in zip(roots, groups):
        study = studies[root]
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
             "reference_Final": parent["reference"], "reference": selected[reference_root][reference]["summary"]}
    evaluator = Evaluator(roots[0] / "shared", study, device)
    try:
        row = evaluator.evaluate(FixedEnsemble(models).to(device), args.output,
                                 "MEAN_" + "_".join(groups), 0, keep_predictions=args.keep_predictions)
    finally:
        evaluator.close()
    write(Path(args.output) / "ensemble.json", {"completed": True, "members": members,
        "summary": row, "reference_best_single": study["reference"],
        "reference_rate_run": str(reference_root),
        "delta_best_single": row["mean_r"] - study["reference"]["mean_r"],
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
