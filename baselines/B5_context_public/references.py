"""Re-evaluate public initial and stable machine with paired K16 before training."""
import argparse
from pathlib import Path
import torch
from baselines.B5.checkpoint import model_from_checkpoint, atomic_save
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import write, sha
from baselines.B5.runtime import device_from_name, seed_all
from .data import load_study
from .common import create_model
from .evaluate import evaluate
from .sampling import validate_machine_references


def run(a):
    root, out = Path(a.run), Path(a.output)
    if out.exists(): raise FileExistsError("Use a new references directory")
    study = load_study(root); out.mkdir(parents=True)
    device = device_from_name(a.device); reports = {}
    bundle = load_bundle(study["stable_bundle"])
    stable = model_from_checkpoint(bundle["fm"], device)
    reports["stable_machine"] = evaluate(stable, root, study, out / "stable_machine", checkpoint_sha=sha(study["stable_bundle"]))
    del stable
    for name, arm in (("public_initial", "control"), ("control_initial", "control"), ("context_public_initial", "context_public")):
        seed_all(study["training"]["seed"], True)
        model = create_model(study, arm, device).eval()
        reports[name] = evaluate(model, root, study, out / name, arm == "context_public", checkpoint_sha=sha(study["public_checkpoint"]))
        del model
    validate_machine_references(reports)
    a0, b0 = reports["control_initial"], reports["context_public_initial"]
    if abs(a0["machine_r"] - b0["machine_r"]) > 2e-6:
        raise ValueError("V6 zero-initialized conditions failed full-record score equivalence")
    write(out / "references.json", dict(completed=True, study_sha256=sha(root / "study.json"),
        reports=reports, source_initial_equivalence=True, tolerance=2e-6))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "output"): p.add_argument("--" + k, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(); torch.set_num_threads(2); run(a)


if __name__ == "__main__": main()
