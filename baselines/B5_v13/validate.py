"""Explicit evaluation of experimental best/last with fixed S50-F and Heun16/K16."""
import argparse

from baselines.B5.runtime import device_from_name
from .common import load_experiment_checkpoint, datasets_for
from .evaluation import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = device_from_name(args.device)
    cp, study, bundle, preprocessor, model = load_experiment_checkpoint(args.checkpoint, args.study, device)
    _, datasets = datasets_for(study, bundle, preprocessor)
    evaluate(model, bundle, datasets, device, args.output, cp["group"], cp["epoch"])


if __name__ == "__main__":
    main()
