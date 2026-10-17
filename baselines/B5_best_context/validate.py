"""Generate base predictions once, or reuse a completed reliable-I evaluation; run three CPU postprocess groups."""
from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import DEFAULT_CONFIG, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Original repository's data/run config")
    parser.add_argument("--output-dir", required=True, help="New run root; tee logs must be outside it")
    parser.add_argument("--baseline-evaluation", help="Completed B5-best reliable-I evaluation with raw window arrays")
    parser.add_argument("--checkpoint", help="Original B5-best bundle, only when generating the baseline")
    parser.add_argument("--profile", default=str(Path(__file__).resolve().parents[2] / "configs/b5_best.json"))
    parser.add_argument("--context-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.baseline_evaluation and args.checkpoint:
        parser.error("A reused baseline already pins the checkpoint; do not provide --checkpoint")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty run directory; keep tee logs outside it")
    settings = load_settings(args.context_config)
    from .evaluate import load_baseline, run_study, HistoryProvider
    from baselines.B5_best.common import write
    baseline_dir = Path(args.baseline_evaluation).resolve() if args.baseline_evaluation else output / "base_inference"
    if args.baseline_evaluation:
        # Fail before any output is created when an old/incomplete report is supplied.
        loaded = load_baseline(baseline_dir)
    else:
        from baselines.B5_best.bundle import load_inference_bundle
        from baselines.B5_best.validate import evaluate
        from baselines.B5_best.common import sha
        from baselines.B5.config import load_config, ModelConfig
        from baselines.B5.checkpoint import checkpoint_preprocessor
        from baselines.B5.data import preprocessing_config
        from baselines.B5.runtime import device_from_name, build_validation_datasets
        bundle, checkpoint = load_inference_bundle(args.checkpoint, args.profile)
        sampling = bundle["sampling"]
        if (sampling.get("solver") != "heun" or sampling.get("steps") != 16
                or sampling.get("samples") != 16 or bundle.get("slow_beta") != .75):
            raise ValueError("Use the unchanged B5-best Heun16/K16 beta=0.75 bundle")
        config = load_config(args.config, data_only=True)
        if ModelConfig.from_dict(config["model"]).fingerprint != bundle["fm"]["architecture_hash"]:
            raise ValueError("Data config and bundled FM architecture differ")
        preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(config))
        datasets = build_validation_datasets(config, preprocessor)
        source = bundle["provenance"].get("source_manifests", {})
        if any(source.get(task) and source[task] != ds.manifest_digest for task, ds in datasets.items()):
            raise ValueError("Validation target population differs from pinned B5-best source")
        evaluate(bundle, datasets, device_from_name(args.device), baseline_dir, sha(checkpoint), args.batch_size)
        loaded = load_baseline(baseline_dir)
    provider = HistoryProvider(args.config, loaded[1]["task2"])
    try:
        rows = run_study(baseline_dir, output / "study", provider, settings, loaded=loaded)
    finally:
        provider.close()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "run.json", {"completed": True, "training_started": False,
        "baseline_evaluation": str(baseline_dir), "study": str(output / "study"),
        "data_config": str(Path(args.config).resolve()), "summary": rows})
    print("Completed: " + str(output / "study/summary.md"), flush=True)


if __name__ == "__main__":
    main()
