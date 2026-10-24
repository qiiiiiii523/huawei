"""Independent ~20M B7-T candidates; reuse waveforms, never baseline weights."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import LossConfig, ModelConfig, SamplingConfig, read_json
from .train import validate_settings

ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    "g6_19m": {"parameters": 19398702, "model": ModelConfig(depths=(2, 2, 6))},
    "w144_g5_21m": {"parameters": 21259214, "model": ModelConfig(
        embed_dims=(144, 288, 432), heads=(4, 8, 12), depths=(2, 2, 5))},
}
STAGES = ("public10", "public40", "huawei40")


def config_directory(preset):
    if preset not in PRESETS:
        raise ValueError("Unknown compact preset")
    return ROOT / "configs" / "b7_t_compact" / preset


def load_configs(preset):
    """Reject accidental changes to the agreed recipe before launching jobs."""
    configs = {}
    for stage in STAGES:
        cfg = read_json(config_directory(preset) / f"b7_{stage}.json")
        model, loss, sampling = validate_settings(cfg)
        short = stage == "public10"
        expected_training = {"seed": 42, "updates": 10000 if short else 1000,
            "validate_every": 10000 if short else (1000 if stage == "public40" else 500),
            "checkpoint_every": 500, "batch_size": 64 if short else 32,
            "learning_rate": .0001 if short else .00001, "weight_decay": .01,
            "ema_decay": .999, "gradient_clip": 1., "boundary_probability": 0. if short else .15}
        if (cfg["stage"] != stage or model != PRESETS[preset]["model"] or loss != LossConfig()
                or sampling != SamplingConfig(window_seconds=10 if short else 40)
                or cfg["training"] != expected_training
                or cfg["validation"] != {"batch_size": 32, "initial": not short}):
            raise ValueError(f"Unexpected {preset}/{stage} recipe; use a separate experiment for other settings")
        configs[stage] = cfg
    return configs


def main():
    from .pipeline import run
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--ptb-root", type=Path, required=True)
    p.add_argument("--reuse-data-from", type=Path, help="Old run with complete data/public10, ptb40, huawei caches")
    p.add_argument("--output", type=Path)
    p.add_argument("--gpu-p", default="0")
    p.add_argument("--gpu-a", default="0")
    p.add_argument("--gpu-b", default="0")
    p.add_argument("--max-train", type=int, choices=(1, 2), default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    load_configs(args.preset)
    args.config_dir = config_directory(args.preset)
    args.output = args.output or ROOT / "results/B7-T-compact" / args.preset / "seed42"
    args.retry_preparation = False
    if args.reuse_data_from and args.output.resolve() == args.reuse_data_from.resolve():
        raise ValueError("New experiment output must differ from the reused cache run")
    print(f"[compact] preset={args.preset} parameters={PRESETS[args.preset]['parameters']} "
          f"P=64x10000 final-validation-only Q/A/B=32x1000 max_train={args.max_train}", flush=True)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
