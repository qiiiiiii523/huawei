"""Pinned assets and settings; paths may move, selected weight bytes may not."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from baselines.B5_best.common import read, sha, write
from . import PROTOCOL

DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "configs/b5_final.json"
BASE_SHA = "990f5d95312dba0ab3e4bde416ee15d3b464b902fc63601aab1f852d9881f6f8"
BODY_SHA = "ed8de0495b9d89a59f7b16b40ae6117b56af5c37ad7add11dabb08069b4a0f8e"
SOURCE_BODY_SHA = "8f850dad07ec88f63ef1e400ec97dc655174f62e09bd5eadc6c22c53f8a1bef3"
BODY_CACHE_SHA = "826f5cc3ba98772b9b71b83b427c213e561b1059600fc1acc692029cba70dde9"
SAMPLING = {"seed": 42, "solver": "heun", "steps": 16, "samples": 16}


def load_profile(path=DEFAULT_PROFILE):
    value = read(path)
    expected = {"protocol": PROTOCOL, "sampling": SAMPLING, "slow_beta": .75,
                "body_alpha": 1., "machine_mode": "fir25", "fir_strength": .25}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("B5-final requires the selected Heun16/K16/seed42/beta.75/alpha1/FIR25 composition")
    if (value.get("base", {}).get("sha256") != BASE_SHA
            or value.get("body", {}).get("sha256") != BODY_SHA
            or value["body"].get("training_seed") != 43
            or value["body"].get("additional_epoch") != 3
            or value["body"].get("feature_encoding") != "legacy_beat16_stats8_v1"
            or value["body"].get("history_variant") != "B_detrend_0p2Hz_then_window"):
        raise ValueError("B5-final pins newFM-best45+P and adapted body seed43-best3")
    return value


@dataclass(frozen=True)
class Assets:
    profile: dict
    base: Path
    body: Path
    validation_config: Path
    baseline_evaluation: Path

    def report(self):
        return {"protocol": PROTOCOL, "base_checkpoint": str(self.base), "base_sha256": BASE_SHA,
                "body_checkpoint": str(self.body), "body_sha256": BODY_SHA,
                "validation_config": str(self.validation_config),
                "baseline_evaluation": str(self.baseline_evaluation),
                "sampling": SAMPLING, "slow_beta": .75, "body_alpha": 1.,
                "machine_mode": "fir25", "fir_strength": .25,
                "training_started": False, "target_used_at_inference": False}


def resolve_assets(repository_root="/home/qht/huawei", profile_path=DEFAULT_PROFILE,
                   checkpoint=None, body_checkpoint=None):
    profile = load_profile(profile_path)
    root = Path(repository_root).resolve()
    def resolve(path):
        path = Path(path)
        return (path if path.is_absolute() else root / path).resolve()
    base = resolve(checkpoint or profile["base"]["path"])
    body = resolve(body_checkpoint or profile["body"]["path"])
    for name, path, digest in (("FM+P", base, BASE_SHA), ("body seed43-best3", body, BODY_SHA)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected {name} weight: {path}; keep the server results or copy identical weights")
        if sha(path) != digest:
            raise ValueError(f"Wrong {name} weight SHA256: {path}")
    return Assets(profile, base, body, resolve(profile["validation_config"]), resolve(profile["baseline_evaluation"]))


def check_payloads(bundle, body):
    """Also enforce the selected training lineage when weights are loaded."""
    from baselines.B5_best_context.body_features import BODY_PROTOCOL, BEAT16_ENCODING, feature_encoding
    from baselines.B5_best_context.body_history import B_POLICY, history_policy
    if bundle.get("sampling") != SAMPLING or bundle.get("slow_beta") != .75:
        raise ValueError("Selected complete bundle sampling/beta mismatch")
    lineage = body.get("body_adaptation", {})
    if (body.get("protocol") != BODY_PROTOCOL or body.get("group") != "context"
            or body.get("record_gate") is not None or body.get("training_seed") != 43
            or body.get("epoch") != 3 or body.get("base_checkpoint_sha256") != BASE_SHA
            or body.get("cache_sha256") != BODY_CACHE_SHA or body.get("sampling") != SAMPLING
            or body.get("slow_beta") != .75 or feature_encoding(body) != BEAT16_ENCODING
            or history_policy(body) != B_POLICY
            or lineage.get("protocol") != "B5-body-new-base-adaptation-v1"
            or lineage.get("source_checkpoint_sha256") != SOURCE_BODY_SHA
            or lineage.get("target_base_sha256") != BASE_SHA):
        raise ValueError("Selected body adaptation lineage/settings mismatch")


def add_asset_arguments(parser):
    parser.add_argument("--repository-root", default="/home/qht/huawei")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--checkpoint", help="Optional identical complete weight at another path; SHA remains pinned")
    parser.add_argument("--body-checkpoint", help="Optional identical adapted body weight at another path")


def assets_from_args(args):
    return resolve_assets(args.repository_root, args.profile, args.checkpoint, args.body_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Check the final selected weight files without training or GPU use")
    add_asset_arguments(parser)
    parser.add_argument("--report")
    args = parser.parse_args()
    assets = assets_from_args(args)
    report = dict(assets.report(), ready=True, check="weight_file_SHA256_only", model_forward_checked=False)
    if args.report:
        if Path(args.report).exists():
            raise FileExistsError("Keep the existing preflight report; use a new path")
        write(args.report, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
