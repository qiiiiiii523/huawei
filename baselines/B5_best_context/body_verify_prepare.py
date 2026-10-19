"""Resolve reported E24 paths and stage immutable optimizer-preserving extensions."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

from baselines.B5_best.common import read, write, sha
from .body_features import BODY_PROTOCOL, LEGACY_ENCODING, SHAPE_ENCODING, BEAT16_ENCODING, feature_encoding
from .body_encoding_prepare import check_cached_pair


def completed_run(folder, cache_sha, encoding, *, seed, epochs, best_epoch=None):
    root = Path(folder).resolve()
    cfg, done = read(root / "config.json"), read(root / "completed.json")
    if (cfg.get("protocol") != BODY_PROTOCOL or cfg.get("group") != "context"
            or cfg.get("record_gate") is not None or feature_encoding(cfg) != encoding
            or cfg.get("seed") != seed or cfg.get("epochs") != epochs or cfg.get("cache_sha256") != cache_sha
            or done.get("completed") is not True or done.get("epochs") != epochs
            or done.get("training_seed") != seed or (best_epoch is not None and done.get("best_epoch") != best_epoch)
            or sha(root / "best.pt") != done.get("best_checkpoint_sha256")):
        raise ValueError("Completed run/cache/encoding/seed/budget/checkpoint mismatch: " + str(root))
    return cfg, done


def find_e24(root, cache_sha):
    candidates = []
    for file in Path(root).rglob("completed.json"):
        try:
            _, done = completed_run(file.parent, cache_sha, SHAPE_ENCODING, seed=42, epochs=30, best_epoch=24)
            if abs(done["best_II_r"] - .786604) <= 5e-7:
                candidates.append(file.parent.resolve())
        except (ValueError, KeyError, OSError):
            continue
    if len(candidates) != 1:
        raise ValueError("Cannot uniquely locate reported E-best24 (30 epochs, seed42, II r≈0.786604). "
                         "Supply --e-run explicitly. Matches: " + str([str(p) for p in candidates]))
    return candidates[0]


def paths(args):
    repository = Path(args.repository_root).resolve()
    results = repository / "results/B5-best-context"
    cache = results / "body_v1/cache"
    encoded = results / "body_encoding_three_v1/encoded_cache"
    clip16 = results / "body_clip16_v1/clip16_cache"
    old, new = check_cached_pair(cache, encoded)
    check_cached_pair(cache, clip16, expected_encoding=BEAT16_ENCODING)
    e = Path(args.e_run).resolve() if args.e_run else find_e24(results, sha(encoded / "cache.json"))
    ecfg, edone = completed_run(e, sha(encoded / "cache.json"), SHAPE_ENCODING, seed=42, epochs=30, best_epoch=24)
    o = Path(args.o30_run).resolve() if args.o30_run else e.parent / "O"
    ocfg, _ = completed_run(o, sha(cache / "cache.json"), LEGACY_ENCODING, seed=42, epochs=30)
    from .body_encoding_study import check_training_pair
    check_training_pair(ocfg, ecfg, sha(cache / "cache.json"), sha(encoded / "cache.json"))
    source = results / "body_clip16_v1/O16"
    completed_run(source, sha(clip16 / "cache.json"), BEAT16_ENCODING, seed=42, epochs=10)
    config = repository / "results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
    if not config.is_file():
        raise FileNotFoundError(config)
    value = {"cache": str(cache), "encoded_cache": str(encoded), "clip16_cache": str(clip16),
        "e_run": str(e), "o30_run": str(o), "o16_source": str(source), "config": str(config),
        "E_best_epoch": edone["best_epoch"], "E_II_r": edone["best_II_r"],
        "E_best_sha256": sha(e / "best.pt"), "O30_best_sha256": sha(o / "best.pt"),
        "source_cache_sha256": sha(cache / "cache.json"), "encoded_cache_sha256": sha(encoded / "cache.json"),
        "clip16_cache_sha256": sha(clip16 / "cache.json"), "config_sha256": sha(config)}
    output = Path(args.output)
    if output.exists() and read(output) != value:
        raise ValueError("Resolved sources changed; use a new verification directory")
    write(output, value)
    values = [value[k] for k in ("cache", "encoded_cache", "clip16_cache", "e_run", "o30_run", "o16_source", "config")]
    if any("\t" in s or "\n" in s for s in values):
        raise ValueError("Paths must not contain tabs/newlines")
    output.with_suffix(".tsv").write_text("\t".join(values) + "\n", encoding="utf-8")
    print({"ready": True, "E_checkpoint": str(e / "best.pt"), "O30": str(o), "optimizer_updates": 0}, flush=True)


def extend(args):
    source, output, cache = Path(args.source_run).resolve(), Path(args.output).resolve(), Path(args.cache).resolve()
    cfg = read(source / "config.json")
    completed_run(source, sha(cache / "cache.json"), BEAT16_ENCODING, seed=42, epochs=10)
    if args.epochs <= cfg["epochs"]:
        raise ValueError("Extension total epochs must exceed source budget")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Extension output must be separate from the original run")
    validation = sorted((source / "validation").glob("epoch_*.json"))
    if not validation or not (source / "last.pt").is_file():
        raise ValueError("Preserving training state needs original last.pt and validation history")
    provenance = {"source_run": str(source), "source_epochs": cfg["epochs"], "total_epochs": args.epochs,
        "source_config_sha256": sha(source / "config.json"), "source_best_sha256": sha(source / "best.pt"),
        "source_last_sha256": sha(source / "last.pt"), "cache_sha256": sha(cache / "cache.json"),
        "validation_sha256": {p.name: sha(p) for p in validation},
        "optimizer_and_RNG_preserved": True, "starts_from": "last.pt; retains historical best.pt"}
    if output.exists():
        if not args.resume or read(output / "extension.json") != provenance:
            raise ValueError("Use a new extension output or --resume with unchanged sources")
        if read(output / "config.json") != {**cfg, "epochs": args.epochs}:
            raise ValueError("Extension training config changed")
        print({"ready": True, "staged": str(output), "reused": True}, flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=output.name + "_staging_", dir=output.parent))
    for name in ("last.pt", "best.pt"):
        shutil.copy2(source / name, staged / name)
    for file in validation:
        dest = staged / "validation" / file.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(file, dest)
    write(staged / "config.json", {**cfg, "epochs": args.epochs})
    write(staged / "extension.json", provenance)
    staged.replace(output)
    print({"ready": True, "staged": str(output), "total_epochs": args.epochs,
           "optimizer_updates": 0, "source_unchanged": True}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    a = commands.add_parser("paths")
    a.add_argument("--repository-root", default="/home/qht/huawei")
    a.add_argument("--e-run")
    a.add_argument("--o30-run")
    a.add_argument("--output", required=True)
    a = commands.add_parser("extend")
    a.add_argument("--source-run", required=True)
    a.add_argument("--cache", required=True)
    a.add_argument("--epochs", type=int, default=30)
    a.add_argument("--output", required=True)
    a.add_argument("--resume", action="store_true")
    args = p.parse_args()
    (paths if args.command == "paths" else extend)(args)


if __name__ == "__main__":
    main()
