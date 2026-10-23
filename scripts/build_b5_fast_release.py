"""Build an explicit allow-list release of fast-only code and its pinned model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output = Path(output).resolve()
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Use a new release path")
    paths = ["README.md", "docs/b5_fast.md", "requirements-fast.txt", "fast_main.py",
             "configs/b5_fast.json", "models/b5_fast.pt", "models/b5_fast.json",
             "baselines/B5/__init__.py", "baselines/B5/model.py", "baselines/B5/conditions.py",
             "baselines/B5/config.py", "baselines/B5/objective.py",
             "baselines/B5/flow.py", "baselines/B5/metadata.py"]
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "baselines/B5_fast").glob("*.py"))]
    if (ROOT / "baselines/__init__.py").is_file():
        paths.append("baselines/__init__.py")
    files = {}
    for name in paths:
        p = ROOT / name
        if not p.is_file():
            raise FileNotFoundError(f"Missing release input: {name}")
        files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    package_sidecar = json.loads((ROOT / "models/b5_fast.json").read_text(encoding="utf-8"))
    if files["models/b5_fast.pt"] != package_sidecar["package_sha256"]:
        raise ValueError("Model file differs from its verified package manifest")
    manifest = {"kind": "B5-fast-standalone", "files": files, "slow_model_included": False,
                "training_data_included": False, "fm_state_sha256": package_sidecar["fm_state_sha256"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in paths:
            archive.write(ROOT / name, "B5-fast/" + name)
        archive.writestr("B5-fast/release_manifest.json", json.dumps(manifest, indent=2))
    manifest["zip_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="release/B5-fast.zip")
    args = p.parse_args()
    value = build(args.output)
    print(json.dumps({"release": str(Path(args.output).resolve()), "sha256": value["zip_sha256"]}))
