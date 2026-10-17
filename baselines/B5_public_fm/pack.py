"""Small reports only: no checkpoints, raw ECGs or generated waveform arrays."""
import argparse
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root, output = Path(args.run).resolve(), Path(args.output).resolve()
    if not (root / "shared/study.json").is_file():
        raise FileNotFoundError("Run has no shared/study.json")
    if output.exists():
        raise FileExistsError("Report archive exists; rename it or choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix in (".json", ".csv", ".log", ".md")
             and path.relative_to(root).as_posix() != "shared/public/manifest.json" and not path.name.endswith(".tmp")]
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False)
    print(f"Reports: {output}\nFiles: {len(files)}; bytes: {output.stat().st_size}")


if __name__ == "__main__":
    main()
