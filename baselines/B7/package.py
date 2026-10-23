"""Export an independent B7 inference checkpoint without training data/state."""
from __future__ import annotations

import argparse
from pathlib import Path
from .checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint, file_sha256


def export(source, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new package path")
    value = load_checkpoint(source)
    model_from_checkpoint(value)  # Strictly validate the complete B7 state before export.
    keep = ("architecture_id", "format_version", "generated_leads", "model_config", "loss", "sampling",
            "scales_uV", "stage", "ema", "initialization", "update", "best_r")
    payload = {k: value[k] for k in keep}
    payload["source_sha256"] = file_sha256(source)
    payload["target_used_at_inference"] = False
    save_checkpoint(destination, payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    export(a.checkpoint, a.output)


if __name__ == "__main__":
    main()
