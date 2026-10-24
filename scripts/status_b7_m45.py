"""Print M4/M5 controller status and recent logs from the correct output root."""
import argparse
from collections import deque
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/B7-M4-M5"))
    args = p.parse_args()
    root = args.output.resolve()
    status = root / "pipeline_status.json"
    if not status.is_file():
        print(f"No controller status yet: {status}")
        return
    report = json.loads(status.read_text(encoding="utf-8"))
    print(f"Output: {root}\nAll stages successful: {report['success']}")
    for name, state in report["jobs"].items():
        result = report.get("training_results", {}).get(name, {})
        print(f"\n{name}: {state}" + (f"  r={result['best_r']:.6f}" if "best_r" in result else ""))
        log = root / "logs" / (name + ".log")
        if log.is_file():
            with log.open(encoding="utf-8", errors="replace") as stream:
                for line in deque(stream, maxlen=4):
                    print(line.rstrip())


if __name__ == "__main__":
    main()
