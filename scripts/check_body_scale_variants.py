"""Read-only body-scale A/B joint-anchor check."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.check_body_scale_record_ablation import main
if __name__ == "__main__": main()
