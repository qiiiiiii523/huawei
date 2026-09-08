"""Read-only body-scale A/B joint-anchor check."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ecg12gen.body_scale import BodyScaleVariantDataset
from ecg12gen.dataset import ECGDataConfig
def main() -> None:
    cfg = ECGDataConfig.from_yaml(ROOT / "configs" / "common.yaml")
    a = BodyScaleVariantDataset(cfg, "train", "A_raw_window")
    b = BodyScaleVariantDataset(cfg, "train", "B_detrend_0p2Hz_then_window")
    five = BodyScaleVariantDataset(cfg, "validation", "A_raw_window", (1,2,3,4,5))
    assert a[0].context_ecg.shape == b[0].context_ecg.shape == (6,5000)
    assert five[0].context_ecg.shape == (5,5000) and five[0].anchor_lead_mask.sum() == 1
    print("PASS: body-scale A/B context variants keep target-time I-only anchor")
if __name__ == "__main__": main()
