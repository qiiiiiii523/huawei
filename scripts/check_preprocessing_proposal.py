"""Synthetic contract check for the non-destructive preprocessing proposal."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def _signals(n: int, channels: int, baseline: float, amplitude: float) -> np.ndarray:
    time = np.linspace(0, 4 * np.pi, 5000, dtype=np.float32)
    wave = amplitude * np.sin(time)[None, None, :]
    lead_bias = np.arange(channels, dtype=np.float32)[None, :, None]
    return baseline + lead_bias + np.broadcast_to(wave, (n, channels, 5000)).copy()


def main() -> None:
    config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    train = {
        "watch_ecg": _signals(3, 1, 3, 200),
        "ecg_machine_d6": _signals(3, 6, 300, 200),
        "body_scale_d6": _signals(3, 6, 80000, 800),
        "d12": _signals(3, 12, 900, 250),
    }
    original = train["body_scale_d6"].copy()
    preprocessor = ECGPreprocessor.fit(config, train)
    body = preprocessor.transform_window(train["body_scale_d6"][0], "body_scale_d6")
    assert np.array_equal(train["body_scale_d6"], original), "raw input was mutated"
    assert np.allclose(np.median(body.model_signal, axis=1), 0.0, atol=1e-6)
    assert np.max(np.abs(body.model_signal)) < config.clip_model_signal
    d12 = preprocessor.transform_d12_target(train["d12"][0])
    assert d12.source_type == "d12" and d12.model_signal.shape == (12, 5000)
    print("PASS: train-only scale fit, non-destructive centering, canonical d12 target transform")


if __name__ == "__main__":
    main()
