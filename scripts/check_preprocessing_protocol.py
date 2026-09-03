"""Synthetic contract check for the frozen non-destructive preprocessing protocol."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.evaluate import evaluate_centered_diagnostic, evaluate_predictions
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def _signals(n: int, channels: int, baseline: float, amplitude: float) -> np.ndarray:
    time = np.linspace(0, 4 * np.pi, 5000, dtype=np.float32)
    wave = amplitude * np.sin(time)[None, None, :]
    lead_bias = np.arange(channels, dtype=np.float32)[None, :, None]
    return baseline + lead_bias + np.broadcast_to(wave, (n, channels, 5000)).copy()


def main() -> None:
    config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    assert config.baseline_method == "per_window_per_lead_median"
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
    morphology_uV = preprocessor.d12_model_view_to_morphology_uV(d12.model_signal)
    restored_for_training_audit = preprocessor.compose_raw_d12_prediction(d12.model_signal, d12.baseline_uV)
    assert d12.source_type == "d12" and d12.model_signal.shape == (12, 5000)
    assert np.allclose(restored_for_training_audit, train["d12"][0], atol=1e-4)
    assert np.allclose(morphology_uV + d12.baseline_uV[:, None], restored_for_training_audit, atol=1e-4)

    # A different baseline per validation window is invisible to centered RMSE,
    # but must remain visible to official raw-μV RMSE.
    target = _signals(3, 12, 0, 250) + np.asarray([0, 400, -600], dtype=np.float32)[:, None, None]
    prediction_without_baseline_head = _signals(3, 12, 0, 250)
    raw_overall, _ = evaluate_predictions(prediction_without_baseline_head, target, "task1")
    centered_overall, _ = evaluate_centered_diagnostic(prediction_without_baseline_head, target, "task1")
    assert raw_overall["twelve_lead_mean_rmse_uV"] > 100.0
    assert centered_overall["twelve_lead_mean_rmse_uV"] < 1e-4
    assert centered_overall["evaluation_view"] == "centered_diagnostic_not_official"
    print("PASS: frozen train-only preprocessing; raw data unchanged; baseline restoration requires a supplied prediction; raw/centered diagnostics differ as expected")


if __name__ == "__main__":
    main()