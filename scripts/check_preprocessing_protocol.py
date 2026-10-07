"""Synthetic contract check for the frozen non-destructive preprocessing protocol."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.evaluate import evaluate_record_predictions
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig, PreprocessingError


def _signals(n: int, channels: int, baseline: float, amplitude: float) -> np.ndarray:
    time = np.linspace(0, 4 * np.pi, 5000, dtype=np.float32)
    wave = amplitude * np.sin(time)[None, None, :]
    lead_bias = np.arange(channels, dtype=np.float32)[None, :, None]
    return baseline + lead_bias + np.broadcast_to(wave, (n, channels, 5000)).copy()


def main() -> None:
    config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    assert config.baseline_method == "context_only_per_record_per_lead_median"
    train = {
        "watch_ecg": _signals(3, 1, 3, 200),
        "ecg_machine_d6": _signals(3, 6, 300, 200),
        "body_scale_d6": _signals(3, 6, 80000, 800),
        "d12": _signals(3, 12, 900, 250),
    }
    original = train["body_scale_d6"].copy()
    preprocessor = ECGPreprocessor.fit(config, train)
    assert np.allclose(preprocessor.scale_uV_by_source["ecg_machine_i"], preprocessor.scale_uV_by_source["d12"][:1])
    body_record_baseline = np.median(train["body_scale_d6"], axis=(0, 2))
    body = preprocessor.transform_window(train["body_scale_d6"][0], "body_scale_d6", body_record_baseline)
    assert np.array_equal(train["body_scale_d6"], original), "raw input was mutated"
    assert config.clip_model_signal is None
    observed = preprocessor.transform_observed_record(
        train["body_scale_d6"].transpose(1, 0, 2).reshape(6, -1), "body_scale_d6")
    assert np.allclose(np.median(observed.model_signal, axis=1), 0.0, atol=1e-6)

    d12_record_baseline = np.median(train["d12"], axis=(0, 2))
    d12 = preprocessor.transform_d12_target(train["d12"][0], d12_record_baseline)
    restored_for_training_audit = preprocessor.compose_raw_d12_prediction(d12.model_signal)
    assert d12.source_type == "d12" and d12.model_signal.shape == (12, 5000)
    assert np.allclose(restored_for_training_audit, train["d12"][0], atol=1e-4)
    assert np.array_equal(d12.baseline_uV, np.zeros(12))
    anchor = preprocessor.transform_observed_record(train["d12"][0, :1], "ecg_machine_i")
    assert np.array_equal(anchor.model_signal, d12.model_signal[:1])
    batch, offsets, _ = preprocessor.transform_batch(train["d12"], "d12")
    assert np.all(offsets == 0) and np.allclose(batch[0], d12.model_signal)
    large = train["d12"][0] * 100
    large_model = preprocessor.transform_d12_target(large).model_signal
    assert np.max(np.abs(large_model)) > 12  # no clipping of unseen test amplitudes
    assert np.allclose(preprocessor.d12_model_view_to_raw_uV(large_model), large, rtol=1e-6)
    with tempfile.TemporaryDirectory() as temporary:
        artifact = Path(temporary) / "scales.npz"
        preprocessor.save(artifact)
        loaded = ECGPreprocessor.load(config, artifact)
        assert np.array_equal(loaded.transform_d12_target(large).model_signal, large_model)
    try:
        preprocessor.compose_raw_d12_prediction(d12.model_signal, d12_record_baseline)
        raise AssertionError("adding a target median was accepted")
    except PreprocessingError:
        pass
    # Per-window and complete-record context paths use the same physical median.
    record = train["body_scale_d6"].transpose(1, 0, 2).reshape(6, -1)
    record[:, 5000:] += 300
    context = preprocessor.transform_observed_record(record, "body_scale_d6")
    pieces = [preprocessor.transform_window(record[:, start:start + 5000], "body_scale_d6", context.baseline_uV).model_signal
              for start in range(0, record.shape[1], 5000)]
    assert np.array_equal(np.concatenate(pieces, axis=1), context.model_signal)
    shifted = {source: values + 1000 for source, values in train.items()}
    shifted_preprocessor = ECGPreprocessor.fit(config, shifted)
    for source in preprocessor.scale_uV_by_source:
        assert np.allclose(preprocessor.scale_uV_by_source[source], shifted_preprocessor.scale_uV_by_source[source], rtol=1e-4)

    # Evaluation preserves record offsets in raw RMSE and never re-centers.
    target = _signals(3, 12, 0, 250) + np.asarray([0, 400, -600], dtype=np.float32)[:, None, None]
    prediction_without_baseline_head = _signals(3, 12, 0, 250)
    metadata = [{"pair_id": f"record_{index}", "target_record_id": f"target_{index}", "start_sample_500hz": "0"}
                for index in range(len(target))]
    raw_overall, _ = evaluate_record_predictions(
        prediction_without_baseline_head, target, "task1", metadata_rows=metadata)
    assert raw_overall["missing11_mean_rmse_uV"] > 100.0
    assert np.isclose(raw_overall["r_missing11"], 1.0)
    assert raw_overall["evaluation_view"] == "raw_uV"
    model_target, _, _ = preprocessor.transform_batch(target, "d12")
    restored = np.stack([preprocessor.d12_model_view_to_raw_uV(window) for window in model_target])
    identity, _ = evaluate_record_predictions(restored, target, "task1", metadata)
    assert np.isclose(identity["r_missing11"], 1)
    assert identity["missing11_mean_rmse_uV"] < 1e-3
    print("PASS: raw anchor/target reversible scaling; record-centered context; frozen scale save/load; evaluation preserves raw offsets")


if __name__ == "__main__":
    main()
