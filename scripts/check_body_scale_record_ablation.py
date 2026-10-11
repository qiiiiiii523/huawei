"""Synthetic + real new A/B checks; no training or production writes."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg12gen.body_scale_ablation import sha256_file, MANIFEST
from ecg12gen.body_scale_detrend import B_VARIANT, detrend_body_scale_record
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.record_context import window_context_record, prepare_record_context_inference, collate_record_context
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from ecg12gen.evaluate import evaluate_record_predictions


def synthetic(preprocessor):
    t = np.arange(65000) / 500
    raw = np.tile(1000 + 300 * np.sin(2 * np.pi * .03 * t) +
                  150 * np.sin(2 * np.pi * 3 * t), (6, 1)).astype(np.float32)
    output, valid, audit = detrend_body_scale_record(raw)
    core = slice(10000, -10000)
    low, high = np.sin(2 * np.pi * .03 * t[core]), np.sin(2 * np.pi * 3 * t[core])
    assert abs(np.mean(output[0, core] * low)) < abs(np.mean(raw[0, core] * low)) * .2
    assert np.isclose(np.mean(output[0, core] * high), np.mean(raw[0, core] * high), rtol=.1)
    assert audit["valid_runs"] == 6 and not audit["short_runs_retained"]
    gapmask = np.ones(raw.shape, bool)
    gapmask[:, 31000:32000] = False
    corrupt = raw.copy()
    corrupt[:, 31000:32000] = 1e9
    gapped, gap_valid, _ = detrend_body_scale_record(corrupt, gapmask)
    assert np.array_equal(gap_valid, gapmask)
    assert np.allclose(gapped[:, :31000], detrend_body_scale_record(raw[:, :31000])[0])
    assert np.allclose(gapped[:, 32000:], detrend_body_scale_record(raw[:, 32000:])[0])
    assert np.isnan(gapped[:, 31000:32000]).all()
    tiny, _, stats = detrend_body_scale_record(np.ones((6, 2), np.float32))
    assert np.array_equal(tiny, np.ones((6, 2))) and stats["short_runs_retained"] == 6
    inference = prepare_record_context_inference(np.ones((1, 60000), np.float32), raw,
                                                 "body_scale_d6", preprocessor, body_scale_variant=B_VARIANT)
    prepared = window_context_record(output, "body_scale_d6", valid)
    expected = (prepared.raw_uV - prepared.baseline_uV[None, :, None]) / 500
    expected[~prepared.time_mask] = 0
    assert np.allclose(inference["context"], expected)
    assert "target" not in inference and inference["body_scale_variant"] == B_VARIANT
    baseline_machine = prepare_record_context_inference(np.ones((1, 60000), np.float32), raw,
                                                        "ecg_machine_d6", preprocessor)
    experiment_machine = prepare_record_context_inference(np.ones((1, 60000), np.float32), raw,
                                                          "ecg_machine_d6", preprocessor,
                                                          body_scale_variant=B_VARIANT)
    assert np.array_equal(baseline_machine["context"], experiment_machine["context"])
    absent = prepare_record_context_inference(np.ones((1, 60000), np.float32), None,
                                              "body_scale_d6", preprocessor, body_scale_variant=B_VARIANT)
    assert not absent["context_available"]
    print("PASS synthetic: drift attenuation, frequency retention, gap isolation, tiny runs and inference parity", flush=True)


def main():
    config = ECGDataConfig.from_yaml(ROOT / "configs/common.yaml")
    pcfg = PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml")
    scales = {source: np.full(c, 500, np.float32) for source, c in pcfg.expected_leads.items()}
    preprocessor = ECGPreprocessor(pcfg, scales)  # audit-only, not fitted model parameters
    synthetic(preprocessor)
    directory = config.path("task2_body_scale_ablation")
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    for name, digest in manifest["overlay_files_sha256"].items():
        assert sha256_file(directory / name) == digest
    for item in manifest["canonical_files_unchanged"].values():
        assert sha256_file(item["path"]) == item["sha256"]
    for record in manifest["records"]:
        assert sha256_file(record["raw_source"]) == record["raw_sha256"]
    for split in ("train", "validation"):
        a = JointAnchorDataset(config, "task2", split)
        b = JointAnchorDataset(config, "task2", split, body_scale_variant=B_VARIANT)
        assert a._indices == b._indices and a._rows == b._rows
        assert a._targets.filename == b._targets.filename
        seen, changed = set(), 0
        try:
            for index in range(len(a)):
                row = a._rows[a._indices[index]]
                key = (row["input_type"], row["input_record_id"])
                if key in seen:
                    continue
                seen.add(key)
                sa, sb = a[index], b[index]
                for field in ("Y_12lead", "anchor_i_ecg", "target_quality_mask", "input_quality_mask"):
                    assert np.array_equal(getattr(sa, field), getattr(sb, field))
                ca, cb = sa.context, sb.context
                assert np.array_equal(ca.time_mask, cb.time_mask)
                assert np.array_equal(ca.valid_lengths, cb.valid_lengths)
                if key[0] == "ecg_machine_d6":
                    assert np.array_equal(ca.raw_uV, cb.raw_uV)
                    assert np.array_equal(ca.baseline_uV, cb.baseline_uV)
                else:
                    assert not np.array_equal(ca.raw_uV, cb.raw_uV)
                    changed += 1
                    for c in range(6):
                        observed = cb.raw_uV[:, c][cb.time_mask[:, c]]
                        assert np.isclose(np.median(observed), cb.baseline_uV[c], rtol=1e-6, atol=1e-3)
                    batch = collate_record_context([sb], preprocessor)
                    assert np.all(batch["context"][~batch["context_time_mask"]] == 0)
                    assert sb.meta["body_filter_protocol"] == manifest["filter"]["protocol"]
            if split == "validation":
                score, _ = evaluate_record_predictions(b._targets, b._targets, "task2", b._rows)
                assert score["r_missing11"] == 1 and score["task2_rmse_bonus_score"] == 10
            print(f"PASS real A/B {split}: {len(a)} eligible target windows, {changed} changed body contexts; machine/target/anchor/QC/masks/score unchanged", flush=True)
        finally:
            a.close()
            b.close()
    # Use a temporary bad manifest to ensure an incompatible/legacy B fails closed,
    # without changing real production metadata or arrays.
    import tempfile
    from ecg12gen.body_scale_ablation import BodyScaleBContextStore
    from ecg12gen.contracts import ContractError
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        (path / MANIFEST).write_text(json.dumps({"protocol": "legacy-window-B"}), encoding="utf-8")
        cfg = ECGDataConfig({**config.raw, "paths": {**config.raw["paths"],
                            "task2_body_scale_ablation": str(path)}}, config.repository_root)
        try:
            BodyScaleBContextStore(cfg, "train")
        except ContractError:
            pass
        else:
            raise AssertionError("Legacy B accepted")
    print("PASS B compatibility guard", flush=True)


if __name__ == "__main__":
    main()
