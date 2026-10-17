"""Synthetic CPU checks; no real ECG training and no FM generation."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5.metadata import encode_demographics
from baselines.B5_best.common import write, read, sha, state_sha
from baselines.B5_best.tests.test_hybrid import fixture
from baselines.B5_best.bundle import export_bundle, load_bundle
from baselines.B5_best.slow import predict_native
from baselines.B5_best.combine import upsample_native
from baselines.B5_best import SAMPLING
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_public_slow import core, ptb, prepare, train
from baselines.B5_public_slow.common import (SourceSampler, validate_record, load_study,
                                            code_digest, safe_file)
from baselines.B5_public_slow.export import export
from baselines.B5_public_slow.compare import compare
from baselines.B5_public_slow import PROTOCOL

torch.set_num_threads(2)


def synthetic_record(n=6000):
    t = np.arange(n * 10, dtype=float)
    raw = np.tile((100 * np.sin(t / 50) + 70 * np.sin(t / 7000))[None], (12, 1)).astype(np.float32)
    scales = np.ones(12, np.float32) * 100
    record = ptb.pack_record(raw, np.ones(12, bool), encode_demographics({"age": 40}), scales)
    if n == 6000:
        record.update(target_uV=raw, fm_uV=raw + np.cos(t / 70).astype(np.float32)[None])
    return record


def study_fixture(root):
    fm, slow, info, profile = fixture(root)
    bundle_path = root / "initial.pt"
    bundle = export_bundle(fm, slow, info, bundle_path, profile)
    source = root / "source/shared"
    source.mkdir(parents=True)
    torch.save(bundle, source / "components.pt")
    write(source / "study.json", {"bundle_sha256": sha(source / "components.pt"),
          "residual_weight": 2.5890171538829843, "source_checks": {}})
    cache, public = root / "source/cache", root / "public"
    cache.mkdir(); public.mkdir()
    record = synthetic_record()
    def entry(where, name, subject, key):
        np.savez(where / (name + ".npz"), **record)
        return {"file": name + ".npz", "sha256": sha(where / (name + ".npz")),
                "subject_id": subject, "target_record_id": key}
    training = entry(cache, "train", "Htrain", "train")
    validation = entry(cache, "validation", "Hval", "val")
    vals, rows = [], {}
    for task in ("task1", "task2"):
        rows[task] = []
        devices = ["ecg_machine_d6"] if task == "task1" else ["ecg_machine_d6", "body_scale_d6"]
        for number, device in enumerate(devices):
            pair = task + str(number)
            vals.append(dict(validation, cache_key=task + ":" + pair, pair_id=pair))
            rows[task] += [{"pair_id": pair, "target_record_id": "val", "subject_id": "Hval",
                "input_type": device, "start_sample_500hz": str(s), "end_sample_500hz": str(s + 5000)}
                for s in range(0, 60000, 5000)]
    write(cache / "cache.json", {"protocol": "B5-best-SlowPCC-fixed-FM-records-v2-pair-validation",
          "completed": True, "sampling": dict(SAMPLING), "study_sha256": sha(source / "study.json"),
          "train": [training], "validation": vals, "validation_rows": rows})
    np.savez(public / "p.npz", **synthetic_record(5500))
    np.savez(public / "v.npz", **synthetic_record(2500))
    write(public / "public.json", {"protocol": PROTOCOL, "completed": True,
          "bundle_scales_sha256": bundle["fm"]["scales_sha256"],
          "train": [{"file": "p.npz", "sha256": sha(public / "p.npz"), "subject_id": "PTB:p",
                     "target_record_id": "PTB:p:record", "raw_wave_sha256": "public_train"}],
          "validation": [{"file": "v.npz", "sha256": sha(public / "v.npz"), "subject_id": "PTB:v",
                          "target_record_id": "PTB:v:record", "raw_wave_sha256": "public_validation"}]})
    profile_path = root / "profile.json"
    write(profile_path, {"components": {"fm_sha256": bundle["provenance"]["fm_sha256"]}})
    with patch.object(prepare, "DEFAULT_PROFILE", profile_path):
        prepare.prepare(bundle_path, cache, public, root / "shared", u1=1, u2=1, validate_every=1)
    return root / "shared/study.json", bundle


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.8))

    def forward(self, anchor, **kwargs):
        return anchor.repeat(1, 11, 1) * self.scale


class PublicSlowTests(unittest.TestCase):
    def test_s50_variable_real_lengths_and_target_free_input(self):
        fields = {k: torch.from_numpy(v[None]) for k, v in encode_demographics({}).items()}
        for n in (2500, 5500, 6000):
            anchor = torch.arange(n, dtype=torch.float32).reshape(1, 1, n)
            model = Tiny()
            output = core.generate(model, {"anchor": anchor, **fields})
            torch.testing.assert_close(output, anchor.repeat(1, 11, 1) * .8)
            output.mean().backward()
            self.assertNotEqual(float(model.scale.grad), 0)
        with self.assertRaises(ValueError):
            core.generate(Tiny(), {"anchor": torch.zeros(1, 1, 2400), **fields})

    def test_train_path_matches_real_deployed_s50_and_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, bundle = study_fixture(Path(tmp))
            self.assertEqual(read(path)["native_max_abs_difference"], 0.)
        native = torch.arange(66000, dtype=torch.float64).reshape(1, 11, 6000)
        np.testing.assert_allclose(core.upsample(native)[0].numpy(), upsample_native(native[0].numpy()), atol=1e-8)

    def test_public_patient_split_no_window_leakage_and_resume_sampler(self):
        subjects = [f"patient{i:03}" for i in range(50)]
        first = ptb.patient_split(subjects)
        self.assertEqual(first, ptb.patient_split(list(reversed(subjects))))
        self.assertEqual(sum(v == "validation" for v in first.values()), 10)
        entries = [{"file": str(i), "subject_id": str(i // 2)} for i in range(10)]
        for balanced in (False, True):
            sampler = SourceSampler(entries, 42, balanced)
            for _ in range(7): sampler.draw()
            state = copy.deepcopy(sampler.state())
            expected = [sampler.draw() for _ in range(15)]
            restored = SourceSampler(entries, 999, balanced)
            restored.restore(state)
            self.assertEqual(expected, [restored.draw() for _ in range(15)])

    def test_calibration_antialias_units_and_115_second_record(self):
        import wfdb
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t = np.arange(115200) / 1000
            signal = np.tile((.2 * np.sin(2 * np.pi * 5 * t) + .1 * np.sin(2 * np.pi * 300 * t))[:, None], (1, 12))
            wfdb.wrsamp("record", fs=1000, units=["mV"] * 12, sig_name=list(ptb.LEADS),
                        p_signal=signal, fmt=["16"] * 12, adc_gain=[2000.] * 12,
                        baseline=[0] * 12, comments=["age: 58", "sex: female"], write_dir=str(root))
            wave, mask, _, meta, _ = ptb.calibrated_record(root / "record")
            self.assertTrue(mask.all())
            self.assertEqual(int(meta["sex"]), 1)
            self.assertAlmostEqual(float(meta["numeric"][0]), .58, places=5)
            expected = 200 * np.sin(2 * np.pi * 5 * np.arange(wave.shape[-1]) / 500)
            self.assertLess(np.sqrt(np.mean((wave[0, 100:-100] - expected[100:-100]) ** 2)), 2.)
            packed = ptb.pack_record(wave, mask, meta, np.ones(12) * 100)
            self.assertEqual(packed["anchor"].shape[-1], 5500)  # 110 real seconds, not fake 120

    def test_masked_bad_target_does_not_change_loss_or_gradient(self):
        record = synthetic_record(2500)
        record["mask"][4] = False
        batch = core.batch_record(record, torch.device("cpu"))
        prediction = torch.randn(1, 11, 2500, requires_grad=True)
        scales = torch.ones(12) * 100
        loss, _ = core.objective(prediction, batch, scales, 2., 0.)
        changed = dict(batch); changed["slow_target"] = batch["slow_target"].clone()
        changed["slow_target"][:, 3] = 1e20
        other, _ = core.objective(prediction, changed, scales, 2., 0.)
        self.assertEqual(float(loss.detach()), float(other.detach()))
        loss.backward()
        self.assertEqual(float(prediction.grad[:, 3].abs().sum()), 0.)

    def test_stage_sources_and_public_never_needs_FM_target_cache(self):
        self.assertEqual(train.sources_for("M", 1), ("huawei", "public"))
        self.assertEqual(train.sources_for("P", 1), ("public", "public"))
        for group in ("C", "M", "P"):
            self.assertEqual(train.sources_for(group, 2), ("huawei", "huawei"))
        batch = core.batch_record(synthetic_record(2500), torch.device("cpu"))
        loss, _ = core.objective(torch.ones(1, 11, 2500, requires_grad=True), batch, torch.ones(12), 2., 0.)
        self.assertTrue(torch.isfinite(loss))

    def test_study_rejects_changed_cache_wrong_FM_and_patient_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = study_fixture(root)
            load_study(path, deep=True)
            with (root / "public/p.npz").open("ab") as handle: handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_study(path, deep=True)
            huawei = read(root / "source/cache/cache.json")
            public = read(root / "public/public.json")
            huawei["train"][0]["subject_id"] = "Hval"
            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare.check_populations(huawei, public)
        with self.assertRaises(ValueError): safe_file(Path("."), "../outside")

    def test_two_stage_synthetic_run_and_exact_resume_all_arms(self):
        # Exercise only a one-parameter toy network, never optimize actual S50/FM on CPU.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, bundle = study_fixture(root)
            def toy_factory(*args, **kwargs): return Tiny()
            def fake_eval(model, study, original, device, output, *args, **kwargs):
                Path(output).mkdir(parents=True)
                r = float(model.scale.detach()) / 2
                return {"task1_r_missing11": r, "task2_r_missing11": r,
                        "main_correlation": r, "score_protocol": SCORE_PROTOCOL, "selection_metric": "main_correlation"}
            with patch.object(train, "model_from_bundle", side_effect=toy_factory), \
                 patch.object(train, "evaluate", side_effect=fake_eval), \
                 patch.object(train, "public_diagnostic", return_value={}):
                for group in ("C", "M", "P"):
                    train.run(path, group, root / group, "cpu")
                    last = torch.load(root / group / "last.pt", weights_only=True)
                    self.assertEqual((last["stage"], last["step"]), (2, 1))
                    self.assertEqual(last["FM_state_sha256"], bundle["fm_state_sha256"])
                    # Resume from stage1 handoff must reproduce the exact stage2 final weights.
                    handoff = torch.load(root / group / "stage1_last.pt", weights_only=True)
                    torch.save(handoff, root / group / "last.pt")
                    train.run(path, group, root / group, "cpu", resume=True)
                    repeated = torch.load(root / group / "last.pt", weights_only=True)
                    torch.testing.assert_close(last["model"]["scale"], repeated["model"]["scale"], rtol=0, atol=0)
            compare(root, root / "comparison")
            self.assertEqual(len(read(root / "comparison/summary.json")["rows"]), 3)

    def test_export_preserves_FM_and_carries_reliable_input_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, bundle = study_fixture(root)
            score = {"task1_r_missing11": .7, "task2_r_missing11": .69, "main_correlation": .695,
                     "score_protocol": SCORE_PROTOCOL}
            cp = {"protocol": PROTOCOL, "study_sha256": sha(path), "group": "P", "stage": 2, "step": 1,
                  "model": bundle["slow"]["model"], "FM_state_sha256": bundle["fm_state_sha256"], "validation_score": score}
            torch.save(cp, root / "best.pt")
            export(path, root / "best.pt", root / "deploy.pt")
            loaded = load_bundle(root / "deploy.pt")
            self.assertEqual(loaded["fm_state_sha256"], bundle["fm_state_sha256"])
            self.assertEqual(loaded["reference_score_protocol"], SCORE_PROTOCOL)
            with self.assertRaises(FileExistsError): export(path, root / "best.pt", root / "deploy.pt")

    def test_actual_hybrid_scoring_filters_bad_I_and_preserves_devices(self):
        from baselines.B5_public_slow.evaluate import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, bundle = study_fixture(root)
            study = read(path)
            cache = root / "source/cache"
            manifest = read(cache / "cache.json")
            bad = synthetic_record(); bad["mask"][0] = False
            np.savez(cache / "bad.npz", **bad)
            original_rows = copy.deepcopy(manifest["validation_rows"])
            for entry in list(manifest["validation"]):
                task = entry["cache_key"].split(":")[0]
                pair = entry["pair_id"] + "bad"
                manifest["validation"].append(dict(entry, file="bad.npz", sha256=sha(cache / "bad.npz"),
                    subject_id="BadI", target_record_id="bad", pair_id=pair, cache_key=task + ":" + pair))
                source_rows = [r for r in original_rows[task] if r["pair_id"] == entry["pair_id"]]
                manifest["validation_rows"][task] += [dict(r, pair_id=pair, target_record_id="bad", subject_id="BadI")
                                                        for r in source_rows]
            write(cache / "cache.json", manifest)
            score = evaluate(Tiny(), study, bundle, torch.device("cpu"), root / "real_score")
            self.assertEqual(score["score_protocol"], SCORE_PROTOCOL)
            self.assertTrue(np.isfinite(score["main_correlation"]))
            # This exercises the real shared scorer, not mocked metric values.
            evaluation = read(root / "real_score/evaluation.json")
            self.assertTrue(evaluation["require_reliable_input"])

    def test_prepare_refuses_original_FM_cache_even_if_directory_says_control100(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, _ = study_fixture(root)
            components = root / "source/shared/components.pt"
            other = load_bundle(components)
            key = next(iter(other["fm"]["ema"]))
            other["fm"]["ema"][key].add_(.01)
            other["fm_state_sha256"] = state_sha(other["fm"]["ema"])
            torch.save(other, components)
            source = root / "source/shared/study.json"
            settings = read(source); settings["bundle_sha256"] = sha(components); write(source, settings)
            manifest_path = root / "source/cache/cache.json"
            manifest = read(manifest_path); manifest["study_sha256"] = sha(source); write(manifest_path, manifest)
            with patch.object(prepare, "DEFAULT_PROFILE", root / "profile.json"):
                with self.assertRaisesRegex(ValueError, "another FM"):
                    prepare.prepare(root / "initial.pt", root / "source/cache", root / "public", root / "wrong")

    def test_public_preparation_runs_from_real_WFDB_files_to_completed_manifest(self):
        import wfdb
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, _ = study_fixture(root)
            raw = root / "raw"
            for i in range(4):
                folder = raw / f"patient{i:03d}"
                folder.mkdir(parents=True)
                signal = np.tile(((.1 + i * .02) * np.sin(np.arange(60000) / 30))[:, None], (1, 12))
                wfdb.wrsamp("record", fs=1000, units=["mV"] * 12, sig_name=list(ptb.LEADS),
                    p_signal=signal, fmt=["16"] * 12, adc_gain=[2000.] * 12, baseline=[0] * 12,
                    comments=["age: 40", "sex: male"], write_dir=str(folder))
            ptb.prepare(raw, root / "initial.pt", root / "processed", min_train_patients=2, min_validation_patients=1)
            result = read(root / "processed/public.json")
            self.assertTrue(result["completed"])
            self.assertEqual(len(result["train"]), 3)
            self.assertEqual(len(result["validation"]), 1)
            for part in ("train", "validation"):
                for entry in result[part]:
                    self.assertEqual(entry["native_samples"], 3000)

    def test_mid_stage2_interruption_restores_optimizer_and_sampling_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = study_fixture(root)
            settings = read(path)
            settings["training"].update(stage1_updates=2, stage2_updates=2)
            write(path, settings)
            def fake_eval(model, study, bundle, device, output, *args, **kwargs):
                Path(output).mkdir(parents=True)
                r = float(model.scale.detach()) / 2
                return {"task1_r_missing11": r, "task2_r_missing11": r,
                        "main_correlation": r, "score_protocol": SCORE_PROTOCOL, "selection_metric": "main_correlation"}
            original_save = train.atomic_save
            def interrupted_save(destination, payload):
                original_save(destination, payload)
                if Path(destination).name == "last.pt" and payload["stage"] == 2 and payload["step"] == 1:
                    raise RuntimeError("synthetic interruption")
            with patch.object(train, "model_from_bundle", side_effect=lambda *args: Tiny()), \
                 patch.object(train, "evaluate", side_effect=fake_eval), \
                 patch.object(train, "public_diagnostic", return_value={}):
                train.run(path, "M", root / "full", "cpu")
                with patch.object(train, "atomic_save", side_effect=interrupted_save):
                    with self.assertRaisesRegex(RuntimeError, "interruption"):
                        train.run(path, "M", root / "interrupted", "cpu")
                self.assertFalse(read(root / "interrupted/completed.json")["completed"])
                train.run(path, "M", root / "interrupted", "cpu", resume=True)
            expected = torch.load(root / "full/last.pt", weights_only=True)
            actual = torch.load(root / "interrupted/last.pt", weights_only=True)
            torch.testing.assert_close(expected["model"]["scale"], actual["model"]["scale"], rtol=0, atol=0)
            self.assertEqual(expected["samplers"], actual["samplers"])


if __name__ == "__main__":
    unittest.main()
