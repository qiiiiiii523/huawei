"""Synthetic CPU checks; never reads user ECGs or starts a server training job."""
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import wfdb

from baselines.B5.checkpoint import load_checkpoint, atomic_save, checkpoint_preprocessor
from baselines.B5.config import load_config
from baselines.B5.metadata import encode_demographics
from baselines.B5.scoring import correlation_score, SCORE_PROTOCOL
from baselines.B5_best.tests.test_promoted_default import promoted_fixture
from baselines.B5_best.combine import combine
from baselines.B5_best.predict import Predictor
from baselines.B5_public_fm import prepare as preparation, train as training, evaluation
from baselines.B5_public_fm import check as checking, recombine, compare, pack
from baselines.B5_public_fm.common import ROOT, read, write, sha, state_sha, load_study, training_lock
from baselines.B5_public_fm.public_data import (
    physical_uV, quality_mask, descriptors, audit_public, PublicDataset, chapman_split)
from baselines.B5_public_fm.stream import PairedStream

torch.set_num_threads(2)


def wave(seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(5000)
    return np.stack([100 * np.sin(t / (30 + j)) + rng.normal(0, .5, 5000) + seed * 2 for j in range(12)]).astype(np.float32)


def public_files(root):
    ptb, chapman = root / "ptb", root / "chapman"
    (ptb / "records500").mkdir(parents=True)
    chapman.mkdir()
    lines = ["ecg_id,patient_id,strat_fold,filename_hr,age,sex,height,weight"]
    for i, fold in enumerate((1, 9, 10)):
        name = f"p{i}"
        wfdb.wrsamp(name, fs=500, units=["mV"] * 12,
                    sig_name=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                    p_signal=wave(i + 1).T.astype(float) / 1000, fmt=["16"] * 12,
                    adc_gain=[1000.] * 12, baseline=[0] * 12, write_dir=str(ptb / "records500"))
        lines.append(f"{i},{i},{fold},records500/{name},40,1,,")
    (ptb / "ptbxl_database.csv").write_text("\n".join(lines), encoding="utf-8")
    selected = {}
    for i in range(100):
        name = f"C{i:03d}"
        selected.setdefault(chapman_split(name), name)
    for i, name in enumerate(selected.values()):
        wfdb.wrsamp(name, fs=500, units=["uV"] * 12,
                    sig_name=["V6", "V5", "V4", "V3", "V2", "V1", "aVF", "aVL", "aVR", "III", "II", "I"],
                    p_signal=wave(i + 10)[::-1].T.astype(float), fmt=["16"] * 12,
                    adc_gain=[1.] * 12, baseline=[0] * 12,
                    comments=["Age: 55", "Sex: Female"], write_dir=str(chapman))
    return ptb, chapman


class TrainFixture:
    manifest_digest = "train-fixture"
    def __len__(self):
        return 1

    def __getitem__(self, index):
        raw = wave(55)
        return {"anchor": raw[:1] / 100, "target": raw / 100,
                "quality_mask": np.ones(12, bool), **encode_demographics({"age": 40, "sex": "male"}),
                "key": "train:0", "evaluation_metadata": {"target_record_id": "train", "start_sample_500hz": "0"}}


class ValidationFixture:
    def __init__(self, task):
        devices = ["watch_ecg"] if task == "task1" else ["ecg_machine_d6", "body_scale_d6"]
        self.rows = [{"pair_id": dev + str(r), "target_record_id": "validation" + str(r), "subject_id": "S", "input_type": dev,
                      "start_sample_500hz": str(s), "expected_window_count": "12"}
                     for dev in devices for r in range(2) for s in range(0, 60000, 5000)]
        self.manifest_digest = task
        t = np.arange(60000)
        self.raw = np.stack([50 * np.sin(t / (20 + j)) + (t >= 5000) * 40 for j in range(12)]).astype(np.float32)
        self.base = SimpleNamespace()
        outer = self
        class Base:
            def __getitem__(self, i):
                s = int(outer.rows[i]["start_sample_500hz"])
                return SimpleNamespace(anchor_i_ecg=outer.raw[:1, s:s + 5000])
        self.base = Base()
        self.demographics = SimpleNamespace(get=lambda _: encode_demographics({}))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        s = int(self.rows[i]["start_sample_500hz"])
        quality = np.ones(12, bool)
        quality[9] = self.rows[i]["target_record_id"] != "validation0"
        return {"target_uV": self.raw[:, s:s + 5000], "quality_mask": quality, "evaluation_metadata": self.rows[i]}


def prepared_fixture(root):
    source, bundle, profile_path, _, raw, _ = promoted_fixture(root)
    public = load_checkpoint(raw[0])
    public.update(stage="public", model=copy.deepcopy(public["ema"]), loss_config={"huber": .1, "pcc": .1, "anchor": .02, "physiology": 0., "huber_delta": 1.})
    atomic_save(raw[0], public)
    config = load_config(ROOT / "configs/experiments/b5_main_v2_public_finetune_meta.yaml")
    config["model"] = public["model_config"]
    config["training"].update(batch_size=1, epochs=5)
    config["paths"]["preprocessing_config"] = str(ROOT / "configs/preprocessing.yaml")
    settings = read(ROOT / "configs/b5_public_fm.json")
    settings["public"].update(updates=4, save_every=2, batch_size=1, validation_records_per_source=1)
    settings["huawei"]["epochs"] = 5
    settings_path = root / "settings.json"
    write(settings_path, settings)
    ptb, chapman = public_files(root)
    data = root / "frozen_source"
    data.write_text("source", encoding="utf-8")
    keys = ("demographics", "subject_split", "device_QC", "task1_pairs", "task2_pairs",
            "task1_metadata", "task2_metadata", "task1_validation_target", "task2_validation_target")
    sources = {k: {"path": str(data), "sha256": sha(data)} for k in keys}
    args = SimpleNamespace(settings=settings_path, repository_root=ROOT, data_reference=None, public_checkpoint=str(raw[0]),
                           fixed_bundle=source, profile=profile_path, data_config=None, output=root / "shared",
                           ptbxl_root=ptb, chapman_root=chapman, chapman_patient_map=None,
                           chapman_probability=None, expected_chapman_records=3)
    val = {t: ValidationFixture(t) for t in ("task1", "task2")}
    with patch.object(preparation, "load_config", return_value=config), \
         patch.object(preparation, "huawei_sources", return_value=sources), \
         patch.object(preparation, "build_datasets", return_value=(TrainFixture(), val)), \
         patch("sys.stdout", io.StringIO()):
        preparation.prepare(args)
    return root / "shared/study.json", val


class PublicDataTests(unittest.TestCase):
    def test_physical_units_lead_order_and_qc_mask(self):
        names = ["V6", "V5", "V4", "V3", "V2", "V1", "aVF", "aVL", "aVR", "III", "II", "I"]
        x = np.tile(np.arange(12), (5000, 1)).astype(float)
        raw = physical_uV(x, ["mV"] * 12, names, 500)
        np.testing.assert_array_equal(raw[:, 0], np.arange(11, -1, -1) * 1000)
        with self.assertRaises(ValueError):
            physical_uV(x, ["unknown"] * 12, names, 500)
        with self.assertRaises(ValueError):
            physical_uV(x, ["uV"] * 12, names, 100)
        raw = wave()
        raw[2] = np.nan
        raw[5] = 0
        digital = np.zeros_like(raw)
        digital[8] = 32767
        mask, why = quality_mask(raw, digital, [16] * 12)
        self.assertFalse(mask[2] or mask[5] or mask[8])
        self.assertTrue(mask[0])
        self.assertIn("ADC_rail", why[8])

    def test_real_wfdb_audit_converts_units_freezes_splits_and_missing_demographics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ptb, chap = public_files(root)
            summary = audit_public(ptb, chap, root / "cache", expected_chapman_records=3)
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["counts"]["chapman"]["train"]["records"], 1)
            manifest = read(root / "cache/manifest.json")
            patients = {}
            for r in manifest:
                self.assertEqual(patients.setdefault(r["patient"], r["split"]), r["split"])
                cached = np.load(root / "cache" / r["cache"])
                self.assertGreater(np.ptp(cached[0]), 150)
            (ptb / "records500/p0.dat").unlink()
            with self.assertRaises(ValueError):
                audit_public(ptb, chap, root / "broken", expected_chapman_records=3)
            self.assertTrue(read(root / "broken/errors.json"))

    def test_duplicate_heldout_waveform_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ptb, chap = public_files(root)
            # Add a train record that is an exact duplicate of PTB validation.
            import shutil
            shutil.copyfile(ptb / "records500/p1.dat", ptb / "records500/dup.dat")
            text = (ptb / "records500/p1.hea").read_text().replace("p1", "dup")
            (ptb / "records500/dup.hea").write_text(text)
            with (ptb / "ptbxl_database.csv").open("a") as f:
                f.write("\n99,99,1,records500/dup,40,1,,")
            summary = audit_public(ptb, chap, root / "cache", expected_chapman_records=3)
            self.assertEqual(summary["excluded"]["exact_duplicate"], 1)
            self.assertFalse(any(r["key"] == "ptbxl:99" for r in read(root / "cache/manifest.json")))

    def test_incomplete_release_and_patient_cross_fold_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ptb, chap = public_files(root)
            with self.assertRaisesRegex(ValueError, "45152"):
                audit_public(ptb, chap, root / "cache")
            meta = ptb / "ptbxl_database.csv"
            meta.write_text(meta.read_text().replace("1,1,9", "1,0,9"))
            with self.assertRaisesRegex(ValueError, "crosses folds"):
                descriptors(ptb, chap)


class PairAndResumeTests(unittest.TestCase):
    def test_stream_restarts_exactly_and_control_never_draws_chapman(self):
        control = PairedStream({"ptbxl": 3, "chapman": 7}, 0, 42)
        self.assertEqual({s for s, _ in control.next(50)}, {"ptbxl"})
        stream = PairedStream({"ptbxl": 3, "chapman": 7}, .7, 42)
        stream.next(19)
        state = stream.state_dict()
        resumed = PairedStream({"ptbxl": 3, "chapman": 7}, .7, 0)
        resumed.load_state_dict(state)
        self.assertEqual(stream.next(37), resumed.next(37))
        self.assertEqual(stream.coverage(), resumed.coverage())

    def test_prepare_and_no_update_CPU_preflight_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, val = prepared_fixture(root)
            with patch.object(checking, "build_datasets", return_value=(TrainFixture(), val)), patch("sys.stdout", io.StringIO()):
                checking.check(path, root / "preflight.json")
            self.assertEqual(read(root / "preflight.json")["optimizer_updates"], 0)
            load_study(path)
            cached = read(root / "shared/public/manifest.json")[0]["cache"]
            (root / "shared/public" / cached).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "waveform cache changed"):
                load_study(path)

    def test_public_actual_synthetic_updates_resume_match_uninterrupted_and_no_slow_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = prepared_fixture(root)
            _, study = load_study(path)
            fixed_sha = sha(root / "shared/fixed_slow_bundle.pt")
            with patch.object(training, "public_diagnostic", side_effect=lambda *args: write(args[-1], {"completed": True})), patch("sys.stdout", io.StringIO()):
                training.run_public(path, study, "expanded", root / "full", torch.device("cpu"))
                original = training.step_update
                calls = [0]
                def interrupt(*args):
                    calls[0] += 1
                    if calls[0] == 3:
                        raise InterruptedError("synthetic interruption after committed update2")
                    return original(*args)
                with patch.object(training, "step_update", side_effect=interrupt), self.assertRaises(InterruptedError):
                    training.run_public(path, study, "expanded", root / "split", torch.device("cpu"))
                training.run_public(path, study, "expanded", root / "split", torch.device("cpu"), True)
            a = load_checkpoint(root / "full/public/last.pt")
            b = load_checkpoint(root / "split/public/last.pt")
            self.assertEqual(a["updates"], 4)
            self.assertEqual(state_sha(a["ema"]), state_sha(b["ema"]))
            self.assertEqual(state_sha(a["model"]), state_sha(b["model"]))
            self.assertEqual(a["stream"], b["stream"])
            self.assertEqual(fixed_sha, sha(root / "shared/fixed_slow_bundle.pt"))
            with self.assertRaises(ValueError):
                training.validate_resume(a, {**a["job_signature"], "group": "control"}, root / "full/public")

    def test_same_group_output_lock_rejects_parallel_training_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            with training_lock(tmp):
                with self.assertRaises(RuntimeError):
                    with training_lock(tmp):
                        pass
            with training_lock(tmp):
                pass

    def test_training_cli_requires_explicit_gate(self):
        with patch("sys.argv", ["train", "--study", "missing", "--group", "control", "--output", "missing"]), \
             patch.object(training, "load_study", side_effect=AssertionError("No data access before training consent")), \
             self.assertRaises(SystemExit):
            training.main()


class EvaluationTests(unittest.TestCase):
    def test_full_record_combination_saved_raw_and_exported_predictor_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, datasets = prepared_fixture(root)
            _, study = load_study(path)
            # Use a deterministic FM echo of I. The test's expected full-record filter
            # crosses the 10-second seam, not a per-window approximation.
            def echo(model, inputs, keys, **sampling):
                return inputs["anchor"].repeat(1, 12, 1)
            with patch.object(evaluation, "build_validation_datasets", return_value=datasets), \
                 patch.object(evaluation, "sample", side_effect=echo) as fm_sample, \
                 patch.object(evaluation, "predict_native", return_value=torch.ones(1, 11, 6000)):
                evaluator = evaluation.HybridEvaluator(path, study, torch.device("cpu"))
                cp = load_checkpoint(root / "shared/public_initializer.pt")
                model = __import__("baselines.B5.checkpoint", fromlist=["model_from_checkpoint"]).model_from_checkpoint(cp, torch.device("cpu"))
                score = evaluator.evaluate(model, root / "eval")
                self.assertEqual(fm_sample.call_count, 4)  # Two targets, two batches; repeated task2 pairs reuse them.
            saved = np.load(root / "eval/task1/prediction_uV.npy")
            raw = np.concatenate(np.load(root / "eval/task1/fm_prediction_uV.npy")[:12], -1)
            expected = combine(raw, np.ones((11, 60000)) * 100, .75)
            np.testing.assert_array_equal(np.concatenate(saved[:12], -1), expected)
            self.assertEqual(score["score_protocol"], SCORE_PROTOCOL)
            # Exported target-free Predictor must produce the same hybrid.
            bundle = evaluation.candidate_bundle(cp, evaluator.fixed, reference=score)
            with patch("baselines.B5_best.predict.sample", side_effect=echo), \
                 patch("baselines.B5_best.predict.predict_native", return_value=torch.ones(1, 11, 6000)):
                result = Predictor(bundle, torch.device("cpu")).predict_encoded(datasets["task1"].raw[:1], encode_demographics({}), "validation", 8)
            np.testing.assert_array_equal(result, expected)
            # Replacing only the slow component must reuse and verify raw FM bytes.
            with patch.object(evaluation, "build_validation_datasets", return_value=datasets), \
                 patch.object(evaluation, "predict_native", return_value=torch.ones(1, 11, 6000)), \
                 patch.object(evaluation, "sample", side_effect=AssertionError("No FM generation during recombination")), \
                 patch("sys.argv", ["recombine", "--study", str(path), "--evaluation", str(root / "eval"),
                      "--slow-bundle", str(root / "shared/fixed_slow_bundle.pt"), "--output", str(root / "recombined"), "--device", "cpu"]), \
                 patch("sys.stdout", io.StringIO()):
                recombine.main()
            np.testing.assert_array_equal(np.load(root / "recombined/task1/prediction_uV.npy"), saved)

    def test_huawei_epoch_zero_selection_candidates_and_export_no_slow_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = prepared_fixture(root)
            _, study = load_study(path)
            with patch.object(training, "public_diagnostic", side_effect=lambda *args: write(args[-1], {"completed": True})), patch("sys.stdout", io.StringIO()):
                training.run_public(path, study, "control", root / "control", torch.device("cpu"))
            fixed = __import__("baselines.B5_best.bundle", fromlist=["load_bundle"]).load_bundle(root / "shared/fixed_slow_bundle.pt")
            class FakeEvaluator:
                def __init__(self, *args):
                    self.fixed = fixed
                def evaluate(self, model, output):
                    Path(output).mkdir(parents=True, exist_ok=True)
                    # Initial model wins: continuation must not erase epoch-zero best.
                    return correlation_score(.8, .8) if str(output).endswith("0000") else correlation_score(.6, .6)
            with patch.object(training, "HuaweiTrainDataset", return_value=TrainFixture()), \
                 patch.object(training, "demographics_table"), patch.object(training, "HybridEvaluator", FakeEvaluator), patch("sys.stdout", io.StringIO()):
                training.run_huawei(path, study, "control", root / "control", torch.device("cpu"))
            output = root / "control/huawei"
            best = load_checkpoint(output / "best.pt")
            last = load_checkpoint(output / "last.pt")
            self.assertEqual(best["epoch"], 0)
            self.assertEqual(last["epoch"], 5)
            self.assertTrue((output / "candidates/epoch_0005.pt").is_file())
            packaged = __import__("baselines.B5_best.bundle", fromlist=["load_bundle"]).load_bundle(output / "best_bundle.pt")
            self.assertEqual(packaged["slow_state_sha256"], fixed["slow_state_sha256"])
            self.assertEqual(packaged["reference_score_protocol"], SCORE_PROTOCOL)
            self.assertEqual(packaged["reference_qc_r"]["task1_r_missing11"], .8)

    def test_comparison_rejects_mixed_QC_and_pack_excludes_waveforms_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = prepared_fixture(root)
            run = root / "run"
            report = {"completed": True, "study_sha256": sha(path), "population": {"task1": "a", "task2": "b"},
                      "score_protocol": SCORE_PROTOCOL, "sampling": read(path)["settings"]["sampling"],
                      "slow_beta": .75, "fixed_slow_state_sha256": "same", "score": correlation_score(.7, .7)}
            write(run / "baseline/evaluation.json", report)
            for group in ("control", "expanded"):
                base = run / group / "huawei"
                write(base / "completed.json", {"completed": True, "study_sha256": sha(path), "best": {"epoch": 5}, "epochs": 5})
                write(base / "evaluations/epoch_0005/evaluation.json", report)
            compare.summarize(path, run / "control", run / "expanded", run / "baseline", run / "comparison")
            self.assertTrue(read(run / "comparison/comparison.json")["completed"])
            wrong = copy.deepcopy(report)
            wrong["population"]["task2"] = "other-QC"
            write(run / "expanded/huawei/evaluations/epoch_0005/evaluation.json", wrong)
            with self.assertRaisesRegex(ValueError, "Cannot compare"):
                compare.summarize(path, run / "control", run / "expanded", run / "baseline", run / "comparison2")
            write(run / "shared/study.json", {})
            (run / "weights.pt").write_bytes(b"large weights")
            (run / "waveforms.npy").write_bytes(b"private waveforms")
            archive = root / "reports.tar.gz"
            with patch("sys.argv", ["pack", "--run", str(run), "--output", str(archive)]), patch("sys.stdout", io.StringIO()):
                pack.main()
            import tarfile
            with tarfile.open(archive) as packed:
                names = packed.getnames()
            self.assertTrue(any(name.endswith("summary.csv") for name in names))
            self.assertFalse(any(name.endswith((".pt", ".npy")) for name in names))


if __name__ == "__main__":
    unittest.main()
