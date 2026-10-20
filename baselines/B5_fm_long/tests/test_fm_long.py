from dataclasses import asdict
import copy
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from baselines.B5.flow import sample, keyed_noise
from baselines.B5.model import B5UNet
from baselines.B5.config import ModelConfig
from baselines.B5.metadata import encode_demographics
from baselines.B5_best.tests.test_hybrid import fixture
from baselines.B5_best.bundle import export_bundle
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_final.profile import BASE_SHA, SAMPLING
from baselines.B5_fm_long import PROTOCOL, PACKAGE_PROTOCOL, FIELDS
from baselines.B5_fm_long.common import (load_experiment, code_sha, validate_record,
    load_study, metadata, record_path, candidate_payload_matches)
from baselines.B5_fm_long.inference import generate_record, paired_noise, outputs
from baselines.B5_fm_long.prepare import train_groups
from baselines.B5_fm_long.evaluate import Evaluator
from baselines.B5_fm_long.predict import Predictor
from baselines.B5_fm_long.train import schedule, training_randomness, microbatches, run
from baselines.B5_fm_long.summarize import collect

torch.set_num_threads(1)
CPU = torch.device("cpu")
FAST_SETTINGS = {"seed": 42, "solver": "heun", "steps": 1, "samples": 1}


class Pointwise(nn.Module):
    """No temporal coupling: paired noise must make10s/40s outputs identical."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.))
        self.widths = []

    def encode_conditions(self, anchor, **unused):
        self.widths.append(anchor.shape[-1])
        return SimpleNamespace(anchor_prediction=anchor, anchor=anchor)

    def velocity(self, state, time, condition):
        return condition.anchor.repeat(1, 11, 1) - state


def record():
    wave = (100 * np.sin(np.arange(60000, dtype=np.float32) / 23)).astype(np.float32)
    return {**encode_demographics({"age": 40, "sex": "female"}),
            "anchor_uV": wave[None].copy(), "target_uV": np.tile(wave, (12, 1)),
            "slow_uV": np.zeros((11, 60000), np.float64), "mask": np.ones(12, bool)}


def small_bundle(root):
    fp, sp, ip, profile = fixture(root)
    path = root / "base.pt"
    bundle = export_bundle(fp, sp, ip, path, profile)
    return path, bundle


class FM40Tests(unittest.TestCase):
    def test_noise_40_is_same_four_old_10s_blocks(self):
        joined = paired_noise("record", 20000, 20000, 3, CPU)
        expected = torch.cat([keyed_noise(["record:" + str(start)], 5000, 42, 3, CPU)[0]
                              for start in range(20000, 40000, 5000)], dim=-1)
        torch.testing.assert_close(joined, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            paired_noise("record", 500, 20000, 0, CPU)

    def test_same_pointwise_model_and_physical_noise_give_same_full_output(self):
        d = record()
        model = Pointwise().eval()
        a = generate_record(model, d["anchor_uV"], metadata(d), "id", np.ones(12), CPU, 10, 2, FAST_SETTINGS)
        self.assertEqual(model.widths, [5000] * 6)
        model.widths.clear()
        b = generate_record(model, d["anchor_uV"], metadata(d), "id", np.ones(12), CPU, 40, 1, FAST_SETTINGS)
        self.assertEqual(model.widths, [20000] * 3)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[0], d["anchor_uV"][0])

    def test_10s_generation_matches_original_sampler(self):
        model = B5UNet(ModelConfig(channels=(4, 8, 8), time_dim=8, metadata_dim=8,
                                  metadata_dropout=0, dilations=(1,))).eval()
        d = record()
        actual = generate_record(model, d["anchor_uV"], metadata(d), "id", np.ones(12), CPU, 10, 3, FAST_SETTINGS)
        expected = []
        for start in range(0, 60000, 5000):
            inputs = {key: torch.from_numpy(np.asarray(d[key])[None].copy()) for key in FIELDS}
            inputs["anchor"] = torch.from_numpy(d["anchor_uV"][:, start:start + 5000][None].copy())
            expected.append(sample(model, inputs, ["id:" + str(start)], **FAST_SETTINGS)[0].numpy())
        np.testing.assert_allclose(actual, np.concatenate(expected, axis=-1), rtol=3e-5, atol=3e-5)

    def test_actual_unet_supports_40s_without_padding_or_weight_change(self):
        model = B5UNet(ModelConfig(channels=(4, 8, 8), time_dim=8, metadata_dim=8,
                                  metadata_dropout=0, dilations=(1,))).eval()
        digest = state_sha(model.state_dict())
        d = record()
        generated = generate_record(model, d["anchor_uV"], metadata(d), "id", np.ones(12), CPU, 40, 1, FAST_SETTINGS)
        self.assertEqual(generated.shape, (12, 60000))
        self.assertTrue(np.isfinite(generated).all())
        self.assertEqual(state_sha(model.state_dict()), digest)
        with self.assertRaises(ValueError):
            generate_record(model, d["anchor_uV"][:, :-1], metadata(d), "id", np.ones(12), CPU, 40)
        self.assertNotIn("target", inspect.signature(generate_record).parameters)

    def test_raw_and_hybrid_share_fm_and_preserve_i(self):
        d = record()
        raw = d["target_uV"].copy()
        modes = outputs(raw, d["slow_uV"])
        self.assertIs(modes["raw"], raw)
        np.testing.assert_array_equal(modes["hybrid"][0], raw[0])
        np.testing.assert_array_equal(raw, d["target_uV"])

    def test_schedule_and_training_noise_time_are_paired(self):
        self.assertEqual(schedule(42, 8, 156, 2), schedule(42, 8, 156, 2))
        self.assertNotEqual(schedule(42, 8, 156, 2), schedule(42, 9, 156, 2))
        d = record()
        z, t = training_randomness(42, 8, 0)
        short = list(microbatches(d, 20000, 10, np.ones(12), z, t, CPU))
        long = list(microbatches(d, 20000, 40, np.ones(12), z, t, CPU))
        self.assertEqual(len(short), 4)
        self.assertEqual(len(long), 1)
        for key in ("anchor", "target"):
            torch.testing.assert_close(torch.cat([x[0][key] for x in short], -1), long[0][0][key])
        torch.testing.assert_close(torch.cat([x[1] for x in short], -1), long[0][1])
        self.assertTrue(all(torch.equal(x[2], long[0][2]) for x in short))

    def test_train_group_rejects_gaps_duplicates_and_patient_mixing(self):
        rows = [{"target_record_id": "a", "subject_id": "HNU1", "source_split": "train",
                 "start_sample_500hz": str(s)} for s in range(0, 60000, 5000)]
        self.assertEqual(train_groups(rows), [list(range(12))])
        for broken in (rows[:-1], rows + rows[:1], [*rows[:-1], {**rows[-1], "subject_id": "HNU2"}]):
            with self.assertRaises(ValueError):
                train_groups(broken)

    def test_validation_keeps_bad_i_and_cache_paths_cannot_escape(self):
        d = record(); d["mask"][0] = False
        validate_record(d, training=False)
        with self.assertRaises(ValueError):
            validate_record(d, training=True)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                record_path(tmp, {"file": "../outside.npz"})

    def test_experiment_rejects_wrong_context_or_sampling(self):
        cfg = load_experiment()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wrong.json"
            for modified in ({**cfg, "context": "both"}, {**cfg, "sampling": {**SAMPLING, "samples": 1}},
                             {**cfg, "training": {**cfg["training"], "updates": 0}}):
                write(p, modified)
                with self.assertRaises(ValueError):
                    load_experiment(p)

    def test_full_evaluation_uses_public_qc_and_reuses_physical_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = {"scale_uV": [1.] * 12, "sampling": SAMPLING, "slow_state_sha256": "frozen",
                     "sources": {"population": "same"}, "validation": {}, "validation_rows": {}}
            for key in ("good", "bad"):
                d = record(); d["mask"][0] = key == "good"
                np.savez(root / (key + ".npz"), **d)
            for task in ("task1", "task2"):
                rows, entries = [], []
                devices = ("watch_ecg",) if task == "task1" else ("ecg_machine_d6", "body_scale_d6")
                for dev in devices:
                    for key in ("good", "bad"):
                        ids = list(range(len(rows), len(rows) + 12))
                        rows += [{"pair_id": dev + key, "target_record_id": key, "subject_id": key,
                                  "input_type": dev, "start_sample_500hz": str(s), "expected_window_count": "12"}
                                 for s in range(0, 60000, 5000)]
                        entries.append({"file": key + ".npz", "target_record_id": key, "window_indices": ids})
                study["validation"][task], study["validation_rows"][task] = entries, rows
            write(root / "study.json", study)
            calls = []
            def generate(model, anchor, meta, key, *args):
                calls.append(key)
                return np.tile(anchor, (12, 1)) * (1 if key == "good" else -1)
            with patch("baselines.B5_fm_long.evaluate.generate_record", side_effect=generate):
                result = Evaluator(root, study, CPU).evaluate(Pointwise().eval(), 40, root / "evaluation")
            self.assertEqual(calls, ["good", "bad"])
            self.assertAlmostEqual(result["summary"]["raw"]["mean_r"], 1.)
            self.assertEqual(result["context"], "disabled")
            self.assertTrue((root / "evaluation/raw/task2/input_qc_exclusions.csv").is_file())

    def test_portable_predictor_does_not_require_original_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bundle = small_bundle(root)
            p = root / "package.pt"
            torch.save({"protocol": PACKAGE_PROTOCOL, "source_base_sha256": BASE_SHA, "window_seconds": 40,
                        "context": "disabled", "sampling": SAMPLING, "slow_beta": .75,
                        "code_sha256": code_sha(), "target_used_at_inference": False,
                        "training_cache_required": False, "bundle": bundle}, p)
            source.unlink()
            engine = Predictor(p, CPU)
            d = record()
            def fast(*args, **kwargs):
                return generate_record(*args, **kwargs, sampling=FAST_SETTINGS)
            with patch("baselines.B5_fm_long.predict.generate_record", side_effect=fast):
                result = engine.predict(d["anchor_uV"], {}, "new")
            self.assertEqual(set(result), {"raw", "hybrid"})
            self.assertTrue(all(v.shape == (12, 60000) and np.isfinite(v).all() for v in result.values()))
            self.assertFalse(any(p.requires_grad for p in engine.slow.parameters()))

    def test_study_rejects_cache_tampering_and_changed_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bundle = small_bundle(root)
            train, val = record(), record()
            np.savez(root / "train.npz", **train)
            np.savez(root / "val.npz", **val)
            study = {"protocol": PROTOCOL, "completed": True, "base_checkpoint": str(source),
                "base_sha256": sha(source), "sampling": SAMPLING, "code_sha256": code_sha(),
                "context": "disabled", "experiment": load_experiment(), "model_config": bundle["fm"]["model_config"],
                "scale_uV": bundle["fm"]["scales"]["d12"], "initial_fm_state_sha256": bundle["fm_state_sha256"],
                "slow_state_sha256": bundle["slow_state_sha256"],
                "train": [{"file": "train.npz", "sha256": sha(root / "train.npz"), "subject_id": "HNU1"}],
                "validation": {}, "validation_rows": {}}
            for task, dev in (("task1", "watch_ecg"), ("task2", "ecg_machine_d6")):
                rows = [{"pair_id": "p", "target_record_id": "val", "subject_id": "HNU2", "input_type": dev,
                         "start_sample_500hz": str(s)} for s in range(0, 60000, 5000)]
                study["validation_rows"][task] = rows
                study["validation"][task] = [{"file": "val.npz", "sha256": sha(root / "val.npz"),
                    "subject_id": "HNU2", "target_record_id": "val", "input_type": dev, "pair_id": "p",
                    "window_indices": list(range(12))}]
            write(root / "study.json", study)
            with patch("baselines.B5_fm_long.common.BASE_SHA", sha(source)):
                self.assertEqual(load_study(root)["context"], "disabled")
                bad = copy.deepcopy(study)
                bad["validation_rows"]["task1"][0]["target_record_id"] = "other"
                write(root / "study.json", bad)
                with self.assertRaisesRegex(ValueError, "identity"):
                    load_study(root)
                write(root / "study.json", study)
                train["target_uV"][1, 0] += 1
                np.savez(root / "train.npz", **train)
                with self.assertRaisesRegex(ValueError, "cache changed"):
                    load_study(root)

    def test_summary_accepts_only_matched_screening_populations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {"protocol": PROTOCOL, "completed": True, "context": "disabled",
                      "study_sha256": "study", "population": {"qc": "same"}, "summary": {}}
            for sec in (10, 40):
                value = copy.deepcopy(report)
                value["summary"] = {mode: {"mode": mode, "label": "initial", "window_seconds": sec,
                    "update": 0, "mean_r": .7, "task1_r": .7, "task2_r": .7} for mode in ("raw", "hybrid")}
                write(root / f"screen/FM{sec}/evaluation.json", value)
            self.assertEqual(len(collect(root)), 4)
            path = root / "screen/FM40/evaluation.json"
            changed = read(path); changed["population"]["qc"] = "different"; write(path, changed)
            with self.assertRaisesRegex(ValueError, "different patient"):
                collect(root)

    def test_synthetic_training_step0_best_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bundle = small_bundle(root)
            d = record(); np.savez(root / "train.npz", **d)
            experiment = load_experiment()
            experiment["training"].update(updates=2, validate_every=1, records_per_update=1)
            study = {"experiment": experiment, "base_checkpoint": str(source), "model_config": bundle["fm"]["model_config"],
                     "scale_uV": [100.] * 12, "fm_loss": {"huber": .1, "pcc": .1, "anchor": .02,
                     "physiology": 0., "huber_delta": 1.}, "initial_fm_state_sha256": bundle["fm_state_sha256"],
                     "train": [{"file": "train.npz", "target_record_id": "a"}]}
            write(root / "study.json", study)
            class FakeEvaluator:
                fail = False
                def __init__(self, *unused): pass
                def evaluate(self, model, seconds, output, label, update):
                    if self.fail and update == 2:
                        raise RuntimeError("simulated interruption")
                    return {"summary": {m: {"mean_r": .5 + update / 100} for m in ("raw", "hybrid")}}
            def arguments(name, resume=False, execute=True):
                return SimpleNamespace(run=str(root), output=str(root / name), window_seconds=40,
                                       device="cpu", batch_size=1, execute_training=execute, resume=resume)
            with patch("baselines.B5_fm_long.train.load_study", return_value=study), \
                 patch("baselines.B5_fm_long.train.Evaluator", FakeEvaluator):
                with self.assertRaisesRegex(ValueError, "opt-in"):
                    run(arguments("disabled", execute=False))
                self.assertFalse((root / "disabled").exists())
                run(arguments("complete"))
                FakeEvaluator.fail = True
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    run(arguments("interrupted"))
                saved = torch.load(root / "interrupted/last.pt", weights_only=True)
                self.assertEqual(saved["update"], 1)
                FakeEvaluator.fail = False
                run(arguments("interrupted", resume=True))
            a = torch.load(root / "complete/last.pt", weights_only=True)
            b = torch.load(root / "interrupted/last.pt", weights_only=True)
            self.assertEqual(a["state_sha256"], b["state_sha256"])
            self.assertEqual(a["best"]["raw"]["update"], 2)
            self.assertTrue(read(root / "complete/completed.json")["completed"])
            self.assertEqual(a["training_waveform_seconds"], 80)
            from baselines.B5_fm_long.predict import export
            package = root / "trained_package.pt"
            with patch("baselines.B5_fm_long.predict.load_study", return_value=study):
                export(root, root / "complete/best_raw.pt", package)
            source.unlink()
            (root / "train.npz").unlink()
            deployed = Predictor(package, CPU)
            self.assertEqual(state_sha(deployed.fm.state_dict()), a["state_sha256"])
            self.assertEqual(deployed.seconds, 40)


if __name__ == "__main__":
    unittest.main()
