from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ecg12gen.contracts import D12_LEADS
from ecg12gen.evaluate import evaluate_record_predictions
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from baselines.B5 import ARCHITECTURE_ID, CONDITION_SCHEMA, PREPROCESSING_VERSION
from baselines.B5.checkpoint import atomic_save, checkpoint_preprocessor, load_checkpoint, model_from_checkpoint, pack_rng, restore_rng, scales_digest, scales_payload
from baselines.B5.conditions import FiLM, MetadataEncoder
from baselines.B5.config import ModelConfig, load_config
from baselines.B5.data import HuaweiTrainDataset, HuaweiValidationDataset, fit_huawei_scales
from baselines.B5.flow import integrate, keyed_noise, linear_path, sample
from baselines.B5.losses import flow_loss
from baselines.B5.metadata import DemographicsTable, encode_demographics
from baselines.B5.model import B5UNet
from baselines.B5.predict import canonical_anchor, predict_record, restore_rate
from baselines.B5.public_adapter import PTBXLDataset, lead_order, locate_ptbxl, physical_to_uV

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[3]


def tiny_model() -> B5UNet:
    torch.manual_seed(4)
    return B5UNet(ModelConfig(channels=(8, 16, 32), time_dim=16, metadata_dim=16, metadata_dropout=0.)).eval()


def inputs(batch: int = 2, length: int = 64) -> dict[str, torch.Tensor]:
    return {"anchor": torch.linspace(-1, 2, length).reshape(1, 1, -1).repeat(batch, 1, 1),
            "numeric": torch.ones(batch, 3) * .5, "sex": torch.zeros(batch, dtype=torch.long),
            "field_mask": torch.ones(batch, 4, dtype=torch.bool), "age_topcoded": torch.zeros(batch, 1)}


def preprocessor() -> ECGPreprocessor:
    config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    scale = np.linspace(100, 1200, 12, dtype=np.float32)
    return ECGPreprocessor(config, {"d12": scale, "ecg_machine_i": scale[:1].copy()})


def checkpoint(model: B5UNet, processor: ECGPreprocessor) -> dict:
    scales = scales_payload(processor)
    return {"architecture_id": ARCHITECTURE_ID, "architecture_hash": model.config.fingerprint,
            "model_config": asdict(model.config), "condition_schema": CONDITION_SCHEMA,
            "preprocessing_version": PREPROCESSING_VERSION, "scales": scales,
            "scales_sha256": scales_digest(scales), "model": model.state_dict(), "ema": model.state_dict(),
            "config": {"sampling": {"seed": 42, "steps": 1, "solver": "euler", "samples": 1}}}


class ModelAndFlowTests(unittest.TestCase):
    def test_default_full_window_and_learned_i(self) -> None:
        model = B5UNet(ModelConfig(metadata_dropout=0.)).eval()
        data = inputs(1, 5000)
        with torch.no_grad():
            velocity, anchor = model(torch.zeros(1, 11, 5000), torch.tensor([.5]), **{
                "anchor": data["anchor"], "numeric": data["numeric"], "sex": data["sex"],
                "field_mask": data["field_mask"], "age_topcoded": data["age_topcoded"]})
        self.assertEqual(velocity.shape, (1, 11, 5000))
        self.assertEqual(anchor.shape, (1, 1, 5000))
        self.assertTrue(torch.isfinite(velocity).all())
        self.assertFalse(torch.equal(anchor, data["anchor"]))

    def test_masked_target_never_enters_state(self) -> None:
        target = torch.ones(2, 11, 64) * 4
        noise = torch.ones_like(target) * -2
        mask = torch.ones(2, 11, dtype=torch.bool)
        mask[:, 3] = False
        target[:, 3] = float("nan")
        state, velocity = linear_path(target, mask, noise, torch.tensor([0., 1.]))
        self.assertTrue(torch.equal(state[0], noise[0]))
        self.assertTrue(torch.equal(state[1, 0], target[1, 0]))
        self.assertTrue(torch.equal(state[:, 3], noise[:, 3]))
        self.assertTrue(torch.equal(velocity[:, 3], torch.zeros_like(velocity[:, 3])))
        self.assertTrue(torch.isfinite(state).all())

    def test_ode_solver_constant_field(self) -> None:
        class Constant(torch.nn.Module):
            def velocity(self, state, time, condition):
                return torch.ones_like(state) * 3
        model = Constant().eval()
        start = torch.randn(2, 11, 64)
        for solver in ("euler", "heun"):
            result = integrate(model, None, start, steps=7, solver=solver)
            torch.testing.assert_close(result, start + 3)

    def test_heun_improves_linear_ode(self) -> None:
        class Linear(torch.nn.Module):
            def velocity(self, state, time, condition):
                return state
        model = Linear().eval()
        initial = torch.ones(1, 11, 8)
        euler = integrate(model, None, initial, 8, "euler")
        heun = integrate(model, None, initial, 8, "heun")
        exact = initial * np.e
        self.assertLess(float((heun - exact).abs().mean()), float((euler - exact).abs().mean()))

    def test_noise_independent_of_batch_and_order(self) -> None:
        together = keyed_noise(["a:0", "b:5000"], 64, 42, 0, torch.device("cpu"))
        separate = keyed_noise(["b:5000"], 64, 42, 0, torch.device("cpu"))
        self.assertTrue(torch.equal(together[1], separate[0]))
        self.assertFalse(torch.equal(together, keyed_noise(["a:0", "b:5000"], 64, 42, 1, torch.device("cpu"))))

    def test_sampling_never_requires_target(self) -> None:
        model = tiny_model()
        data = inputs()
        first = sample(model, data, ["a:0", "b:0"], 42, 2, "heun", 2)
        second = sample(model, data, ["a:0", "b:0"], 42, 2, "heun", 2)
        self.assertEqual(first.shape, (2, 12, 64))
        self.assertTrue(torch.equal(first, second))
        self.assertNotIn("target", data)
        model.train()
        with self.assertRaises(ValueError):
            sample(model, data, ["a:0", "b:0"])

    def test_film_identity_and_observed_condition_effect(self) -> None:
        film = FiLM(16, 8)
        x, metadata = torch.randn(2, 8, 64), torch.randn(2, 16)
        self.assertTrue(torch.equal(x, film(x, metadata)))
        encoder = MetadataEncoder(16, True, 0.).eval()
        data = inputs()
        missing = torch.zeros_like(data["field_mask"])
        first = encoder(data["numeric"], data["sex"], missing, data["age_topcoded"])
        second = encoder(torch.full_like(data["numeric"], float("nan")), torch.ones_like(data["sex"]), missing,
                         torch.full_like(data["age_topcoded"], float("nan")))
        self.assertTrue(torch.equal(first, second))
        observed = encoder(data["numeric"] + .3, data["sex"], data["field_mask"], data["age_topcoded"])
        self.assertFalse(torch.equal(first, observed))

    def test_loss_gradients_without_any_optimizer_or_training_run(self) -> None:
        model = tiny_model().train()
        data = inputs()
        data["target"] = torch.randn(2, 12, 64)
        data["target"][:, :1] = data["anchor"]
        data["quality_mask"] = torch.ones(2, 12, dtype=torch.bool)
        data["quality_mask"][:, 7] = False
        data["target"][:, 7] = float("nan")
        weights = {"huber": .1, "pcc": .1, "anchor": .02, "physiology": 0., "huber_delta": 1.}
        loss, parts = flow_loss(model, data, weights, torch.ones(12), time=torch.tensor([.25, .75]))
        before = model.output.weight.detach().clone()
        loss.backward()  # A derivative check only: never create an optimizer or update parameters.
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"fm", "huber", "pcc", "anchor", "physiology"})
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(before, model.output.weight))
        self.assertTrue(torch.isfinite(model.metadata_encoder.mlp[0].weight).all())


class DataAndCheckpointTests(unittest.TestCase):
    def test_demographics_conflicts_missing_and_public_gender(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "userinfo.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["externalid", "age", "gender", "height", "weight"])
                writer.writeheader()
                writer.writerows([{"externalid": "HNU1", "age": 24, "gender": "男", "height": 175, "weight": 70},
                                  {"externalid": "HNU1", "age": 25, "gender": "男", "height": 175, "weight": 70}])
            table = DemographicsTable(path)
            value = table.get(" hnu1 ")
            self.assertEqual(value["field_mask"].tolist(), [False, True, True, True])
            self.assertEqual(table.get("unknown")["field_mask"].sum(), 0)
            self.assertEqual(table.audit()["conflicting_subjects"], 1)
        public = encode_demographics({"age": 302, "sex": 0}, "ptbxl")
        self.assertEqual(int(public["sex"]), 1)
        self.assertAlmostEqual(float(public["numeric"][0]), .9)
        self.assertEqual(float(public["age_topcoded"][0]), 1.)

    def test_raw_voltage_round_trip_and_no_median_subtraction(self) -> None:
        processor = preprocessor()
        raw = np.arange(12 * 64, dtype=np.float32).reshape(12, 64) + 900
        transformed = processor.transform_d12_target(raw, np.median(raw, axis=1))
        self.assertTrue(np.array_equal(transformed.baseline_uV, np.zeros(12)))
        np.testing.assert_allclose(processor.d12_model_view_to_raw_uV(transformed.model_signal), raw, rtol=1e-6)
        np.testing.assert_allclose(processor.transform_window(raw[:1], "ecg_machine_i").model_signal,
                                   transformed.model_signal[:1])

    def test_units_and_avr_avl_reordering(self) -> None:
        names = ["I", "II", "III", "AVL", "AVR", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        raw = np.tile(np.arange(1, 13, dtype=float), (5000, 1))
        converted = physical_to_uV(raw, ["mV"] * 12, names, 500)
        self.assertEqual(converted[3, 0], 5000.)
        self.assertEqual(converted[4, 0], 4000.)
        with self.assertRaises(ValueError):
            physical_to_uV(raw, ["mV"] * 12, names, 100)
        with self.assertRaises(ValueError):
            lead_order(names[:-1] + ["I"])

    def test_public_real_wfdb_fixture_split_and_partial_download(self) -> None:
        import wfdb
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "records500" / "00000"
            directory.mkdir(parents=True)
            rows = []
            wave = np.sin(np.arange(5000) * .02)[:, None] * .02 + np.arange(12)[None, :] * .003
            names = ["I", "II", "III", "AVL", "AVR", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            for index, fold in enumerate((1, 9, 10), 1):
                name = f"{index:05d}_hr"
                wfdb.wrsamp(name, fs=500, units=["mV"] * 12, sig_name=names, p_signal=wave,
                            fmt=["16"] * 12, adc_gain=[1000.] * 12, baseline=[100] * 12, write_dir=str(directory))
                rows.append({"ecg_id": str(index), "patient_id": str(index), "strat_fold": str(fold),
                             "filename_hr": f"records500/00000/{name}", "age": "302", "sex": "0"})
            path = root / "ptbxl_database.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(locate_ptbxl(root), root)
            processor = preprocessor()
            dataset = PTBXLDataset(root, "train", processor)
            item = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(item["target"].shape, (12, 5000))
            expected_avr = wave[:, 4] * 1000
            np.testing.assert_allclose(item["target_uV"][3], expected_avr, atol=.51)
            self.assertTrue(item["quality_mask"].all())
            directory.joinpath("00001_hr.dat").unlink()
            with self.assertRaises(FileNotFoundError):
                PTBXLDataset(root, "train", processor)
            rows[1]["patient_id"] = "1"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                PTBXLDataset(root, "validation", processor, check_files=False)

    def test_checkpoint_safe_load_scale_and_architecture_rejection(self) -> None:
        model, processor = tiny_model(), preprocessor()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.pt"
            payload = checkpoint(model, processor)
            atomic_save(path, payload)
            loaded = load_checkpoint(path)
            restored = model_from_checkpoint(loaded, torch.device("cpu"))
            self.assertTrue(torch.equal(model.output.weight, restored.output.weight))
            rebuilt = checkpoint_preprocessor(loaded, processor.config)
            np.testing.assert_array_equal(rebuilt.scale_uV_by_source["d12"], processor.scale_uV_by_source["d12"])
            payload["scales"]["d12"][0] += 1
            atomic_save(path, payload)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_tail_preserved_batch_independent_and_rate_conversion(self) -> None:
        model, processor = tiny_model(), preprocessor()
        anchor = np.linspace(-300, 400, 5031, dtype=np.float32)[None, :]
        demo = encode_demographics({"age": 24, "gender": "男", "height": 175, "weight": 70})
        settings = {"seed": 42, "steps": 1, "solver": "euler", "samples": 1}
        one = predict_record(model, processor, anchor, demo, "visible-record", torch.device("cpu"), settings, 1)
        two = predict_record(model, processor, anchor, demo, "visible-record", torch.device("cpu"), settings, 2)
        self.assertEqual(one.shape, (12, 5031))
        np.testing.assert_allclose(one, two, rtol=1e-5, atol=1e-3)
        canonical, length = canonical_anchor(np.linspace(-.3, .4, 10001), 1000, "mV")
        self.assertEqual(canonical.shape, (1, 5001))
        restored = restore_rate(np.tile(canonical, (12, 1)), length, 1000, 1000)
        self.assertEqual(restored.shape, (12, 10001))

    def test_main_task2_bonus_is_chest_only_without_pcc_preprocessing(self) -> None:
        t = np.arange(5000, dtype=np.float32)
        target = np.tile(np.sin(t * .01), (1, 12, 1)) * 100
        prediction = target.copy()
        prediction[:, 1:6] += 300
        rows = [{"pair_id": "pair", "target_record_id": "target", "start_sample_500hz": "0", "expected_window_count": "1"}]
        summary, _ = evaluate_record_predictions(prediction, target, "task2", rows)
        self.assertAlmostEqual(summary["r_missing11"], 1., places=6)
        self.assertAlmostEqual(summary["task2_missing_lead_mean_rmse_uV"], 0.)
        self.assertGreater(summary["missing11_mean_rmse_uV"], 0.)

    def test_all_experiment_configs_are_loadable(self) -> None:
        for path in sorted((ROOT / "configs" / "experiments").glob("b5_*.yaml")):
            config = load_config(path)
            self.assertEqual(len(config["model"]["channels"]), 3)
            self.assertIn(config["stage"], {"local", "public", "finetune"})

    def test_training_cli_requires_explicit_opt_in_before_loading_config(self) -> None:
        for module in ("train_public", "train_huawei", "overfit"):
            completed = subprocess.run([sys.executable, "-m", "baselines.B5." + module,
                                        "--config", "a_nonexistent_config.yaml"], cwd=ROOT,
                                       text=True, capture_output=True, timeout=20)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("Training was not started", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)

    def test_rng_checkpoint_roundtrip_without_parameter_updates(self) -> None:
        state = pack_rng()
        expected = (random.random(), np.random.random(3), torch.rand(3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rng.pt"
            atomic_save(path, {"rng": state})
            value = torch.load(path, weights_only=True)
            restore_rng(value["rng"])
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.random(3), expected[1])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))


class RealHuaweiReadOnlyTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("B5_REAL_HW_ROOT"), "Set B5_REAL_HW_ROOT for real-cache read-only integration checks")
    def test_real_caches_scales_and_validation_contract(self) -> None:
        hw = Path(os.environ["B5_REAL_HW_ROOT"])
        config = load_config(ROOT / "configs" / "experiments" / "b5_local_meta.yaml")
        config["paths"]["huawei_data_root"] = str(hw)
        config["paths"]["data_root"] = str(hw / "Data")
        processor, report = fit_huawei_scales(config)
        table = DemographicsTable(hw / "Data" / "userinfobean.csv")
        dataset = HuaweiTrainDataset(config, processor, table)
        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), report["train_windows"])
        item = dataset[0]
        raw = dataset.base[0].Y_12lead
        np.testing.assert_allclose(item["target"] * processor.scale_uV_by_source["d12"][:, None], raw, rtol=1e-6, atol=.001)
        self.assertTrue(np.array_equal(item["anchor"], item["target"][:1]))
        train_subjects = {r["subject_id"] for r in dataset.rows}
        for task in ("task1", "task2"):
            validation = HuaweiValidationDataset(config, task, processor, table)
            self.assertFalse(train_subjects & {r["subject_id"] for r in validation.rows})
            targets, metadata = [], []
            for index in range(len(validation)):
                value = validation[index]
                targets.append(value["target_uV"])
                metadata.append(value["evaluation_metadata"])
            targets = np.stack(targets)
            summary, _ = evaluate_record_predictions(targets, targets, task, metadata)
            self.assertAlmostEqual(summary["r_missing11"], 1.)
            self.assertAlmostEqual(summary["missing11_mean_rmse_uV"], 0.)


if __name__ == "__main__":
    unittest.main()
