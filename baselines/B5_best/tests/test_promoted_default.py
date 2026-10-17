"""Default promotion and byte-preserving deployment, without training/real data."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5_best import LEGACY_BUNDLE_PROTOCOL, SAMPLING
from baselines.B5_best import bundle as packaging
from baselines.B5_best import predict, validate
from baselines.B5_best.common import read, write, sha
from baselines.B5_best.tests.test_hybrid import fixture


def promoted_fixture(root):
    fm, slow, prepared, legacy = fixture(root)
    source = root / "verified.pt"
    bundle = packaging.export_bundle(fm, slow, prepared, source, legacy)
    profile = {"bundle_protocol": bundle["bundle_protocol"], "slow_beta": .75,
               "sampling": dict(SAMPLING), "default_bundle": str(source),
               "default_bundle_sha256": sha(source), "reference_qc_r": bundle["reference_qc_r"],
               "reference_tolerance": 2e-5, "components": {
                   "fm_sha256": bundle["provenance"]["fm_sha256"],
                   "slow_sha256": bundle["provenance"]["slow_sha256"]}}
    path = root / "profile.json"; write(path, profile)
    return source, bundle, path, profile, (fm, slow, prepared), legacy


class PromotedDefaultTests(unittest.TestCase):
    def test_current_profile_pins_the_verified_control100_PCC20_package(self):
        cfg = read(packaging.DEFAULT_PROFILE)
        self.assertTrue(cfg["default_bundle"].endswith("seed42_20261017_094115_417122/control100__pcc20.pt"))
        self.assertEqual(cfg["default_bundle_sha256"],
                         "a8d32cd598f73a0d3335ff4453da9168eccbc518a9b508ae81fdb8babcce254b")
        self.assertEqual(cfg["sampling"], SAMPLING)
        self.assertEqual(cfg["slow_beta"], .75)
        self.assertAlmostEqual(sum(cfg["reference_qc_r"].values()) / 2, .6982156358198193)
        old = read(packaging.DEFAULT_PROFILE.parent / cfg["legacy_diagnostics_profile"])
        self.assertEqual(old["fm_sha256"], "11dfcb8de1b90bfe5b33b320c4a243b3693d329e04c44ccb9cb9221d8cc70ca7")

    def test_missing_or_changed_verified_package_never_falls_back_to_raw_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, bundle, profile_path, profile, _, _ = promoted_fixture(Path(tmp))
            for field, value in (("default_bundle_sha256", "wrong"), ("slow_beta", 1.),
                                 ("sampling", dict(SAMPLING, samples=1)),
                                 ("reference_qc_r", {"task1_r_missing11": .1, "task2_r_missing11": .2}),
                                 ("components", {"fm_sha256": "wrong"})):
                cfg = copy.deepcopy(profile); cfg[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    packaging.verified_source(cfg)
            source.unlink()
            with patch.object(packaging, "load_checkpoint", side_effect=AssertionError("No raw fallback")):
                with self.assertRaises(FileNotFoundError):
                    packaging.load_inference_bundle(None, profile_path)

    def test_stable_copy_is_byte_identical_and_works_without_original_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bundle, profile_path, profile, raw, _ = promoted_fixture(root)
            for path in raw:
                path.unlink()
            destination = root / "stable/model.pt"
            with patch.object(packaging, "export_bundle", side_effect=AssertionError("Do not rebuild weights")):
                packaging.export_profile(profile_path, destination)
            self.assertEqual(sha(destination), profile["default_bundle_sha256"])
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            source.unlink()
            loaded, path = packaging.load_inference_bundle(destination, root / "missing_profile.json")
            self.assertEqual(path, destination.resolve())
            engine = predict.Predictor(loaded, torch.device("cpu"))
            with patch.object(predict, "sample", side_effect=lambda model, x, keys, **kw:
                              torch.zeros(len(keys), 12, 5000)), \
                 patch.object(predict, "predict_native", return_value=torch.ones(1, 11, 6000)):
                result = engine.predict(np.zeros(60000, np.float32), {}, "new_patient", 8)
            np.testing.assert_array_equal(result[1:], 75.)
            self.assertFalse(read(destination.with_suffix(".json"))["training_started"])
            with self.assertRaises(FileExistsError):
                packaging.export_profile(profile_path, destination)

    def test_explicit_legacy_bundle_still_uses_its_own_beta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, bundle, profile_path, _, _, _ = promoted_fixture(root)
            old = copy.deepcopy(bundle); old["bundle_protocol"] = LEGACY_BUNDLE_PROTOCOL
            old.pop("slow_beta")
            path = root / "old.pt"; torch.save(old, path)
            loaded, _ = packaging.load_inference_bundle(path, root / "missing.json")
            self.assertEqual(packaging.bundle_slow_beta(loaded), 1.)

    def test_default_predict_cli_loads_verified_profile_without_checkpoint_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bundle, profile_path, _, _, _ = promoted_fixture(root)
            anchor = root / "I.npy"; np.save(anchor, np.zeros(60000, np.float32))
            with patch("sys.argv", ["predict", "--profile", str(profile_path), "--anchor-npy", str(anchor),
                                     "--record-key", "new", "--output", str(root / "prediction"), "--device", "cpu"]), \
                 patch.object(predict, "sample", side_effect=lambda model, x, keys, **kw:
                              torch.zeros(len(keys), 12, 5000)), \
                 patch.object(predict, "predict_native", return_value=torch.ones(1, 11, 6000)):
                predict.main()
            report = read(root / "prediction/inference.json")
            self.assertEqual(report["checkpoint_sha256"], sha(source))
            self.assertEqual(report["slow_beta"], .75)
            self.assertEqual(report["sampling"], SAMPLING)
            self.assertFalse(report["target_used_at_inference"])

    def test_default_validate_cli_selects_correct_bundle_and_unchanged_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bundle, profile_path, _, _, _ = promoted_fixture(root)
            datasets = {t: SimpleNamespace(manifest_digest="fixture") for t in ("task1", "task2")}
            with patch("sys.argv", ["validate", "--profile", str(profile_path), "--config", "data.yaml",
                                     "--output-dir", str(root / "evaluation"), "--device", "cpu"]), \
                 patch.object(validate, "load_config", return_value={"model": bundle["fm"]["model_config"]}), \
                 patch.object(validate, "preprocessing_config"), patch.object(validate, "checkpoint_preprocessor"), \
                 patch.object(validate, "build_validation_datasets", return_value=datasets), \
                 patch.object(validate, "evaluate") as evaluate:
                validate.main()
            args = evaluate.call_args.args
            self.assertEqual(args[0]["fm_state_sha256"], bundle["fm_state_sha256"])
            self.assertEqual(args[1], datasets)
            self.assertEqual(args[4], sha(source))

    def test_legacy_profile_export_remains_available_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, raw, legacy = promoted_fixture(root)
            legacy.update(fm_checkpoint=str(raw[0]), slow_checkpoint=str(raw[1]), prepared_info=str(raw[2]))
            path = root / "legacy.json"; write(path, legacy)
            value = packaging.export_profile(path, root / "old_copy.pt")
            self.assertEqual(value["reference_qc_r"], legacy["reference_qc_r"])


if __name__ == "__main__":
    unittest.main()
