from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5.flow import keyed_noise
from baselines.B5.metadata import encode_demographics
from baselines.B5_best.combine import combine, slow_curve
from baselines.B5_fast import COMPONENT_PROTOCOL, PROTOCOL
from baselines.B5_fast.components import WaveComponent, compose, extract_fast, full_record_slow
from baselines.B5_fast.package import load, state_sha, validate_package
from baselines.B5_fast.predict import Predictor
from baselines.B5_fast.profile import FM_STATE_SHA, load_profile, write
from baselines.B5_fast.sampling import member_noise, sample_fast_fm

torch.set_num_threads(2)


class FastTests(unittest.TestCase):
    def test_original_filter_and_composition_are_exact(self):
        rng = np.random.default_rng(123)
        fm = rng.normal(size=(12, 60000)).astype(np.float32) * 500
        slow = rng.normal(size=(11, 60000)) * 100
        fast = WaveComponent("record", extract_fast(fm), "fast", PROTOCOL)
        new_slow = WaveComponent("record", slow, "slow", COMPONENT_PROTOCOL)
        np.testing.assert_array_equal(full_record_slow(fm[1:]), slow_curve(fm[1:]))
        np.testing.assert_array_equal(compose(fm[:1], fast), combine(fm, np.zeros_like(slow), 1.))
        np.testing.assert_array_equal(compose(fm[:1], fast, new_slow), combine(fm, slow, 1.))

    def test_filter_crosses_original_ten_second_seam(self):
        fm = np.zeros((12, 60000), np.float32)
        fm[1:, 5000:] = 100
        fast = extract_fast(fm)
        self.assertLess(fast[0, 4999], -40.)
        self.assertGreater(fast[0, 5000], 40.)
        np.testing.assert_allclose(extract_fast(np.full_like(fm, 27)), 0)

    def test_compose_keeps_observed_I_without_target(self):
        anchor = np.linspace(-100, 100, 60000, dtype=np.float32)[None]
        fast = WaveComponent("r", np.zeros((11, 60000)), "fast", PROTOCOL)
        np.testing.assert_array_equal(compose(anchor, fast)[0], anchor[0])

    def test_component_copies_input_and_rejects_other_record(self):
        wave = np.zeros((11, 60000))
        fast = WaveComponent("r", wave, "fast", PROTOCOL)
        wave += 1
        self.assertFalse(fast.values_uV.flags.writeable)
        self.assertEqual(fast.values_uV.max(), 0)
        slow = WaveComponent("different", wave, "slow", COMPONENT_PROTOCOL)
        with self.assertRaises(ValueError):
            compose(np.zeros((1, 60000)), fast, slow)

    def test_invalid_component_contract_is_rejected(self):
        wave = np.zeros((11, 60000))
        for extra in ({"sampling_rate_hz": 50}, {"start_sample": 5000},
                      {"lead_order": tuple(reversed(load_profile()["slow_interface"]["lead_order"]))}):
            with self.assertRaises(ValueError):
                WaveComponent("r", wave, "slow", COMPONENT_PROTOCOL, **extra)
        with self.assertRaises(ValueError):
            WaveComponent("r", wave[:, :5000], "fast", PROTOCOL)
        wave[0, 0] = np.nan
        with self.assertRaises(ValueError):
            WaveComponent("r", wave, "fast", PROTOCOL)

    def test_antithetic_pairs_keep_legacy_seed_and_unit_scale(self):
        keys = ["a:0", "b:5000"]
        for pair in range(8):
            positive = member_noise(keys, 80, pair * 2, torch.device("cpu"))
            negative = member_noise(keys, 80, pair * 2 + 1, torch.device("cpu"))
            self.assertTrue(torch.equal(positive, -negative))
            self.assertTrue(torch.equal(positive, keyed_noise(keys, 80, 42, pair, torch.device("cpu"))))
            reversed_batch = member_noise(list(reversed(keys)), 80, pair * 2, torch.device("cpu"))
            self.assertTrue(torch.equal(positive[0], reversed_batch[1]))
        with self.assertRaises(ValueError):
            member_noise(keys, 80, 16, torch.device("cpu"))

    def test_sampler_passes_clean_I_and_averages_all_sixteen(self):
        anchor = torch.linspace(-1, 1, 5000).reshape(1, 1, 5000)
        before = anchor.clone()
        condition = SimpleNamespace(anchor_prediction=anchor * 0)
        model = SimpleNamespace(training=False, encode_conditions=lambda *args: condition)
        meta = encode_demographics({})
        inputs = {"anchor": anchor, **{k: torch.from_numpy(v[None].copy()) for k, v in meta.items()}}
        captured = []

        def ode(m, c, z, steps, solver):
            self.assertIs(c, condition)
            self.assertEqual((steps, solver), (16, "heun"))
            captured.append(z.clone())
            return z.square()

        with patch("baselines.B5_fast.sampling.integrate", side_effect=ode):
            output = sample_fast_fm(model, inputs, ["r:0"])
        self.assertEqual(len(captured), 16)
        self.assertTrue(torch.equal(anchor, before))
        for i in range(0, 16, 2):
            self.assertTrue(torch.equal(captured[i], -captured[i + 1]))
        expected = torch.zeros_like(captured[0])
        for z in captured:
            expected += z.square()
        self.assertTrue(torch.equal(output[:, 1:], expected / 16))

    def test_predict_stitches_twelve_windows_before_filter(self):
        engine = Predictor.__new__(Predictor)
        engine.device = torch.device("cpu")
        engine.scale = np.ones(12, np.float32)
        engine.fm = SimpleNamespace(training=False, parameters=lambda: [])
        raw = np.arange(60000, dtype=np.float32)[None]
        seen = []

        def sampler(model, batch, keys):
            seen.extend(keys)
            return batch["anchor"].repeat(1, 12, 1)

        with patch("baselines.B5_fast.predict.sample_fast_fm", side_effect=sampler):
            result = engine.predict(raw, {}, "rec", 5)
        expected = np.repeat(raw, 12, axis=0)
        np.testing.assert_array_equal(result.values_uV, extract_fast(expected))
        self.assertEqual(seen, ["rec:" + str(s) for s in range(0, 60000, 5000)])

    def test_predict_refuses_unfrozen_or_training_model(self):
        engine = Predictor.__new__(Predictor)
        engine.fm = SimpleNamespace(training=True)
        with self.assertRaises(RuntimeError):
            engine.predict(np.zeros((1, 60000)), {}, "rec")

    def test_profile_cannot_silently_change_sampling_or_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            value = load_profile()
            value["sampling"]["noise_scale"] = .5
            write(path, value)
            with self.assertRaises(ValueError):
                load_profile(path)
            value = load_profile()
            value["checkpoint"] = "/another/path/fm_only.pt"
            write(path, value)
            self.assertEqual(load_profile(path)["checkpoint"], value["checkpoint"])


@unittest.skipUnless(os.environ.get("B5_FAST_TEST_PACKAGE"), "Set B5_FAST_TEST_PACKAGE for selected real EMA checks")
class RealPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ["B5_FAST_TEST_PACKAGE"])
        cls.payload = load(cls.path)

    def test_real_package_has_only_selected_fm_and_frozen_parameters(self):
        self.assertEqual(set(self.payload), {"protocol", "contract", "fm", "provenance"})
        self.assertEqual(state_sha(self.payload["fm"]["ema"]), FM_STATE_SHA)
        engine = Predictor(self.path)
        self.assertFalse(engine.fm.training)
        self.assertTrue(all(not p.requires_grad for p in engine.fm.parameters()))

    def test_package_rejects_extra_network_or_changed_ema(self):
        bad = dict(self.payload, slow={})
        with self.assertRaises(ValueError):
            validate_package(bad, load_profile())
        bad = copy.deepcopy(self.payload)
        next(iter(bad["fm"]["ema"].values())).view(-1)[0] += 1
        with self.assertRaises(ValueError):
            validate_package(bad, load_profile())

    def test_package_rejects_changed_scale_even_with_new_digest(self):
        from baselines.B5.checkpoint import scales_digest
        bad = copy.deepcopy(self.payload)
        bad["fm"]["scales"]["d12"][1] *= 2
        bad["fm"]["scales_sha256"] = scales_digest(bad["fm"]["scales"])
        with self.assertRaises(ValueError):
            validate_package(bad, load_profile())


if __name__ == "__main__":
    unittest.main()
