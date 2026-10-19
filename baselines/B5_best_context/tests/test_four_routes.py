"""Frozen-body transfer guards and actual four-route QC scoring, without torch."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.B5_best.common import sha, read
from baselines.B5_best_context.four_routes import check_contract, cases, SAMPLING
from baselines.B5_best_context.body_features import BODY_PROTOCOL, BEAT16_ENCODING
from baselines.B5_best_context.body_alpha_core import SharedCorrection, FixedAlphaAdapter
from baselines.B5_best_context.body_study import run_study
from .test_context import StudyTests


class FourRouteTests(unittest.TestCase):
    def fixture(self):
        population = {t: {k: t + k for k in ("metadata_sha256", "target_sha256", "QC_sha256")}
                      for t in ("task1", "task2")}
        frozen_population = {t: {**v, "prediction_sha256": "old_prediction"} for t, v in population.items()}
        cache = {"frozen": {"base_checkpoint_sha256": "old_base", "sampling": SAMPLING,
                            "slow_beta": .75, "baseline_population": frozen_population}}
        payload = dict(protocol=BODY_PROTOCOL, group="context", feature_encoding=BEAT16_ENCODING,
                       training_seed=42, epoch=27, base_checkpoint_sha256="old_base",
                       cache_sha256="cache", sampling=SAMPLING, slow_beta=.75)
        baseline = dict(checkpoint_sha256="new_P_base", sampling=SAMPLING, slow_beta=.75)
        return payload, cache, baseline, population

    def test_transfer_requires_explicit_flag_and_never_relabels_training_base(self):
        payload, cache, base, pop = self.fixture()
        with self.assertRaisesRegex(ValueError, "allow-base-transfer"):
            check_contract(payload, cache, "cache", base, pop, allow_base_transfer=False)
        before = deepcopy(payload)
        result = check_contract(payload, cache, "cache", base, pop, allow_base_transfer=True)
        self.assertTrue(result["frozen_body_transfer"])
        self.assertEqual(result["body_training_base_sha256"], "old_base")
        self.assertEqual(result["evaluated_base_sha256"], "new_P_base")
        self.assertEqual(payload, before)
        self.assertFalse(check_contract(payload, cache, "cache",
            {**base, "checkpoint_sha256": "old_base"}, pop, allow_base_transfer=False)["frozen_body_transfer"])

    def test_transfer_does_not_allow_different_QC_encoding_sampling_or_selected_weight(self):
        payload, cache, base, pop = self.fixture()
        for key, value in (("epoch", 20), ("training_seed", 43), ("feature_encoding", "shape_amplitude_v1"),
                           ("cache_sha256", "wrong_cache"), ("base_checkpoint_sha256", "wrong_source"),
                           ("slow_beta", 1.)):
            with self.assertRaises(ValueError):
                check_contract({**payload, key: value}, cache, "cache", base, pop, allow_base_transfer=True)
        for key in ("metadata_sha256", "target_sha256", "QC_sha256"):
            changed = deepcopy(pop)
            changed["task2"][key] = "different"
            with self.assertRaisesRegex(ValueError, "Changed validation"):
                check_contract(payload, cache, "cache", base, changed, allow_base_transfer=True)
        with self.assertRaises(ValueError):
            check_contract(payload, cache, "cache", {**base, "sampling": {**SAMPLING, "seed": 43}},
                           pop, allow_base_transfer=True)

    def test_four_routes_use_same_prediction_and_independent_device_effects(self):
        class Engine:
            def __call__(self, prediction, observed_i, history):
                prediction[1] += .03 * observed_i[0]
                return prediction, {"synthetic": True}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            StudyTests().baseline(root / "base")
            original = sha(root / "base/task2/prediction_uV.npy")
            shared = SharedCorrection(Engine())
            rows = run_study(root / "base", root / "study", lambda row: None, None, None,
                case_specs=cases(), adapter_map={"body": FixedAlphaAdapter(shared, 1.)},
                notes=["Synthetic actual four-route scoring"])
            self.assertEqual([r["group"] for r in rows], ["baseline", "d6_only", "body_only", "both"])
            groups = {r["group"]: r for r in rows}
            for row in rows:
                self.assertEqual(row["task1_r"], rows[0]["task1_r"])
            self.assertEqual(groups["both"]["body_r"], groups["body_only"]["body_r"])
            self.assertEqual(groups["both"]["machine_r"], groups["d6_only"]["machine_r"])
            self.assertEqual(groups["body_only"]["machine_r"], groups["baseline"]["machine_r"])
            self.assertEqual(groups["d6_only"]["body_r"], groups["baseline"]["body_r"])
            self.assertAlmostEqual(groups["both"]["mean_r"] - groups["d6_only"]["mean_r"],
                                   groups["body_only"]["mean_r"] - groups["baseline"]["mean_r"])
            self.assertEqual(shared.forward_calls, 1)  # Identical visible synthetic inputs shared by both groups.
            for group in groups:
                report = read(root / f"study/{group}/evaluation.json")
                for task in ("task1", "task2"):
                    self.assertEqual(report["invariance"][task]["forbidden_lead_or_route_max_abs_uV"], 0.)
            self.assertEqual(sha(root / "base/task2/prediction_uV.npy"), original)


if __name__ == "__main__":
    unittest.main()
