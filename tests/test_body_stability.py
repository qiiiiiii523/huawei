import unittest
import numpy as np
from baselines.B5_final.body_stability import paired_bootstrap


class StabilityTests(unittest.TestCase):
    def test_constant_paired_gain_respects_task_device_weight(self):
        reference = {(str(i), str(i)): np.full(11, .5) for i in range(4)}
        candidate = {k: v + .004 for k, v in reference.items()}
        result, samples = paired_bootstrap(reference, candidate, draws=100)
        self.assertAlmostEqual(result["delta_mean_r"], .001)
        np.testing.assert_allclose(samples, .001, atol=1e-12)
        self.assertEqual(result["subjects_improved"], 4)

    def test_rejects_changed_qc_or_target_population(self):
        reference = {("a", "t"): np.full(11, .5)}
        candidate = {("a", "t"): np.full(11, .5)}
        candidate[("a", "t")][0] = np.nan
        with self.assertRaises(ValueError):
            paired_bootstrap(reference, candidate)
        with self.assertRaises(ValueError):
            paired_bootstrap(reference, {})

    def test_repeated_targets_stay_in_the_same_subject_cluster(self):
        reference = {("a", "t1"): np.full(11, .5), ("a", "t2"): np.full(11, .5),
                     ("b", "t3"): np.full(11, .5)}
        candidate = {k: v + (.004 if k[0] == "a" else -.004) for k, v in reference.items()}
        result, samples = paired_bootstrap(reference, candidate, draws=100)
        self.assertEqual(result["subjects"], 2)
        self.assertAlmostEqual(result["delta_mean_r"], .004 / 12)
        self.assertLess(samples.min(), 0)
        self.assertGreater(samples.max(), 0)


if __name__ == "__main__":
    unittest.main()
