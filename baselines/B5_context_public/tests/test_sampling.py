import copy
import importlib.util
import unittest
from baselines.B5_context_public.sampling import SAMPLING, member_spec, validate_settings, validate_machine_references


class ContractTests(unittest.TestCase):
    def test_eight_pairs_no_amplitude_change(self):
        self.assertEqual([member_spec(i) for i in range(16)], [(j, s) for j in range(8) for s in (1, -1)])
        for bad in (-1, 16, True, 1.5):
            with self.assertRaises(ValueError): member_spec(bad)
        self.assertEqual(SAMPLING["noise_scale"], 1.)

    def test_old_noise_reports_rejected(self):
        report = dict(completed=True, route="ecg_machine_d6", full_record_seconds=120, slow_beta=.75,
                      sampling=dict(SAMPLING))
        for k in ("checkpoint_sha256", "population_sha256", "record_keys_sha256", "scales_sha256",
                  "slow_assets_sha256", "terminal_rules_sha256"):
            report[k] = "a" * 64
        reports = {k: copy.deepcopy(report) for k in ("public_initial", "stable_machine", "control_initial", "context_public_initial")}
        self.assertTrue(validate_machine_references(reports))
        reports["stable_machine"]["sampling"]["noise_mode"] = "independent"
        with self.assertRaises(ValueError): validate_machine_references(reports)
        reports["stable_machine"] = copy.deepcopy(report)
        reports["control_initial"]["record_keys_sha256"] = "b" * 64
        with self.assertRaises(ValueError): validate_machine_references(reports)

    def test_no_silent_settings_change(self):
        for field, value in (("samples", 8), ("noise_scale", .75), ("steps", 32), ("seed", 43)):
            with self.assertRaises(ValueError): validate_settings(dict(SAMPLING, **{field: value}))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required on server")
class TorchTests(unittest.TestCase):
    def test_actual_pair_negation_and_batch_order(self):
        import torch
        from baselines.B5_context_public.sampling import member_noise
        for index in range(0, 16, 2):
            plus = member_noise(["record:0", "other:0"], 16, index, "cpu", torch.float32)
            minus = member_noise(["record:0", "other:0"], 16, index + 1, "cpu", torch.float32)
            self.assertTrue(torch.equal(plus, -minus))
            single = member_noise(["other:0"], 16, index, "cpu", torch.float32)
            self.assertTrue(torch.equal(single[0], plus[1]))


if __name__ == "__main__": unittest.main()
