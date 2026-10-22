import tempfile
import unittest
from pathlib import Path

import torch

from baselines.B5_best.common import write
from baselines.B5_final.g120_c3 import (
    C3_PROTOCOLS,
    _checkpoint_payload,
    _slow_record_indices,
    comparison_rows,
)


class G120C3Tests(unittest.TestCase):
    def test_slow_records_are_indexed_without_ambiguous_sources(self):
        study = {
            "validation": [
                {"task": "task1", "pair_id": "p1"},
                {"task": "task2", "pair_id": "p1"},
            ],
            "train": [{"target_record_id": "t1"}, {"target_record_id": "t2"}],
        }
        validation, training = _slow_record_indices(study)
        self.assertEqual(set(validation), {("task1", "p1"), ("task2", "p1")})
        self.assertEqual(set(training), {"t1", "t2"})
        with self.assertRaises(ValueError):
            _slow_record_indices({"validation": study["validation"] * 2, "train": study["train"]})
        with self.assertRaises(ValueError):
            _slow_record_indices({"validation": study["validation"], "train": study["train"] * 2})

    def test_only_supported_c3_payloads_are_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i, protocol in enumerate(sorted(C3_PROTOCOLS)):
                path = root / f"valid_{i}.pt"
                torch.save({"protocol": protocol, "model": {"weight": torch.ones(1)}}, path)
                self.assertEqual(_checkpoint_payload(path)["protocol"], protocol)
            bad = root / "bad.pt"
            torch.save({"protocol": "unrelated", "model": {}}, bad)
            with self.assertRaises(ValueError):
                _checkpoint_payload(bad)

    def test_three_arm_summary_uses_complete_mean_r(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "group_A_g120_original_body").mkdir()
            (run / "group_B_g120_old_c3").mkdir()
            (run / "group_C_train/selected_best").mkdir(parents=True)
            common = {"task1_r": .70, "machine_r": .71, "task2_r": .70}
            write(run / "group_A_g120_original_body/evaluation.json",
                  {"summary": {**common, "body_r": .69, "mean_r": .70}})
            write(run / "group_B_g120_old_c3/score.json",
                  {**common, "body_r": .70, "mean_r": .7025})
            write(run / "group_C_train/selected_best/score.json",
                  {**common, "body_r": .71, "mean_r": .705})
            rows = comparison_rows(run)
            self.assertEqual([row["group"] for row in rows],
                ["A_G120_original_body", "B_G120_old_C3", "C_G120_new_C3"])
            self.assertAlmostEqual(rows[2]["delta_mean_r_vs_A"], .005)


if __name__ == "__main__":
    unittest.main()
