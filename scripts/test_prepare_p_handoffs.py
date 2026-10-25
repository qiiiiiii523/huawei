"""Check portable P handoffs without data, duplicate weights or architecture mismatch."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from scripts import prepare_p_handoffs as packer
from scripts.test_b7_a1 import parent_at


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_bundles_preserve_parent_and_validate_all_templates(self):
        for name, module in (("B7_M2_P", "baselines.B7"), ("B7T_g6_19m_P", "baselines.B7T_A1")):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parent = parent_at(module, root)
                original = parent.read_bytes()
                checkpoint = importlib.import_module(module + ".checkpoint")
                value = checkpoint.load_checkpoint(parent)
                target = root / name
                count = sum(p.numel() for p in checkpoint.model_from_checkpoint(value).parameters())
                info = packer.prepare_bundle(parent, name, target, (module, value, count))
                self.assertEqual(parent.read_bytes(), original)
                self.assertEqual((target / "P/best_r.pt").read_bytes(), original)
                self.assertEqual(list(target.rglob("*.pt")), [target / "P/best_r.pt"])
                self.assertFalse(list(target.rglob("*.npz")))
                self.assertFalse(info["data_included"])
                self.assertTrue((target / "P/completed.json").is_file())
                self.assertTrue((target / "scripts/run_b7_a1.py").is_file())
                self.assertTrue((target / Path(*module.split('.')) / "train.py").is_file())
                trainer = importlib.import_module(module + ".train")
                stages = set()
                for path in (target / "configs").glob("*.json"):
                    cfg = json.loads(path.read_text())
                    trainer.validate_settings(cfg)
                    stages.add(cfg["stage"])
                    self.assertEqual(cfg["training"]["updates"], 1000)
                    self.assertEqual(cfg["training"]["validate_every"], 1000)
                self.assertEqual(stages, {"huawei10", "huawei40", "public40"})
                from baselines.B7.checkpoint import file_sha256
                for line in (target / "SHA256SUMS").read_text().splitlines():
                    digest, relative = line.split("  ", 1)
                    self.assertEqual(file_sha256(target / relative), digest)
                # Import and load from the handoff alone, not this source checkout.
                code = """from pathlib import Path
from scripts.run_b7_a1 import read_parent, make_config, parse_args
m, p = read_parent(Path('P/best_r.pt'))
a = parse_args(['--parent-checkpoint', 'P/best_r.pt', '--output', 'runs/A1', '--gpu', '0'])
c = make_config(m, p, a)
assert c['stage'] == 'huawei10'
print(m)
"""
                probe = subprocess.run([sys.executable, "-c", code], cwd=target, env=os.environ.copy(),
                                       capture_output=True, text=True)
                self.assertEqual(probe.returncode, 0, probe.stderr)
                self.assertIn(module, probe.stdout)

    def test_wrong_size_and_nonfinal_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = parent_at("baselines.B7", root)
            with self.assertRaisesRegex(ValueError, "Wrong B7_M2_P architecture"):
                packer.inspect_parent(path, "B7_M2_P")
            from baselines.B7 import checkpoint
            value = checkpoint.load_checkpoint(path)
            count = sum(p.numel() for p in checkpoint.model_from_checkpoint(value).parameters())
            value["update"] = 1
            checkpoint.save_checkpoint(path, value)
            with patch.dict(packer.EXPECTED, {"B7_M2_P": ("baselines.B7", count)}):
                with self.assertRaisesRegex(ValueError, "not final step"):
                    packer.inspect_parent(path, "B7_M2_P")


if __name__ == "__main__":
    unittest.main()
