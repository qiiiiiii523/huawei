"""CPU-only checks of the independent configurations and experiment scheduler."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("large_runner", ROOT / "scripts/run_b7_large.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from baselines.B7.pipeline import plan, fingerprint


class LargeTests(unittest.TestCase):
    def args(self, output):
        return SimpleNamespace(data_root=ROOT, metadata=ROOT / "metadata", ptb_root=ROOT,
            reuse_data_from=ROOT / "existing_cache", output=output, gpu="0", dry_run=False)

    def test_matched_sample_and_validation_budgets(self):
        for stage in ("public10", "public40", "huawei40"):
            configs = [json.loads((ROOT / f"configs/u7_large_bs{b}/b7_{stage}.json").read_text()) for b in (32, 64)]
            a, b = [c["training"] for c in configs]
            self.assertEqual(configs[0]["model"], configs[1]["model"])
            self.assertEqual(configs[0]["model"]["channels"], [152, 304, 608])
            self.assertEqual(a["updates"] * 32, b["updates"] * 64)
            self.assertEqual(a["validate_every"] * 32, b["validate_every"] * 64)
            self.assertEqual(a["checkpoint_every"] * 32, b["checkpoint_every"] * 64)
            self.assertAlmostEqual(b["ema_decay"], a["ema_decay"] ** 2)
            self.assertEqual(a["learning_rate"], b["learning_rate"])
            self.assertEqual(configs[0]["sampling"], configs[1]["sampling"])
            self.assertEqual(configs[0]["sampling"]["samples"], 16)
            self.assertEqual(a["updates"] % a["validate_every"], 0)

    def test_launcher_plan_uses_independent_outputs_no_small_checkpoint(self):
        args = self.args(ROOT / "new_output")
        for batch in (32, 64):
            jobs = runner.commands(args, batch)
            self.assertEqual([n for n, _ in jobs], ["probe_public10", "probe_public40", "pipeline"])
            cmd = jobs[-1][1]
            self.assertIn(str(args.output / f"bs{batch}_seed42"), cmd)
            self.assertIn(str(ROOT / f"configs/u7_large_bs{batch}"), cmd)
            self.assertEqual(cmd[cmd.index("--max-train") + 1], "1")
            self.assertNotIn("--init-checkpoint", cmd)

    def test_pipeline_uses_custom_configs_and_parent_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            args.gpu_p = args.gpu_a = args.gpu_b = "0"
            args.config_dir = ROOT / "configs/u7_large_bs64"
            jobs = {j.name: j for j in plan(args)}
            self.assertEqual(jobs["A"].dependencies, ("P",))
            self.assertEqual(jobs["B"].dependencies, ("Q",))
            self.assertNotIn("--init-checkpoint", jobs["P"].arguments)
            for job in jobs.values():
                cfg = Path(job.arguments[job.arguments.index("--config") + 1])
                self.assertEqual(cfg.parent, args.config_dir)
                self.assertTrue(cfg.is_file())
            old = fingerprint(args)
            args.config_dir = ROOT / "configs/u7_large_bs32"
            self.assertNotEqual(fingerprint(args), old)

    def test_probe_failure_blocks_its_pipeline_but_other_experiment_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            events = []
            def execute(command, log, env):
                events.append(log.stem)
                return 1 if log.stem == "bs32_probe_public40" else 0
            with patch.object(runner, "execute", side_effect=execute):
                self.assertEqual(runner.run(args), 1)
            self.assertNotIn("bs32_pipeline", events)
            self.assertIn("bs64_pipeline", events)
            report = json.loads((args.output / "experiments_status.json").read_text())
            self.assertEqual(report["experiments"], {"bs32": "failed", "bs64": "done"})
            self.assertFalse(report["success"])

    def test_pipeline_failure_does_not_block_other_independent_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            with patch.object(runner, "execute", side_effect=[0, 0, 1, 0, 0, 0]):
                self.assertEqual(runner.run(args), 1)
            report = json.loads((args.output / "experiments_status.json").read_text())
            self.assertEqual(report["experiments"]["bs64"], "done")

    def test_success(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            with patch.object(runner, "execute", return_value=0):
                self.assertEqual(runner.run(args), 0)
            self.assertTrue(json.loads((args.output / "experiments_status.json").read_text())["success"])


if __name__ == "__main__":
    unittest.main()
