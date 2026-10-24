"""M2/M3 graph, endpoint-only validation and recovery checks on CPU."""
from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from baselines.B7 import train as trainer, pipeline
from baselines.B7.checkpoint import load_checkpoint
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.model import B7UNet
from baselines.B7.tests.test_b7 import make_manifest, small_config
from scripts import run_b7_structures as runner


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, root):
        return Namespace(output=root / "suite", data_root=root, metadata=root, ptb_root=root,
            reuse_data_from=root / "cache", gpu="0", dry_run=False, max_train=2, max_peak_gib=40.)

    def test_config_budgets_and_exact_parameter_counts(self):
        for name, count in (("m2", 11477663), ("m3", 11136927)):
            models = []
            for stage in ("public10", "public40", "huawei40"):
                cfg = json.loads((runner.ROOT / "configs" / ("u7_" + name) / f"b7_{stage}.json").read_text())
                mc, lc, sc = trainer.validate_settings(cfg)
                models.append(mc)
                self.assertEqual(sum(p.numel() for p in B7UNet(mc).parameters()), count)
                tr = cfg["training"]
                self.assertEqual(tr["batch_size"], 64 if stage == "public10" else 32)
                self.assertEqual(tr["updates"], 10000 if stage == "public10" else 1000)
                self.assertEqual(tr["validation_mode"], "final_only")
                self.assertEqual(tr["checkpoint_every"], 500)
                self.assertEqual([s for s in range(1, tr["updates"] + 1) if trainer.validation_due(tr, s)], [tr["updates"]])
                self.assertEqual((sc.samples, sc.steps, sc.core_seconds), (16, 16, 10))
                self.assertEqual(sc.window_seconds, 10 if stage == "public10" else 40)
                self.assertEqual((lc.ii_weight, lc.mae_weight, lc.pcc_weight), (5, 1, .1))
            self.assertTrue(all(mc == models[0] for mc in models))

    def test_graph_has_independent_parents_and_reuses_only_data(self):
        args = self.args(Path("/example"))
        jobs = runner.suite_jobs(args)
        self.assertEqual([j.name for j in jobs[:2]], ["M2_P", "M3_P"])
        self.assertEqual(len(jobs), 8)
        for job in jobs:
            candidate, stage = job.name.split("_")
            self.assertEqual(job.module, "train")
            self.assertEqual(job.gpu, "0")
            expected = () if stage == "P" else (candidate + ("_Q" if stage == "B" else "_P"),)
            self.assertEqual(job.dependencies, expected)
            self.assertIn(candidate + "_seed42", str(job.artifact))
            if stage == "P":
                self.assertNotIn("--init-checkpoint", job.arguments)
            else:
                parent = job.arguments[job.arguments.index("--init-checkpoint") + 1]
                self.assertIn(candidate + "_seed42", parent)

    def test_memory_budget_serializes_or_blocks(self):
        self.assertEqual(runner.choose_parallelism([32., 34.], 94., 2), 2)
        self.assertEqual(runner.choose_parallelism([32., 34.], 70., 2), 1)
        self.assertEqual(runner.choose_parallelism([32.], 94., 1), 1)
        with self.assertRaises(RuntimeError):
            runner.choose_parallelism([39.], 25., 2)

    def test_default_run_skips_probes_without_memory_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            manifest = {"scales_uV": [100.] * 12, "records": [{"resolved_path": str(make_manifest(root))}]}
            with patch("baselines.B7.data.read_manifest", return_value=manifest), \
                 patch.object(runner, "checked_subprocess", side_effect=AssertionError("Probe must not run")), \
                 patch.object(pipeline, "run", return_value=0) as execute:
                self.assertEqual(runner.run(args), 0)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(args.max_train, 2)
            plan = json.loads((args.output / "gpu_plan.json").read_text())
            self.assertFalse(plan["gpu_probe"])
            self.assertIsNone(plan["memory_limit_GiB"])

    def test_combined_scheduler_global_cap_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_manifest(root)
            args = self.args(root)
            args.config_dir = runner.ROOT / "configs"
            args.experiment_suite = True
            args.retry_preparation = False
            jobs = runner.suite_jobs(args)
            for job in jobs:
                job.arguments[job.arguments.index("--manifest") + 1] = str(manifest)
            running, maximum, started, finished = set(), [0], [], set()
            class Process:
                def __init__(self, command, **kwargs):
                    directory = command[command.index("--output") + 1]
                    self.name = next(j.name for j in jobs if str(j.artifact.parent) == directory)
                    job = next(j for j in jobs if j.name == self.name)
                    assert all(dep in finished for dep in job.dependencies)
                    running.add(self.name)
                    maximum[0] = max(maximum[0], len(running))
                    started.append(self.name)
                def poll(self):
                    running.discard(self.name)
                    finished.add(self.name)
                    return 1 if self.name == "M2_Q" else 0
            original_cwd = Path.cwd()
            try:
                with patch.object(torch.cuda, "is_available", return_value=True), \
                     patch.object(pipeline.subprocess, "Popen", Process), \
                     patch.object(pipeline, "validate_artifact", return_value={"best_r": .2}), \
                     patch.object(pipeline.time, "sleep"):
                    self.assertEqual(pipeline.run(args, jobs), 1)
            finally:
                import os
                os.chdir(original_cwd)
            self.assertEqual(maximum[0], 2)
            self.assertNotIn("M2_B", started)
            self.assertIn("M3_B", started)
            status = json.loads((args.output / "pipeline_status.json").read_text())
            self.assertEqual(status["jobs"]["M2_B"], "blocked")
            self.assertEqual(status["jobs"]["M3_B"], "done")

    def test_final_only_interrupted_validation_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_manifest(root)
            cfg = {"stage": "public10", "model": small_config().to_dict(), "loss": asdict(LossConfig()),
                   "sampling": asdict(SamplingConfig(window_seconds=10, samples=1, steps=1)),
                   "training": {"seed": 4, "updates": 3, "validate_every": 3, "batch_size": 1,
                     "learning_rate": .0001, "weight_decay": .01, "ema_decay": .9,
                     "gradient_clip": 1., "boundary_probability": 0., "checkpoint_every": 1,
                     "validation_mode": "final_only"}}
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            args = Namespace(config=config, manifest=manifest, output=root / "full", device="cpu",
                             resume=False, init_checkpoint=None)
            with patch.object(trainer, "evaluate", return_value=({"r": .2}, [])) as evaluate:
                trainer.train(args)
                self.assertEqual(evaluate.call_count, 1)
            args.output = root / "interrupted"
            with patch.object(trainer, "evaluate", side_effect=RuntimeError("interrupted")) as evaluate:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    trainer.train(args)
                self.assertEqual(evaluate.call_count, 1)
            saved = load_checkpoint(args.output / "last.pt")
            self.assertEqual(saved["update"], 3)
            self.assertTrue(saved["pending_validation"])
            self.assertEqual(saved["last_validated_update"], -1)
            self.assertFalse((args.output / "completed.json").exists())
            args.resume = True
            with patch.object(trainer, "evaluate", return_value=({"r": .2}, [])) as evaluate:
                trainer.train(args)
                self.assertEqual(evaluate.call_count, 1)
            a = load_checkpoint(root / "full/best_r.pt")
            b = load_checkpoint(args.output / "best_r.pt")
            for key in a["ema"]:
                self.assertTrue(torch.equal(a["ema"][key], b["ema"][key]))
            self.assertEqual(b["update"], 3)
            self.assertEqual(json.loads((args.output / "completed.json").read_text())["selection"], "final_ema")
            self.assertEqual(len((args.output / "history.jsonl").read_text().splitlines()), 1)
            # A periodic save before any validation must preserve optimizer/RNG on resume.
            args.output = root / "mid_training"
            args.resume = False
            save = trainer.save_checkpoint
            def interrupt_after_save(path, value):
                save(path, value)
                if Path(path).name == "last.pt" and value["update"] == 1:
                    raise RuntimeError("mid-training interruption")
            with patch.object(trainer, "save_checkpoint", side_effect=interrupt_after_save), \
                 patch.object(trainer, "evaluate") as evaluate:
                with self.assertRaisesRegex(RuntimeError, "mid-training"):
                    trainer.train(args)
                self.assertEqual(evaluate.call_count, 0)
            mid = load_checkpoint(args.output / "last.pt")
            self.assertFalse(mid["pending_validation"])
            args.resume = True
            with patch.object(trainer, "evaluate", return_value=({"r": .2}, [])) as evaluate:
                trainer.train(args)
                self.assertEqual(evaluate.call_count, 1)
            recovered = load_checkpoint(args.output / "last.pt")
            full = load_checkpoint(root / "full/last.pt")
            for key in full["model"]:
                self.assertTrue(torch.equal(full["model"][key], recovered["model"][key]))
                self.assertTrue(torch.equal(full["ema"][key], recovered["ema"][key]))
            for index, state in full["optimizer"]["state"].items():
                for key, tensor in state.items():
                    self.assertTrue(torch.equal(tensor, recovered["optimizer"]["state"][index][key]))


if __name__ == "__main__":
    unittest.main()
