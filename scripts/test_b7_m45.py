"""CPU verification of M4/M5 configs, direct-I interfaces, scheduling and restart."""
from argparse import Namespace
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from baselines.B7 import pipeline, train as trainer
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.checkpoint import load_checkpoint
from baselines.B7.model import B7UNet
from baselines.B7.tests.test_b7 import make_manifest, small_config
from scripts import run_b7_m45 as runner


class M45Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, root, gpus=("0", "1")):
        return Namespace(output=root / "suite", data_root=root, metadata=root, ptb_root=root,
            reuse_data_from=root / "cache", models=["M4", "M5"], gpu_m4=gpus[0], gpu_m5=gpus[1],
            max_train=2, max_per_gpu=1, dry_run=False, config_dir=runner.ROOT / "configs",
            experiment_suite=True, gpu_is_runtime=True, recover_initial=True, retry_preparation=False)

    def test_all_configs_and_exact_parameters(self):
        for name, blocks, count in (("M4", 6, 14685087), ("M5", 8, 18233247)):
            models = []
            for stage in ("public10", "public40", "huawei40"):
                cfg = json.loads((runner.ROOT / f"configs/u7_{name.lower()}/b7_{stage}.json").read_text())
                mc, lc, sc = trainer.validate_settings(cfg)
                model = B7UNet(mc)
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                self.assertEqual(mc.channels, (128, 256, 512))
                self.assertEqual(len(model.bottleneck), blocks)
                self.assertEqual(mc.dilations, (1, 2, 4, 8, 16, 32) + ((1, 2) if blocks == 8 else ()))
                models.append(mc)
                tr = cfg["training"]
                end = 20000 if stage == "public10" else 1000
                self.assertEqual(tr["updates"], end)
                self.assertEqual(tr["validate_every"], end)
                self.assertEqual(tr["validation_mode"], "final_only")
                self.assertEqual(tr["checkpoint_every"], 500)
                self.assertEqual(tr["batch_size"], 64 if stage == "public10" else 32)
                self.assertEqual([i for i in range(1, end + 1) if trainer.validation_due(tr, i)], [end])
                self.assertEqual((sc.window_seconds, sc.core_seconds), (10 if stage == "public10" else 40, 10))
                self.assertEqual((sc.samples, sc.steps, sc.solver), (16, 16, "heun"))
                self.assertEqual((lc.mae_weight, lc.pcc_weight, lc.ii_weight), (1, .1, 5))
            self.assertTrue(all(mc == models[0] for mc in models))

    def test_both_models_40s_direct_clean_I_and_seven_outputs(self):
        for name in ("m4", "m5"):
            cfg = json.loads((runner.ROOT / f"configs/u7_{name}/b7_huawei40.json").read_text())
            mc, _, _ = trainer.validate_settings(cfg)
            model = B7UNet(mc).eval()
            anchor = torch.linspace(-1, 1, 20000).reshape(1, 1, -1)
            clean = anchor.clone()
            state = torch.randn(1, 7, 20000)
            observed = []
            hook = model.stem.register_forward_pre_hook(lambda _, a: observed.append(a[0].clone()))
            with torch.no_grad():
                result = model(state, torch.tensor([.5]), anchor)
            hook.remove()
            self.assertEqual(result.shape, (1, 7, 20000))
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(torch.equal(anchor, clean))
            self.assertTrue(torch.equal(observed[0][:, :1], clean))
            self.assertTrue(torch.equal(observed[0][:, 1:], state))

    def test_graph_gpu_and_initialization_lineages(self):
        args = self.args(Path("/example"))
        jobs = runner.suite_jobs(args)
        self.assertEqual([j.name for j in jobs[:2]], ["M4_P", "M5_P"])
        self.assertEqual(len(jobs), 8)
        for job in jobs:
            name, stage = job.name.split("_")
            self.assertEqual(job.module, "train")
            self.assertEqual(job.gpu, "0" if name == "M4" else "1")
            self.assertEqual(job.dependencies, () if stage == "P" else (name + ("_Q" if stage == "B" else "_P"),))
            if stage == "P":
                self.assertNotIn("--init-checkpoint", job.arguments)
            else:
                path = Path(job.arguments[job.arguments.index("--init-checkpoint") + 1])
                self.assertEqual(path.parent.name, "Q" if stage == "B" else "P")
                self.assertEqual(path.parent.parent.name, name + "_seed42")
        args.models = ["M5"]
        self.assertEqual(len(runner.suite_jobs(args)), 4)

    def test_gpu_selection_required_and_default_limits(self):
        with self.assertRaises(SystemExit):
            runner.parse_args([])
        with self.assertRaises(SystemExit):
            runner.parse_args(["--gpu-m4", "0", "--gpu-m5", "0,1"])
        args = runner.parse_args(["--gpu-m4", "0", "--gpu-m5", "1"])
        self.assertEqual((args.max_train, args.max_per_gpu), (2, 1))
        args = runner.parse_args(["--models", "M4", "--gpu-m4", "2"])
        self.assertEqual(args.models, ["M4"])

    def test_per_gpu_limit_on_same_card(self):
        self.scheduler(("1", "1"), fail=False, expected_peak=1)

    def test_two_cards_failure_isolation_and_gpu_relocation(self):
        self.scheduler(("0", "1"), fail=True, expected_peak=2)

    def scheduler(self, gpus, fail, expected_peak):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_manifest(root)
            args = self.args(root, gpus)
            jobs = runner.suite_jobs(args)
            for job in jobs:
                job.arguments[job.arguments.index("--manifest") + 1] = str(manifest)
            active, finished, events, peak = {}, set(), [], [0]
            class Process:
                def __init__(self, command, env, **kwargs):
                    out = command[command.index("--output") + 1]
                    self.job = next(j for j in jobs if str(j.artifact.parent) == out)
                    assert all(dep in finished for dep in self.job.dependencies)
                    assert sum(gpu == self.job.gpu for gpu in active.values()) < args.max_per_gpu
                    assert env["CUDA_VISIBLE_DEVICES"] == self.job.gpu
                    self.job.artifact.parent.mkdir(parents=True, exist_ok=True)
                    active[self.job.name] = self.job.gpu
                    peak[0] = max(peak[0], len(active))
                    events.append(self.job.name)
                def poll(self):
                    active.pop(self.job.name)
                    finished.add(self.job.name)
                    if fail and self.job.name == "M4_Q":
                        return 1
                    self.job.artifact.write_text('{}')
                    return 0
            cwd = Path.cwd()
            try:
                with patch.object(torch.cuda, "is_available", return_value=True), \
                     patch.object(pipeline.subprocess, "Popen", Process), \
                     patch.object(pipeline, "validate_artifact", return_value={"best_r": .3}), \
                     patch.object(pipeline.time, "sleep"):
                    self.assertEqual(pipeline.run(args, jobs), 1 if fail else 0)
                    self.assertEqual(peak[0], expected_peak)
                    status = json.loads((args.output / "pipeline_status.json").read_text())
                    self.assertEqual(status["jobs"]["M5_B"], "done")
                    self.assertEqual(status["jobs"]["M4_B"], "blocked" if fail else "done")
                    if not fail:
                        # A new physical GPU is a runtime choice: completed jobs stay reusable.
                        for job in jobs:
                            job.gpu = "2"
                        with patch.object(pipeline.subprocess, "Popen", side_effect=AssertionError("Completed jobs must not restart")):
                            self.assertEqual(pipeline.run(args, jobs), 0)
            finally:
                os.chdir(cwd)

    def test_failed_first_update_recovers_verified_initial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_manifest(root)
            cfg = {"stage": "public10", "model": small_config().to_dict(), "loss": asdict(LossConfig()),
                "sampling": asdict(SamplingConfig(window_seconds=10, samples=1, steps=1)),
                "training": {"seed": 4, "updates": 2, "validate_every": 2, "batch_size": 1,
                    "learning_rate": .0001, "weight_decay": .01, "ema_decay": .9, "gradient_clip": 1.,
                    "boundary_probability": 0., "checkpoint_every": 1, "validation_mode": "final_only"}}
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            args = Namespace(config=config, manifest=manifest, output=root / "A", device="cpu",
                             resume=False, init_checkpoint=None)
            with patch.object(trainer, "flow_loss", side_effect=RuntimeError("simulated first-step failure")):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    trainer.train(args)
            initial_path = args.output / "initial.pt"
            initial_bytes = initial_path.read_bytes()
            job = pipeline.Job("A", (), "train", ["--manifest", str(manifest), "--config", str(config),
                "--output", str(args.output), "--device", "cpu"], args.output / "completed.json", "0")
            with self.assertRaisesRegex(RuntimeError, "without last.pt"):
                pipeline.command_for(job)
            changed = json.loads(config.read_text())
            changed["training"]["learning_rate"] *= 2
            config.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "identity"):
                pipeline.command_for(job, recover_initial=True)
            self.assertFalse((args.output / "last.pt").exists())
            config.write_text(json.dumps(cfg))
            command = pipeline.command_for(job, recover_initial=True)
            self.assertIn("--resume", command)
            self.assertEqual(initial_bytes, initial_path.read_bytes())
            self.assertEqual(load_checkpoint(args.output / "last.pt")["update"], 0)
            args.resume = True
            with patch.object(trainer, "evaluate", return_value=({"r": .3}, [])) as evaluate:
                trainer.train(args)
                self.assertEqual(evaluate.call_count, 1)
            self.assertTrue(json.loads(job.artifact.read_text())["completed"])


if __name__ == "__main__":
    unittest.main()
