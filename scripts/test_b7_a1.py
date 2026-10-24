"""A1: 10s Huawei training, whole120s scoring, U7/T P reuse and safe resume."""
from argparse import Namespace
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from scripts import run_b7_a1 as runner


MODULES = ("baselines.B7", "baselines.B7T_A1")


def manifest_at(root):
    from baselines.B7.data import DATA_PROTOCOL
    anchor = (100 * np.sin(np.arange(60000, dtype="float32") / 50))[None]
    target = np.repeat(anchor, 12, axis=0)
    records = []
    for name, group, split in (("train", "task1", "train"), ("v1", "task1", "validation"),
                               ("v2", "machine", "validation"), ("v3", "body", "validation")):
        path = root / (name + ".npz")
        np.savez(path, anchor_uV=anchor, target_uV=target, lead_mask=np.ones(12, dtype=bool))
        records.append(dict(record_id=name, target_record_id=name, subject_id=name, source="huawei",
                            group=group, split=split, path=path.name, input_reliable=True))
    path = root / "manifest.json"
    path.write_text(json.dumps(dict(protocol=DATA_PROTOCOL, sample_rate=500, scales_uV=[100.] * 12, records=records)))
    return path


def parent_at(module, root):
    config = importlib.import_module(module + ".config")
    checkpoint = importlib.import_module(module + ".checkpoint")
    package = importlib.import_module(module)
    if module == "baselines.B7":
        mc = config.ModelConfig(channels=(8, 16, 32), time_dim=8, metadata_dim=8, dilations=(1,))
        model = importlib.import_module(module + ".model").B7UNet(mc)
    else:
        mc = config.ModelConfig(embed_dims=(16, 32, 48), heads=(2, 4, 6), depths=(2, 2, 2),
                               window_size=16, mlp_ratio=2, time_dim=8, metadata_dim=8)
        model = importlib.import_module(module + ".model").B7Transformer(mc)
    directory = root / "P"
    directory.mkdir()
    value = dict(architecture_id=package.ARCHITECTURE_ID, format_version=package.CHECKPOINT_VERSION,
        generated_leads=package.GENERATED_LEADS, model_config=mc.to_dict(), loss=asdict(config.LossConfig()),
        sampling=asdict(config.SamplingConfig(window_seconds=10)), stage="public10", update=2,
        best_r=.4, scales_uV=[100.] * 12, ema=model.state_dict(),
        identity={"config": {"training": {"updates": 2}}},
        initialization={"kind": "test_P", "B5_weights_used": False})
    path = directory / "best_r.pt"
    checkpoint.save_checkpoint(path, value)
    (directory / "completed.json").write_text(json.dumps(dict(completed=True, update=2, best_r=.4)))
    return path


def args_at(root, parent, manifest):
    return Namespace(parent_checkpoint=parent, manifest=manifest, output=root / "A1", gpu="0",
        updates=2, batch_size=1, learning_rate=1e-5, checkpoint_every=1, seed=42,
        ode_batch=1, validation_records=1, dry_run=False)


class Marker(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.))


class A1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_recipe_and_parent_auto_detection_for_both_architectures(self):
        for module in MODULES:
            with self.subTest(module=module), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = parent_at(module, root)
                selected, parent = runner.read_parent(path)
                self.assertEqual(selected, module)
                args = runner.parse_args(["--parent-checkpoint", str(path), "--output", str(root / "A1"), "--gpu", "1"])
                cfg = runner.make_config(module, parent, args)
                self.assertEqual(cfg["stage"], "huawei10")
                self.assertEqual(cfg["sampling"]["window_seconds"], 10)
                self.assertEqual(cfg["training"]["updates"], 1000)
                self.assertEqual(cfg["training"]["validate_every"], 1000)
                self.assertEqual(cfg["training"]["batch_size"], 32)
                self.assertEqual(cfg["training"]["boundary_probability"], 0)
                self.assertEqual((cfg["sampling"]["steps"], cfg["sampling"]["samples"]), (16, 16))
                self.assertEqual(cfg, json.loads(json.dumps(cfg)))
                bad = json.loads(json.dumps(cfg))
                bad["sampling"]["window_seconds"] = 40
                with self.assertRaisesRegex(ValueError, "window"):
                    importlib.import_module(module + ".train").validate_settings(bad)

    def test_parent_rejects_wrong_stage_or_incomplete_P(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = parent_at(MODULES[0], root)
            cp = importlib.import_module(MODULES[0] + ".checkpoint")
            saved = cp.load_checkpoint(path)
            saved["stage"] = "huawei40"
            cp.save_checkpoint(path, saved)
            with self.assertRaisesRegex(ValueError, "P/public10"):
                runner.read_parent(path)
            saved["stage"] = "public10"
            cp.save_checkpoint(path, saved)
            (path.parent / "completed.json").write_text(json.dumps(dict(completed=False)))
            with self.assertRaisesRegex(ValueError, "completion"):
                runner.read_parent(path)

    def test_dry_run_does_not_create_output_or_launch_training(self):
        for module in MODULES:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = args_at(root, parent_at(module, root), manifest_at(root))
                args.dry_run = True
                with patch.object(runner.subprocess, "Popen", side_effect=AssertionError("No training on dry-run")):
                    self.assertEqual(runner.run(args), 0)
                self.assertFalse(args.output.exists())

    def test_output_cannot_modify_parent_or_existing_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_at(root, root / "P/best_r.pt", root / "manifest.json")
            args.output = root / "P/new_A1"
            with self.assertRaisesRegex(ValueError, "P directory"):
                runner.run(args)
            args.output = root / "old_A"
            args.output.mkdir()
            marker = args.output / "initial.pt"
            marker.write_bytes(b"existing stage")
            with self.assertRaisesRegex(ValueError, "existing P/A/Q/B"):
                runner.run(args)
            self.assertEqual(marker.read_bytes(), b"existing stage")
            self.assertFalse((args.output / "a1_spec.json").exists())

    def test_short_losses_use_whole10s_and_I_stays_clean(self):
        for module in MODULES:
            loss = importlib.import_module(module + ".losses")
            config = importlib.import_module(module + ".config")
            y = torch.sin(torch.arange(5000) / 25)[None, None].repeat(1, 7, 1)
            p = y.clone()
            p[:, :, :500] *= -1
            parts = loss.reconstruction_losses(p, y, torch.ones_like(y, dtype=torch.bool), config.LossConfig())
            self.assertGreater(parts["mae"].item(), 0)
            self.assertGreater(parts["pcc"].item(), 0)
            self.assertEqual(loss.core_slice(5000), slice(0, 5000))
            anchor = torch.randn(1, 1, 5000)
            observed = []
            class Velocity(nn.Module):
                def forward(self, state, t, a, metadata=None):
                    observed.append(a.clone())
                    return state * 0
            loss.flow_loss(Velocity(), {"anchor": anchor, "target": y,
                "lead_mask": torch.ones(1, 7, dtype=torch.bool)}, config.LossConfig())
            self.assertTrue(torch.equal(observed[0], anchor))

    def test_120s_stitch_uses12_nonoverlapping_full10s_windows(self):
        anchor = torch.arange(60000, dtype=torch.float32)[None]
        for module in MODULES:
            inference = importlib.import_module(module + ".inference")
            config = importlib.import_module(module + ".config")
            seen = []
            def sample(model, q, keys, metadata, sc, **kwargs):
                self.assertEqual(q.shape[-1], 5000)
                seen.extend((float(a[0, 0]), float(a[0, -1])) for a in q)
                return q.repeat(1, 7, 1)
            with patch.object(inference, "sample", side_effect=sample):
                result = inference.predict_record(Marker().eval(), anchor, [1.] * 12, "record",
                    config=config.SamplingConfig(window_seconds=10))
            self.assertEqual(seen, [(float(s), float(s + 4999)) for s in range(0, 60000, 5000)])
            self.assertEqual(result.shape, (12, 60000))
            self.assertTrue(torch.equal(result[0], anchor[0]))
            self.assertTrue(torch.equal(result[1], anchor[0]))
            self.assertTrue(torch.equal(result[6:], anchor.expand(6, -1)))

    def test_huawei_weighting_and_full_record_evaluation(self):
        for module in MODULES:
            evaluation = importlib.import_module(module + ".evaluate")
            rows = [{"group": g, "target_record_id": g,
                     "leads": {str(j): {"r": score} for j in range(1, 12)}}
                    for g, score in (("task1", .9), ("machine", .1), ("body", .3))]
            self.assertAlmostEqual(evaluation.aggregate(rows, "huawei10")["r"], .55)
            self.assertEqual(evaluation.aggregate(rows, "huawei10"), evaluation.aggregate(rows, "huawei40"))
            with tempfile.TemporaryDirectory() as tmp:
                path = manifest_at(Path(tmp))
                data = importlib.import_module(module + ".data").read_manifest(path)
                sampling = importlib.import_module(module + ".config").SamplingConfig(window_seconds=10)
                lengths = []
                def predict(model, anchors, *args, **kwargs):
                    lengths.extend(a.shape[-1] for a in anchors)
                    return torch.stack([torch.tensor(a).repeat(12, 1) for a in anchors])
                with patch.object(evaluation, "predict_records", side_effect=predict):
                    metrics, records = evaluation.evaluate(Marker().eval(), data, sampling, "huawei10")
                self.assertEqual(lengths, [60000] * 3)
                self.assertAlmostEqual(metrics["r"], 1.)
                self.assertEqual(len(records), 3)
                # A short validation target must be rejected, never silently scored as public10.
                np.savez(Path(tmp) / "v1.npz", anchor_uV=np.zeros((1, 5000), dtype="float32"),
                    target_uV=np.zeros((12, 5000), dtype="float32"), lead_mask=np.ones(12, dtype=bool))
                with self.assertRaisesRegex(ValueError, "full120s"):
                    evaluation.evaluate(Marker().eval(), data, sampling, "huawei10")

    def test_end_to_end_final_only_parent_loading_and_completed_reuse(self):
        for module in MODULES:
            self.exercise_runner(module, interrupt=False)

    def test_interrupted_final_validation_resumes_and_allows_gpu_change(self):
        for module in MODULES:
            self.exercise_runner(module, interrupt=True)

    def exercise_runner(self, module, interrupt):
        with self.subTest(module=module, interrupt=interrupt), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = parent_at(module, root)
            parent_bytes = path.read_bytes()
            args = args_at(root, path, manifest_at(root))
            trainer = importlib.import_module(module + ".train")
            checkpoint = importlib.import_module(module + ".checkpoint")
            commands = []
            first = [True]
            class Process:
                def __init__(self, command, env, **kwargs):
                    self.command = command
                    commands.append(command)
                    assert env["CUDA_VISIBLE_DEVICES"] == args.gpu
                def wait(self):
                    def value(flag):
                        return self.command[self.command.index(flag) + 1]
                    child = Namespace(config=value("--config"), manifest=value("--manifest"), output=value("--output"),
                        device="cpu", resume="--resume" in self.command,
                        init_checkpoint=value("--init-checkpoint") if "--init-checkpoint" in self.command else None)
                    error = interrupt and first[0]
                    first[0] = False
                    side_effect = RuntimeError("interrupted validation") if error else None
                    try:
                        with patch.object(trainer, "evaluate", return_value=({"r": .5}, []), side_effect=side_effect) as evaluate:
                            trainer.train(child)
                            assert evaluate.call_count == 1
                    except RuntimeError as exc:
                        if "interrupted validation" not in str(exc):
                            raise
                        return 1
                    return 0
            with patch.object(runner.subprocess, "Popen", Process):
                self.assertEqual(runner.run(args), 1 if interrupt else 0)
                if interrupt:
                    saved = checkpoint.load_checkpoint(args.output / "train/last.pt")
                    self.assertEqual(saved["update"], 2)
                    self.assertTrue(saved["pending_validation"])
                    args.gpu = "1"
                    self.assertEqual(runner.run(args), 0)
                    self.assertIn("--resume", commands[-1])
                self.assertEqual(runner.run(args), 0)
            self.assertEqual(path.read_bytes(), parent_bytes)
            initial = checkpoint.load_checkpoint(args.output / "train/initial.pt")
            source = checkpoint.load_checkpoint(path)
            for key in source["ema"]:
                self.assertTrue(torch.equal(initial["ema"][key], source["ema"][key]))
            best = checkpoint.load_checkpoint(args.output / "train/best_r.pt")
            self.assertEqual(best["update"], 2)
            self.assertEqual(best["stage"], "huawei10")
            self.assertEqual(len((args.output / "train/history.jsonl").read_text().splitlines()), 1)
            args.learning_rate *= 2
            with self.assertRaisesRegex(RuntimeError, "identity"):
                runner.run(args)


if __name__ == "__main__":
    unittest.main()
