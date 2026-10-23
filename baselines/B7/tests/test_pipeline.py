from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baselines.B7.pipeline import Job, command_for, plan, run, validate_artifact


class PipelineTests(unittest.TestCase):
    def args(self, root):
        return Namespace(output=root / "output", data_root=root / "raw", metadata=root / "metadata",
                         ptb_root=root / "ptb", gpu_p="0", gpu_a="0", gpu_b="1",
                         max_train=2, dry_run=False)

    def test_real_plan_has_all_prepares_and_correct_dependencies(self):
        jobs = {job.name: job for job in plan(self.args(Path("/example")))}
        self.assertEqual(jobs["P"].dependencies, ("prepare_public",))
        self.assertEqual(jobs["A"].dependencies, ("P", "prepare_huawei"))
        self.assertEqual(jobs["Q"].dependencies, ("P", "prepare_ptb"))
        self.assertEqual(jobs["B"].dependencies, ("Q", "prepare_huawei"))
        self.assertEqual(jobs["B"].gpu, "1")
        self.assertIn("--scale-manifest", jobs["prepare_huawei"].arguments)
        self.assertNotIn("--init-checkpoint", jobs["P"].arguments)

    def test_resume_removes_initialization_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "last.pt").write_bytes(b"saved")
            job = Job("A", (), "train", ["--output", temp, "--init-checkpoint", "P/best_r.pt"], directory / "completed.json", "0")
            command = command_for(job)
            self.assertIn("--resume", command)
            self.assertNotIn("--init-checkpoint", command)
            self.assertEqual((directory / "last.pt").read_bytes(), b"saved")

    def test_partial_prepare_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            job = Job("prepare", (), "prepare", ["--output", temp], directory / "manifest.json")
            with self.assertRaisesRegex(RuntimeError, "partial output"):
                command_for(job)

    def test_completed_stage_requires_consistent_valid_weights_and_reports_score(self):
        from baselines.B7.tests.test_b7 import make_manifest, small_config, payload
        from baselines.B7.model import B7UNet
        from baselines.B7.checkpoint import save_checkpoint, file_sha256
        from baselines.B7.train import code_sha
        from baselines.B7.data import read_manifest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = make_manifest(root)
            # make_manifest returns a path, and each prepared record is immutable.
            config = {"training": {"updates": 2}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = read_manifest(manifest_path)
            value = payload(B7UNet(small_config()))
            value.update(update=2, best_r=.75, identity={"config": config, "code_sha256": code_sha(),
                "manifest_sha256": file_sha256(manifest_path), "record_sha256": {
                    row["record_id"]: file_sha256(row["resolved_path"]) for row in manifest["records"]}})
            save_checkpoint(root / "last.pt", value)
            value = dict(value, update=1)
            save_checkpoint(root / "best_r.pt", value)
            completed = root / "completed.json"
            completed.write_text(json.dumps({"completed": True, "update": 2, "best_r": .75}), encoding="utf-8")
            job = Job("P", (), "train", ["--config", str(config_path), "--manifest", str(manifest_path)], completed, "0")
            self.assertEqual(validate_artifact(job)["best_r"], .75)
            completed.write_text(json.dumps({"completed": True, "update": 2, "best_r": .99}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checkpoints disagree"):
                validate_artifact(job)

    def simulated_run(self, failed=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args(root)
            for folder in (args.data_root, args.metadata, args.ptb_root):
                folder.mkdir()
            jobs = [Job(name, deps, "train", ["--output", str(args.output/name)],
                        args.output/name/"completed.json", "0")
                    for name, deps in (("P", ()), ("A", ("P",)), ("Q", ("P",)), ("B", ("Q",)))]
            events = []
            by_output = {str(job.artifact.parent): job for job in jobs}
            class FakeProcess:
                def __init__(self, command, **kwargs):
                    self.job = by_output[command[command.index("--output") + 1]]
                    self.polls = 0
                    events.append("start:" + self.job.name)
                def poll(self):
                    self.polls += 1
                    if self.polls < (6 if self.job.name == "A" else 1):
                        return None
                    events.append("end:" + self.job.name)
                    return 1 if self.job.name == failed else 0
            with patch("baselines.B7.pipeline.plan", return_value=jobs), \
                 patch("baselines.B7.pipeline.fingerprint", return_value="fixed"), \
                 patch("torch.cuda.is_available", return_value=True), \
                 patch("baselines.B7.pipeline.subprocess.Popen", FakeProcess), \
                 patch("baselines.B7.pipeline.validate_artifact", side_effect=lambda job:
                       {"best_r": .75, "best_checkpoint": str(job.artifact.parent/"best_r.pt")}), \
                 patch("baselines.B7.pipeline.time.sleep"):
                result = run(args)
            status = json.loads((args.output/"pipeline_status.json").read_text())
            return result, status, events

    def test_B_starts_after_Q_without_waiting_for_A(self):
        result, status, events = self.simulated_run()
        self.assertEqual(result, 0)
        self.assertTrue(status["success"])
        self.assertLess(events.index("end:P"), events.index("start:A"))
        self.assertLess(events.index("end:Q"), events.index("start:B"))
        self.assertLess(events.index("start:B"), events.index("end:A"))
        self.assertEqual(set(status["training_results"]), {"P", "A", "Q", "B"})
        self.assertEqual(status["training_results"]["A"]["best_r"], .75)
        self.assertTrue(status["training_results"]["B"]["best_checkpoint"].endswith("best_r.pt"))

    def test_A_failure_does_not_block_Q_and_B(self):
        result, status, events = self.simulated_run("A")
        self.assertEqual(result, 1)
        self.assertEqual(status["jobs"], {"P": "done", "A": "failed", "Q": "done", "B": "done"})
        self.assertNotIn("A", status["training_results"])

    def test_Q_failure_blocks_B_but_allows_A_to_finish(self):
        result, status, events = self.simulated_run("Q")
        self.assertEqual(result, 1)
        self.assertEqual(status["jobs"], {"P": "done", "A": "done", "Q": "failed", "B": "blocked"})
        self.assertNotIn("start:B", events)

    def test_P_failure_blocks_both_routes(self):
        result, status, events = self.simulated_run("P")
        self.assertEqual(result, 1)
        self.assertEqual(status["jobs"]["A"], "blocked")
        self.assertEqual(status["jobs"]["Q"], "blocked")
        self.assertEqual(status["jobs"]["B"], "blocked")


if __name__ == "__main__":
    unittest.main()
