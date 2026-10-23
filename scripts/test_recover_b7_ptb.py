from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.recover_b7_ptb import canonical_index, run, wait_for


class PTBRecoveryTests(unittest.TestCase):
    def headers(self, root):
        for i in range(1, 31):
            directory = root / f"patient{i:03d}"
            directory.mkdir(parents=True)
            (directory / f"s{i:04d}.hea").write_text("sample 12 1000 40000\n#Age: 50\n#Sex: Male", encoding="utf-8")

    def test_only_patient_records_retained_and_root_headers_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.headers(root)
            (root / "s0001.hea").write_bytes((root / "patient001/s0001.hea").read_bytes())
            (root / "scratch.hea").write_text("scratch", encoding="utf-8")
            rows, audit = canonical_index(root)
            self.assertEqual(audit["headers"], 32)
            self.assertEqual(audit["retained"], 30)
            self.assertEqual(set(audit["excluded_root_headers"]), {"s0001.hea", "scratch.hea"})
            self.assertEqual(len({r["target_record_id"] for r in rows}),30)
            self.assertIn("ptbdb:patient001/s0001", {r["target_record_id"] for r in rows})
            self.assertTrue((root / "scratch.hea").exists())

    def test_unexpected_nested_header_requires_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.headers(root)
            (root / "other").mkdir()
            (root / "other/a.hea").write_text("test", encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Unexpected nested"):
                canonical_index(root)

    def test_failed_parent_does_not_wait_forever(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root / "pipeline_status.json").write_text(json.dumps({"jobs":{"P":"failed"}}))
            with self.assertRaisesRegex(RuntimeError,"P is failed"):
                wait_for(root / "P/completed.json",root,"P",60)

    def test_cannot_race_an_original_active_Q(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root / "pipeline_status.json").write_text(json.dumps({"jobs":{"Q":"running","B":"pending"}}))
            with self.assertRaisesRegex(RuntimeError,"blocked"):
                run(Namespace(output=root,ptb_root=root,gpu="0",wait_seconds=60))

    def test_route_prepares_then_waits_P_and_runs_Q_B_with_GPU0(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root / "pipeline_status.json").write_text(json.dumps({"jobs":{"Q":"blocked","B":"blocked"}}))
            events=[]
            def fake_process(command, **kwargs):
                directory=Path(command[command.index("--output")+1])
                directory.mkdir(parents=True,exist_ok=True)
                stage="prepare" if command[3].endswith("prepare") else directory.name
                artifact=directory / ("manifest.json" if stage=="prepare" else "completed.json")
                artifact.write_text("{}")
                events.append(stage)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "" if stage=="prepare" else "0")
            def fake_wait(path, output, stage, timeout):
                events.append("wait:"+stage)
            with patch("scripts.recover_b7_ptb.canonical_index",return_value=([],{})), \
                 patch("baselines.B7.data.read_manifest",return_value={"scales_uV":[100.]*12}), \
                 patch("baselines.B7.pipeline.validate_artifact",return_value={"best_r":.7}), \
                 patch("scripts.recover_b7_ptb.wait_for",side_effect=fake_wait), \
                 patch("scripts.recover_b7_ptb.subprocess.run",side_effect=fake_process):
                self.assertEqual(run(Namespace(output=root,ptb_root=root,gpu="0",wait_seconds=60)),0)
            self.assertEqual(events,["prepare","wait:P","Q","wait:prepare_huawei","B"])
            state=json.loads((root / "ptb_recovery/status.json").read_text())
            self.assertTrue(state["success"])
            self.assertEqual(set(state["training_results"]),{"P","Q","B"})


if __name__=="__main__":
    unittest.main()
