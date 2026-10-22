"""Synthetic checks for fusion, actual S100 P changes, and restart bookkeeping."""
import argparse
from copy import deepcopy
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_best.slow import SlowCNN, predict_native
from baselines.B5_slowrate import RATES, PROTOCOL as RATE_PROTOCOL
from baselines.B5_slowrate.common import code_sha as rate_code_sha
from baselines.B5_slowrate.model import RateCorrection, visible_inputs
from baselines.B5_slowrate.tests import test_slowrate as rate_tests
from baselines.B5_slowrate.prepare import prepare as prepare_rates
from baselines.B5_slowrate_followup.model import Followup, PDelta, FixedEnsemble, inputs, array_batch
from baselines.B5_slowrate_followup.prepare import prepare
from baselines.B5_slowrate_followup.common import load_study
from baselines.B5_slowrate_followup import train
from baselines.B5_slowrate_followup.suite import Runner
from baselines.B5_slowrate_followup.summarize import summarize


def batch():
    raw = rate_tests.fixture()
    fields = {"signal_" + g: visible_inputs(raw["anchor_uV"], 100., raw, g)["global_i"] for g in RATES}
    fields.update({k: raw[k] for k in ("numeric", "sex", "field_mask", "age_topcoded")})
    fields["anchor50"] = raw["anchor_uV"].reshape(1, 6000, 10).mean(-1) / 100.
    return array_batch(fields, "cpu")


def sources():
    result = {}
    for g in RATES:
        model = RateCorrection(g, channels=8, layers=1, heads=2)
        torch.nn.init.normal_(model.output.weight, std=.02)
        result[g] = model.eval()
    return result


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_fusion_controls_reproduce_reference_and_freeze_sources(self):
        data, src = batch(), sources()
        hashes = {g: state_sha(m.state_dict()) for g, m in src.items()}
        initial_heads = []
        for group in ("FUSION_SAME", "FUSION"):
            torch.manual_seed(43)
            model = Followup(group, deepcopy(src)).train()
            initial_heads.append(state_sha(model.fusion.state_dict()))
            self.assertTrue(torch.equal(model(**data), model.reference_delta(**data)))
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            for _ in range(2):
                opt.zero_grad()
                (model(**data) - .2).square().mean().backward()
                opt.step()
            self.assertGreater(float(model.fusion[-1].weight.detach().abs().sum()), 0.)
            self.assertEqual({g: state_sha(m.state_dict()) for g, m in model.sources.items()}, hashes)
        self.assertEqual(initial_heads[0], initial_heads[1])

    def test_fixed_fusion_means_corrections_before_context(self):
        data, src = batch(), sources()
        model = FixedEnsemble(list(src.values()))
        with torch.no_grad():
            expected = torch.stack([m(global_i=data["signal_" + g], **{k: data[k] for k in
                                    ("numeric", "sex", "field_mask", "age_topcoded")}) for g, m in src.items()]).mean(0)
            self.assertTrue(torch.equal(model(**data), expected))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_P50_tail_is_actual_P_change_and_only_tail_gets_gradients(self):
        data = batch()
        original = SlowCNN(channels=4).eval()
        model = PDelta(original)
        meta = {k: data[k] for k in ("numeric", "sex", "field_mask", "age_topcoded")}
        zero = model(data["anchor50"], **meta)
        self.assertEqual(torch.count_nonzero(zero).item(), 0)
        self.assertEqual(model.receptive_field_samples, 2045)
        with torch.no_grad():
            model.p.output.weight.add_(.01)
        actual = model(data["anchor50"], **meta)
        with torch.no_grad():
            expected = predict_native(model.p.eval(), {"anchor": data["anchor50"], **meta}) - predict_native(
                original, {"anchor": data["anchor50"], **meta})
        self.assertTrue(torch.equal(actual, expected))
        actual.square().mean().backward()
        self.assertIsNone(model.p.input.conv.weight.grad)
        self.assertIsNone(model.p.blocks[0].one.conv.weight.grad)
        self.assertGreater(float(model.p.blocks[-1].two.conv.weight.grad.abs().sum()), 0.)
        self.assertFalse(any(p.requires_grad for p in model.reference_blocks.parameters()))

    def test_S100_control_and_expanded_rf_initial_match_and_far_input_dependency(self):
        data = batch()
        meta = {k: data[k] for k in ("numeric", "sex", "field_mask", "age_topcoded")}
        original = SlowCNN(channels=4).eval()
        control = PDelta(original, context_seconds=100).eval()
        expanded = PDelta(original, context_seconds=100, extra_dilation=256).eval()
        self.assertEqual(expanded.receptive_field_samples, 4093)
        self.assertTrue(torch.equal(control(data["anchor50"], **meta), expanded(data["anchor50"], **meta)))
        self.assertTrue(torch.equal(expanded(data["anchor50"], **meta), expanded(data["anchor50"], initial=True, **meta)))
        torch.nn.init.normal_(expanded.extra[0].two.conv.weight, std=.03)
        modified = data["anchor50"].clone()
        modified[..., 4500:4750] += 2.  # 90..95s, outside the old RF of a55s prediction.
        with torch.no_grad():
            local_change = (control(data["anchor50"], **meta) - control(modified, **meta))[..., 2750]
            remote_change = (expanded(data["anchor50"], **meta) - expanded(modified, **meta))[..., 2750]
        self.assertEqual(torch.count_nonzero(local_change).item(), 0)
        self.assertGreater(float(remote_change.abs().max()), 1e-7)

    def test_all_followups_accept_only_visible_input_allowlist(self):
        data, src = batch(), sources()
        data["target_uV"] = torch.ones(1, 12, 60000) * 999
        allowed = inputs(data)
        self.assertNotIn("target_uV", allowed)
        for group in ("CONT", "PTAIL", "PCTX", "PCTX_RF"):
            model = Followup(group, deepcopy(src), original_p=SlowCNN(channels=4)).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(**allowed).shape), (1, 11, 6000))
                self.assertTrue(torch.equal(model(**allowed), model.reference_delta(**allowed)))
        longer = PDelta(SlowCNN(channels=4).eval(), context_seconds=120, extra_dilation=384)
        self.assertEqual(longer.receptive_field_samples, 5117)
        self.assertEqual(tuple(longer(data["anchor50"], **{k: data[k] for k in
            ("numeric", "sex", "field_mask", "age_topcoded")} ).shape), (1, 11, 6000))


class WorkflowTests(unittest.TestCase):
    def fixture(self, root):
        helper = rate_tests.WorkflowTests()
        source = helper.parent(root)
        shared = root / "rates/shared"
        prepare_rates(helper.prepare_args(source, shared))
        study = read(shared / "study.json")
        for group, hz in RATES.items():
            torch.manual_seed(study["training"]["seed"])
            model = RateCorrection(group, **study["model"])
            torch.nn.init.normal_(model.output.weight, std=.02)
            folder = root / "rates" / group
            folder.mkdir()
            cp = {"protocol": RATE_PROTOCOL, "group": group, "study_sha256": sha(shared / "study.json"),
                  "code_sha256": rate_code_sha(), "base_sha256": study["base_sha256"],
                  "body_sha256": study["body_sha256"], "sampling_hz": hz, "model": model.state_dict(),
                  "epoch": 2, "best_r": .72}
            torch.save(cp, folder / "best.pt")
            write(root / "rates/verification" / group / "best/evaluation.json", {"completed": True,
                "summary": {"group": group, "epoch": 2, **{k: .72 for k in
                  ("mean_r", "task1_r", "machine_r", "body_r", "task2_r")}}})
        args = argparse.Namespace(rate_run=str(root / "rates"), output=str(root / "followup/shared"),
            reference_group="R5", groups="CONT,FUSION_SAME,FUSION", training_seed=43, epochs=2,
            validate_every=1, batch_records=1, learning_rate=1e-4, fusion_learning_rate=1e-4,
            p_preservation=.1, tail_blocks=2)
        prepare(args)
        return root / "followup/shared"

    def test_preparation_resume_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shared = self.fixture(root)
            study = load_study(shared)
            old_hash = sha(root / "rates/R5/best.pt")
            def args(name, resume=None):
                return argparse.Namespace(run=str(shared), output=str(root / name), group="CONT", device="cpu", resume=resume)
            with patch.object(train, "Evaluator", rate_tests.FakeEvaluator):
                train.run(args("full"))
                real_save = train.atomic_save
                def interrupted(path, payload):
                    real_save(path, payload)
                    if Path(path).name == "last.pt" and payload["epoch"] == 1:
                        raise InterruptedError("Synthetic interruption")
                with patch.object(train, "atomic_save", interrupted):
                    with self.assertRaises(InterruptedError):
                        train.run(args("resumed"))
                train.run(args("resumed", str(root / "resumed/last.pt")))
            a = torch.load(root / "full/last.pt", weights_only=True)
            b = torch.load(root / "resumed/last.pt", weights_only=True)
            self.assertEqual(state_sha(a["model"]), state_sha(b["model"]))
            self.assertEqual(sha(root / "rates/R5/best.pt"), old_hash)
            self.assertEqual(study["reference"]["mean_r"], .72)
            with (root / "rates/R5/best.pt").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                load_study(shared)

    def test_suite_skips_completed_and_retains_incomplete_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = {"steps": {}}
            runner = Runner(root, state)
            out = root / "step"
            def process(*_, **__):
                write(out / "completed.json", {"completed": True})
                return SimpleNamespace(pid=123, stdout=io.StringIO("synthetic output\n"), wait=lambda: 0)
            with patch("baselines.B5_slowrate_followup.suite.subprocess.Popen", side_effect=process) as mocked:
                for _ in range(2):
                    runner.step("test", "synthetic.module", [], out / "completed.json", output=out)
                self.assertEqual(mocked.call_count, 1)
                (out / "completed.json").unlink()
                runner.step("test", "synthetic.module", [], out / "completed.json", output=out)
            self.assertEqual(len(list(root.glob("step_interrupted_*"))), 1)

    def test_P_training_updates_actual_tail_and_summary_uses_initial_and_actual_best(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shared = self.fixture(root)
            manifest = read(shared / "study.json")
            manifest["groups"] = ["PTAIL", "PCTX", "PCTX_RF"]
            manifest["learning_rates"].update({g: 1e-4 for g in manifest["groups"]})
            candidates = {}
            for g in manifest["groups"]:
                torch.manual_seed(43)
                candidates[g] = Followup(g, sources(), original_p=SlowCNN(channels=4))
                manifest["initial_weights_sha256"][g] = state_sha(candidates[g].state_dict())
            write(shared / "study.json", manifest)
            def initialize(g, study, device):
                return deepcopy(candidates[g]).to(device)
            # Patient/context scoring is the only mocked component; actual P
            # objective, optimizer, gradients and checkpoints run on synthetic I.
            with patch.object(train, "initialize", initialize), patch.object(train, "Evaluator", rate_tests.FakeEvaluator):
                train.run(argparse.Namespace(run=str(shared), output=str(root / "PTAIL"),
                                            group="PTAIL", device="cpu", resume=None))
            cp = torch.load(root / "PTAIL/last.pt", weights_only=True)
            self.assertGreater(cp["updates"], 0)
            self.assertNotEqual(state_sha({"tail": cp["model"]["p_delta.p.output.weight"]}),
                                state_sha({"tail": candidates["PTAIL"].state_dict()["p_delta.p.output.weight"]}))
            self.assertTrue(torch.equal(cp["model"]["p_delta.p.input.conv.weight"],
                                        candidates["PTAIL"].state_dict()["p_delta.p.input.conv.weight"]))
            follow = root / "followup"
            for g, value in (("PTAIL", .72), ("PCTX", .721), ("PCTX_RF", .725)):
                write(follow / g / "completed.json", {"completed": True, "updates": 2, "best_r": value})
                def row(r):
                    return {"group": g, "epoch": 2, **{k: r for k in
                        ("mean_r", "task1_r", "machine_r", "body_r", "task2_r")}}
                write(follow / g / "validation/epoch_0000/evaluation.json", {"summary": row(.71)})
                for name in ("best", "last"):
                    write(follow / "verification" / g / name / "evaluation.json",
                          {"completed": True, "population": {}, "summary": row(value)})
            summarize(follow, follow / "comparison")
            report = read(follow / "comparison/summary.json")
            self.assertAlmostEqual(report["contrasts"]["PCTX_RF_minus_PCTX"], .004)
            self.assertIn("PCTX_RF_initial", {r["candidate"] for r in report["summary"]})


if __name__ == "__main__":
    unittest.main()
