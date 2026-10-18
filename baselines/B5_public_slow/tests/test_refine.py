"""CPU synthetic regression: historical handoff identity and exact stage2 replay."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_public_slow import train, refine, PROTOCOL
from baselines.B5_public_slow.common import model_from_bundle, load_study
from baselines.B5_public_slow.tests.test_public_slow import study_fixture, Tiny
from baselines.B5.scoring import SCORE_PROTOCOL


def fake_eval(model, study, bundle, device, output):
    Path(output).mkdir(parents=True)
    torch.rand(7)  # Additional validation must not disturb the training RNG.
    r = float(model.scale.detach()) / 2
    result = {"task1_r_missing11": r, "task2_r_missing11": r, "main_correlation": r,
              "score_protocol": SCORE_PROTOCOL, "selection_metric": "main_correlation"}
    write(Path(output) / "correlation_score.json", result)
    return result


class RefinementTests(unittest.TestCase):
    def test_prepare_four_trials_preserves_sources_and_pairs_only_learning_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "seed42"
            original.mkdir()
            path, bundle = study_fixture(original)
            settings = read(path)
            settings["training"].update(stage1_updates=1560, stage2_updates=1560)
            historical = []
            reference_score = {"task1_r_missing11": .7, "task2_r_missing11": .71,
                               "main_correlation": .705, "score_protocol": SCORE_PROTOCOL}
            for seed in (42, 43):
                folder = root / f"seed{seed}"
                folder.mkdir(exist_ok=True)
                study_path = folder / "shared/study.json"
                if seed == 43:
                    study_path.parent.mkdir()
                    shutil.copyfile(path.parent / "components.pt", study_path.parent / "components.pt")
                source = copy.deepcopy(settings)
                source["training"]["seed"] = seed
                source["code_sha256"] = f"historical-seed{seed}-code"
                write(study_path, source)
                (folder / "P").mkdir()
                torch.save({"protocol": PROTOCOL, "study_sha256": sha(study_path), "group": "P",
                            "stage": 1, "step": 1560, "training_seed": seed,
                            "model": bundle["slow"]["model"], "FM_state_sha256": bundle["fm_state_sha256"],
                            "initial_slow_state_sha256": bundle["slow_state_sha256"]}, folder / "P/stage1_last.pt")
                write(folder / "P/completed.json", {"completed": True})
                write(folder / "P/validation/stage2_step000390/correlation_score.json", reference_score)
                historical.append(sha(study_path))
            with patch("baselines.B5_public_slow.prepare.DEFAULT_PROFILE", original / "profile.json"):
                refine.prepare_trials(root / "seed42", root / "seed43", root / "trials")
            for seed in (42, 43):
                pair = []
                for arm in refine.ARMS:
                    p = root / "trials" / f"seed{seed}_{arm}/shared/study.json"
                    trial, _ = load_study(p, deep=True)
                    self.assertEqual(trial["training"]["stage1_updates"], 0)
                    self.assertEqual(trial["training"]["stage2_updates"], 780)
                    self.assertEqual(trial["training"]["validate_every"], 78)
                    self.assertEqual(trial["sampling"]["seed"], 42)
                    self.assertEqual(trial["huawei"], settings["huawei"])
                    pair.append(trial)
                    run_folder = p.parents[1] / "P"
                    run_folder.mkdir()
                    write(run_folder / "completed.json", {"completed": True})
                    for step in range(0, 781, 78):
                        score = reference_score | {"main_correlation": .705}
                        write(run_folder / "validation" / f"stage2_step{step:06d}/correlation_score.json", score)
                    torch.save({"study_sha256": sha(p), "stage": 2, "step": 390,
                                "validation_score": reference_score}, run_folder / "best.pt")
                self.assertEqual(pair[0]["stage2_restart"], pair[1]["stage2_restart"])
                c, d = copy.deepcopy(pair[0]), copy.deepcopy(pair[1])
                c["training"].pop("learning_rate"); d["training"].pop("learning_rate")
                self.assertEqual(c, d)
            self.assertEqual(historical, [sha(root / f"seed{s}/shared/study.json") for s in (42, 43)])
            refine.summarize(root / "trials", root / "summary")
            self.assertEqual(len(read(root / "summary/summary.json")["rows"]), 4)
            with self.assertRaises(FileExistsError):
                with patch("baselines.B5_public_slow.prepare.DEFAULT_PROFILE", original / "profile.json"):
                    refine.prepare_trials(root / "seed42", root / "seed43", root / "trials")

    def test_replay_same_handoff_matches_original_and_resume_at_both_initial_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = study_fixture(root)
            settings = read(path)
            settings["training"].update(stage1_updates=2, stage2_updates=3, validate_every=3)
            write(path, settings)
            with patch.object(train, "model_from_bundle", side_effect=lambda *a: Tiny()), \
                 patch.object(train, "evaluate", side_effect=fake_eval), \
                 patch.object(train, "public_diagnostic", return_value={}):
                train.run(path, "P", root / "original", "cpu")
                cp_path = root / "original/stage1_last.pt"
                handoff = torch.load(cp_path, weights_only=True)
                initial_sha = sha(cp_path)
                newpath = root / "replay/shared/study.json"
                newpath.parent.mkdir(parents=True)
                shutil.copyfile(path.parent / "components.pt", newpath.parent / "components.pt")
                trial = copy.deepcopy(settings)
                trial["training"].update(stage1_updates=0, validate_every=1)
                trial["stage2_restart"] = {"checkpoint": str(cp_path), "sha256": initial_sha,
                    "source_study_sha256": sha(path), "source_step": 2,
                    "slow_state_sha256": state_sha(handoff["model"])}
                trial["external_checks"].append({"path": str(cp_path), "sha256": initial_sha})
                write(newpath, trial)
                train.run(newpath, "P", root / "full", "cpu")
                expected = torch.load(root / "original/last.pt", weights_only=True)
                actual = torch.load(root / "full/last.pt", weights_only=True)
                torch.testing.assert_close(actual["model"]["scale"], expected["model"]["scale"], rtol=0, atol=0)
                self.assertEqual(expected["samplers"], actual["samplers"])
                self.assertEqual(sha(cp_path), initial_sha)
                self.assertFalse((root / "full/stage1_last.pt").exists())
                self.assertEqual(read(root / "full/completed.json")["stage1_updates"], 0)
                self.assertTrue(read(root / "full/completed.json")["stage1_reused"])
                # Interrupt immediately after common-initial save, or after stage2 step0 save.
                original_save = train.atomic_save
                for interrupted_stage in (0, 2):
                    destination = root / f"resume{interrupted_stage}"
                    def stop(path, payload):
                        original_save(path, payload)
                        if Path(path).name == "last.pt" and payload["stage"] == interrupted_stage and payload["step"] == 0:
                            raise RuntimeError("synthetic stop")
                    with patch.object(train, "atomic_save", side_effect=stop):
                        with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                            train.run(newpath, "P", destination, "cpu")
                    train.run(newpath, "P", destination, "cpu", resume=True)
                    resumed = torch.load(destination / "last.pt", weights_only=True)
                    torch.testing.assert_close(actual["model"]["scale"], resumed["model"]["scale"], rtol=0, atol=0)
                    self.assertEqual(actual["samplers"], resumed["samplers"])
                trial["training"]["learning_rate"] = 3e-5
                lowpath = root / "low/shared/study.json"
                lowpath.parent.mkdir(parents=True)
                shutil.copyfile(path.parent / "components.pt", lowpath.parent / "components.pt")
                write(lowpath, trial)
                train.run(lowpath, "P", root / "low/run", "cpu")
                low = torch.load(root / "low/run/last.pt", weights_only=True)
                self.assertEqual(low["samplers"], actual["samplers"])
                self.assertNotEqual(float(low["model"]["scale"]), float(actual["model"]["scale"]))
                with self.assertRaisesRegex(ValueError, "P stage1"):
                    train.run(lowpath, "M", root / "wrong", "cpu")

    def test_historical_source_allows_recorded_old_code_but_rejects_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, bundle = study_fixture(root)
            source = read(path)
            source["code_sha256"] = "historical-code-fingerprint"
            source["training"].update(stage1_updates=1560, stage2_updates=1560)
            write(path, source)
            cp_path = root / "P/stage1_last.pt"
            cp_path.parent.mkdir()
            cp = {"protocol": PROTOCOL, "study_sha256": sha(path), "group": "P", "stage": 1,
                  "step": 1560, "training_seed": 42, "model": bundle["slow"]["model"],
                  "FM_state_sha256": bundle["fm_state_sha256"], "initial_slow_state_sha256": bundle["slow_state_sha256"]}
            torch.save(cp, cp_path)
            write(root / "P/completed.json", {"completed": True})
            write(root / "P/validation/stage2_step000390/correlation_score.json", {"score_protocol": SCORE_PROTOCOL})
            _, _, imported = refine.historical_handoff(root, 42)
            self.assertEqual(imported["source_code_sha256"], "historical-code-fingerprint")
            with self.assertRaisesRegex(ValueError, "Study code changed"):
                load_study(path)
            cp["stage"] = 2
            torch.save(cp, cp_path)
            with self.assertRaisesRegex(ValueError, "stage1_last"):
                refine.historical_handoff(root, 42)

    def test_step390_reference_guard_detects_changed_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score = {"task1_r_missing11": .7, "task2_r_missing11": .71, "main_correlation": .705,
                     "score_protocol": SCORE_PROTOCOL}
            write(root / "reference.json", score)
            write(root / "P/validation/stage2_step000390/correlation_score.json", score)
            study = {"training": {"learning_rate": 1e-4},
                     "stage2_restart": {"reference_step390": str(root / "reference.json")}}
            with patch.object(refine, "load_study", return_value=(study, {})):
                refine.verify_reproduction(root)
                self.assertTrue(read(root / "reproduction.json")["reference_matched"])
                write(root / "P/validation/stage2_step000390/correlation_score.json", score | {"main_correlation": .8})
                with self.assertRaisesRegex(ValueError, "not reproduced"):
                    refine.verify_reproduction(root)


if __name__ == "__main__":
    unittest.main()
