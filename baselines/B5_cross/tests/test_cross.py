"""CPU synthetic tests: no real data, FM generation, optimizer or training loop."""
import copy
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5_best import SAMPLING
from baselines.B5_best.bundle import export_bundle, load_bundle
from baselines.B5_best.common import write, read, csv_write, sha, state_sha
from baselines.B5_best.combine import combine
from baselines.B5.metadata import encode_demographics
from baselines.B5_best.slow import SlowCNN
from baselines.B5_best.tests.test_hybrid import fixture
from baselines.B5_slow_pcc import CACHE_PROTOCOL
from baselines.B5_slow_pcc.common import code_digest
from baselines.B5_cross.components import control_payload, legacy_state_sha, candidate, save_candidate
from baselines.B5_cross.run import (SavedControl, validation_manifest, predict_slow, score_combo,
                                    check_reference, noise_inputs, run)

torch.set_num_threads(2)


def setup(root):
    fp, sp, info, profile = fixture(root)
    bundle = export_bundle(fp, sp, info, root / "components.pt", profile)
    study_path = root / "study.json"; write(study_path, {"fixture": True})
    cache = root / "cache"; cache.mkdir()
    raw = np.tile((100 * np.sin(np.arange(60000) / 25)).astype(np.float32), (12, 1))
    packed = {"anchor": raw[:1].reshape(1, 6000, 10).mean(-1) / 100,
              "target_uV": raw, "fm_uV": raw.copy(), "mask": np.ones(12, bool),
              "slow_target": np.zeros((11, 6000), np.float32), **encode_demographics({})}
    rows, entries = {}, []
    for task, devices in (("task1", ("watch_ecg",)), ("task2", ("ecg_machine_d6", "body_scale_d6"))):
        rows[task] = []
        for n, device in enumerate(devices):
            name = task + device
            data = {k: v.copy() for k, v in packed.items()}
            # Same target, deliberately different historical task/pair FM.
            data["fm_uV"][6] += (n + (task == "task2")) * 10 * np.cos(np.arange(60000) / 80)
            np.savez(cache / (name + ".npz"), **data)
            entries.append({"cache_key": task + ":" + name, "pair_id": name,
                            "target_record_id": "target", "subject_id": "patient",
                            "file": name + ".npz", "sha256": sha(cache / (name + ".npz"))})
            rows[task] += [{"pair_id": name, "target_record_id": "target", "subject_id": "patient",
                           "input_type": device, "start_sample_500hz": str(s), "expected_window_count": "12"}
                          for s in range(0, 60000, 5000)]
    manifest = {"protocol": CACHE_PROTOCOL, "completed": True, "study_sha256": sha(study_path),
                "sampling": dict(SAMPLING), "validation": entries, "validation_rows": rows}
    write(cache / "cache.json", manifest)
    evaluation = root / "control_eval"; evaluation.mkdir()
    for task, metadata in rows.items():
        folder = evaluation / task; folder.mkdir()
        target = np.stack([raw[:, int(r["start_sample_500hz"]):int(r["start_sample_500hz"]) + 5000]
                           for r in metadata])
        # Permuted rows and arrays must be correctly realigned, never indexed by position.
        order = np.arange(len(metadata))[::-1]
        csv_write(folder / "window_metadata.csv", [metadata[i] for i in order])
        np.save(folder / "target_uV.npy", target[order])
        np.save(folder / "prediction_uV.npy", (target + 3 * np.cos(np.arange(5000) / 33))[order].astype(np.float32))
        np.save(folder / "target_quality_mask.npy", np.ones((len(metadata), 12), bool))
    state = copy.deepcopy(bundle["fm"]["ema"])
    cp = {"protocol": "B5-scratch-early-public-v1", "group": "control", "slow_variant": "none",
          "epoch": 100, "sampling": dict(SAMPLING), "model": state,
          "model_ema_sha256": legacy_state_sha(state), "model_config": bundle["fm"]["model_config"],
          "base_architecture_hash": bundle["fm"]["architecture_hash"],
          "scales": bundle["fm"]["scales"], "scales_sha256": bundle["fm"]["scales_sha256"]}
    cp_path = root / "control100.pt"; torch.save(cp, cp_path)
    report = {"completed": True, "epoch": 100, "checkpoint_sha256": sha(cp_path),
              "target_used_at_inference": False, "fingerprint": {"group": "control",
              "sampling": dict(SAMPLING), "model_ema_sha256": cp["model_ema_sha256"]}}
    write(evaluation / "evaluation.json", report)
    return bundle, study_path, cache, manifest, raw, cp_path, evaluation


class CrossTests(unittest.TestCase):
    def test_control100_stores_EMA_in_model_and_rejects_wrong_group_epoch_scale_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, _, _, _, _, cp_path, evaluation = setup(root)
            payload, _ = control_payload(cp_path, evaluation, bundle)
            self.assertEqual(state_sha(payload["ema"]), bundle["fm_state_sha256"])
            cp = torch.load(cp_path, weights_only=True)
            for field, value in (("group", "slow"), ("epoch", 45), ("scales_sha256", "wrong"),
                                 ("model_ema_sha256", "wrong")):
                changed = dict(cp); changed[field] = value; torch.save(changed, cp_path)
                report = read(evaluation / "evaluation.json")
                report["checkpoint_sha256"] = sha(cp_path)
                write(evaluation / "evaluation.json", report)
                with self.assertRaises(ValueError):
                    control_payload(cp_path, evaluation, bundle)
            torch.save(cp, cp_path)

    def test_saved_predictions_align_by_identity_and_reject_changed_targets_QC(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, _, _, manifest, raw, _, evaluation = setup(Path(temp))
            saved = SavedControl(evaluation, manifest["validation_rows"])
            try:
                fm = saved.record("task1", list(range(12)), raw, np.ones(12, bool))
                self.assertEqual(fm.shape, (12, 60000))
                self.assertTrue(np.isfinite(fm).all())
                with self.assertRaises(ValueError):
                    saved.record("task1", list(range(12)), raw + 1, np.ones(12, bool))
                bad_mask = np.ones(12, bool); bad_mask[8] = False
                with self.assertRaises(ValueError):
                    saved.record("task1", list(range(12)), raw, bad_mask)
            finally:
                saved.close()

    def test_validation_cache_integrity_and_no_training_archives_required(self):
        with tempfile.TemporaryDirectory() as temp:
            _, study, cache, manifest, _, _, _ = setup(Path(temp))
            checked = validation_manifest(cache, study)
            self.assertNotIn("train", checked)
            with (cache / manifest["validation"][0]["file"]).open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                validation_manifest(cache, study)

    def test_slow_uses_visible_fields_once_per_target_and_rejects_different_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, cache, manifest, _, _, _ = setup(Path(temp))
            calls = []
            def fake_generate(model, inputs):
                calls.append(set(inputs))
                return torch.zeros(1, 11, 6000)
            with patch("baselines.B5_cross.run.generate", side_effect=fake_generate):
                values = predict_slow(cache, manifest["validation"], torch.nn.Identity(),
                                      np.ones(12), torch.device("cpu"))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], {"anchor", "numeric", "sex", "field_mask", "age_topcoded"})
            self.assertEqual(values["target"].shape, (11, 60000))

    def test_pair_specific_FM_full_record_filter_and_portable_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, _, cache, manifest, raw, _, evaluation = setup(root)
            learned = {"target": np.zeros((11, 60000))}
            saved = SavedControl(evaluation, manifest["validation_rows"])
            try:
                scores = []
                for fm in ("original", "control100"):
                    folder = root / fm
                    scores.append(score_combo(fm, "pcc20", learned, cache, manifest, saved, None,
                                              folder, dict(SAMPLING)))
                    result = np.load(folder / "task2/prediction_uV.npy")
                    if fm == "original":
                        for n, entry in enumerate(manifest["validation"][1:]):
                            with np.load(cache / entry["file"]) as archive:
                                np.testing.assert_array_equal(np.concatenate(result[n*12:(n+1)*12], axis=-1),
                                                              combine(archive["fm_uV"], learned["target"], .75))
                self.assertEqual(scores[0]["selection_metric"], "main_correlation")
                provenance = {"control100_checkpoint_sha256": "fm100", "pcc20_sha256": "slowpcc", "pcc20_epoch": 20}
                portable = candidate(bundle, bundle["fm"], bundle["slow"]["model"], "control100", "pcc20", provenance)
                save_candidate(root / "cross.pt", portable, scores[1], 42)
                loaded = load_bundle(root / "cross.pt")
                self.assertEqual(loaded["slow_beta"], .75)
                self.assertEqual(loaded["provenance"]["fm_sha256"], "fm100")
                self.assertFalse(loaded["target_used_at_inference"])
                save_candidate(root / "seed43.pt", portable, scores[1], 43)
                self.assertEqual(load_bundle(root / "seed43.pt")["sampling"]["seed"], 42)
                self.assertIsNone(load_bundle(root / "seed43.pt")["reference_qc_r"])
            finally:
                saved.close()

    def test_six_combo_end_to_end_without_training_or_FM_sampling_and_digest_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, study_path, cache, manifest, _, cp_path, evaluation = setup(root)
            slow_run = root / "slow_run"; (slow_run / "shared").mkdir(parents=True)
            study = {"beta": .75}
            write(slow_run / "shared/study.json", read(study_path))
            # Real source paths are supplied through mocks; all scoring/export remains real.
            for group in ("control", "pcc"):
                (slow_run / group).mkdir()
                (slow_run / group / "best.pt").write_bytes(group.encode())
            (slow_run / "cache").mkdir()
            for entry in manifest["validation"]:
                shutil.copyfile(cache / entry["file"], slow_run / "cache" / entry["file"])
            write(slow_run / "cache/cache.json", manifest)
            def load_cp(path, study_path, device):
                group = Path(path).parent.name
                model = SlowCNN()
                model.load_state_dict(bundle["slow"]["model"])
                return ({"group": group, "cache_sha256": sha(slow_run / "cache/cache.json"), "epoch": 20,
                         "validation_score": {"task1_r_missing11": 0., "task2_r_missing11": 0.}},
                        study, bundle, model)
            before = code_digest()
            args = SimpleNamespace(output=str(root / "result"), slow_run=str(slow_run),
                     control_checkpoint=str(cp_path), control_evaluation=str(evaluation), device="cpu", seed=42)
            with patch("baselines.B5_cross.run.load_study", return_value=(study, bundle)), \
                 patch("baselines.B5_cross.run.validation_manifest", return_value=manifest), \
                 patch("baselines.B5_cross.run.load_checkpoint", side_effect=load_cp), \
                 patch("baselines.B5_cross.run.slow_model", return_value=torch.nn.Identity()), \
                 patch("baselines.B5_cross.run.predict_slow", return_value={"target": np.zeros((11, 60000))}) as slow, \
                 patch("baselines.B5_cross.run.check_reference") as references, \
                 patch("baselines.B5_cross.run.generate_fm", side_effect=AssertionError("No FM inference in seed42")):
                run(args)
            result = read(root / "result/result.json")
            self.assertTrue(result["completed"])
            self.assertEqual(len(result["scores"]), 6)
            self.assertEqual(slow.call_count, 3)
            self.assertEqual(references.call_count, 4)
            self.assertFalse(result["training_started"])
            self.assertEqual(code_digest(), before)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_reference_guard_rejects_wrong_known_score(self):
        with self.assertRaises(ValueError):
            check_reference({"task1_r_missing11": .6, "task2_r_missing11": .7},
                            {"task1_r_missing11": .7, "task2_r_missing11": .7}, "fixture")

    def test_seed43_reads_visible_I_metadata_not_hidden_targets(self):
        from baselines.B5_slow_pcc.core import model_inputs
        with tempfile.TemporaryDirectory() as temp:
            bundle, _, cache, manifest, raw, _, _ = setup(Path(temp))
            meta = encode_demographics({})
            class VisibleBase:
                def __init__(self, rows): self.rows = rows
                def __getitem__(self, index):
                    start = int(self.rows[index]["start_sample_500hz"])
                    return SimpleNamespace(anchor_i_ecg=raw[:1, start:start+5000])
            datasets = {}
            for task, rows in manifest["validation_rows"].items():
                datasets[task] = SimpleNamespace(rows=rows, base=VisibleBase(rows), manifest_digest=task,
                                                demographics=SimpleNamespace(get=lambda _: meta))
            study = {"data_config": {}, "manifests": {t: t for t in datasets}}
            with patch("baselines.B5_cross.run.preprocessing_config"), \
                 patch("baselines.B5_cross.run.checkpoint_preprocessor"), \
                 patch("baselines.B5_cross.run.build_validation_datasets", return_value=datasets):
                observed = noise_inputs(study, bundle, manifest["validation_rows"])
            self.assertEqual(len(observed), 1)
            np.testing.assert_array_equal(observed["target"][0], raw[:1])
            self.assertEqual(set(observed["target"][1]), set(model_inputs({k: None for k in
                             ("anchor", "numeric", "sex", "field_mask", "age_topcoded")})) - {"anchor"})

    def test_seed43_generated_FM_replaces_saved_predictions_and_is_shared_across_slows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, cache, manifest, raw, _, evaluation = setup(root)
            saved = SavedControl(evaluation, manifest["validation_rows"])
            generated = {"target": raw + 30 * np.cos(np.arange(60000) / 20)}
            learned = {"target": np.zeros((11, 60000))}
            try:
                with patch.object(saved, "record", side_effect=AssertionError("Seed43 must use new FM")):
                    score_combo("control100", "pcc20", learned, cache, manifest, saved, generated,
                                root / "seed43", dict(SAMPLING, seed=43))
                result = np.load(root / "seed43/task2/prediction_uV.npy")
                expected = combine(generated["target"], learned["target"], .75)
                for i in (0, 12):
                    np.testing.assert_array_equal(np.concatenate(result[i:i+12], axis=-1), expected)
                self.assertEqual(read(root / "seed43/evaluation.json")["sampling"]["seed"], 43)
            finally:
                saved.close()


if __name__ == "__main__":
    unittest.main()
