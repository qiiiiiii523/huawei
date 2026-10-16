from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from baselines.B5_best import SAMPLING
from baselines.B5_best.common import read, write, sha
from baselines.B5_best import diagnose
from baselines.B5_best.diagnostics_core import (
    corr, moving_average, qrs_candidates, local_shape, component_metrics, boundary_metrics)


def fixture(root):
    """Original evaluation + slow evaluation + B5-S50 QC, deliberately NO weights."""
    fm_dir = root / "old_B5_evaluation"
    slow_dir = root / "old_S50_F" / "evaluation_best_score"
    qc_dir = root / "old_B5_S50" / "control" / "evaluations" / "best_r"
    t = np.arange(60000) / 500
    signals = []
    for subject in range(2):
        phase = np.mod(t * (1.2 + subject * .05), 1)
        pulse = np.exp(-((phase - .15) / .02) ** 2) - .4 * np.exp(-((phase - .2) / .025) ** 2)
        y = np.stack([(100 + j * 20) * pulse + j * 20 * np.sin(2 * np.pi * .045 * t + j / 5)
                      for j in range(12)])
        p = .85 * y + np.arange(12)[:, None] * 50
        s = 1.15 * moving_average(y[1:]) + 20
        h = p.copy()
        h[1:] = p[1:] - moving_average(p[1:]) + s
        mask = np.ones(12, bool)
        if subject == 0:
            mask[9] = False  # V4 intentionally corrupted but excluded by QC.
            p[9] = np.sin(t * 17) * 100000
            h[9] = np.cos(t * 13) * 200000
        signals.append((y.astype(np.float32), p.astype(np.float32), h.astype(np.float32), mask))
    files = {}
    for task in ("task1", "task2"):
        specifications = ([(0, "ecg_machine_d6"), (0, "ecg_machine_d6"), (1, "ecg_machine_d6")]
                          if task == "task1" else
                          [(0, "ecg_machine_d6"), (1, "ecg_machine_d6"), (0, "body_scale_d6"), (1, "body_scale_d6")])
        rows, ys, ps, hs, masks = [], [], [], [], []
        for pair, (subject, device) in enumerate(specifications):
            y, p, h, mask = signals[subject]
            for start in range(0, 60000, 5000):
                rows.append({"pair_id": f"{task}_{pair}", "target_record_id": f"target_{subject}",
                             "subject_id": f"subject_{subject}", "input_type": device,
                             "start_sample_500hz": str(start), "expected_window_count": "12"})
                ys.append(y[:, start:start + 5000])
                ps.append(p[:, start:start + 5000])
                hs.append(h[:, start:start + 5000])
                masks.append(mask)
        a, b, c = fm_dir / task, slow_dir / "learned_slow_hybrid" / task, qc_dir / task
        for d in (a, b, c):
            d.mkdir(parents=True)
        np.save(a / "target_uV.npy", np.stack(ys))
        np.save(a / "prediction_uV.npy", np.stack(ps))
        np.save(b / "prediction_uV.npy", np.stack(hs))
        diagnose.csv_save(a / "window_metadata.csv", rows)
        diagnose.csv_save(c / "window_metadata.csv", rows)
        np.save(c / "target_quality_mask.npy", np.stack(masks))
        write(c / "target_reference.json", {"path": str(a / "target_uV.npy"), "sha256": sha(a / "target_uV.npy")})
        for name in ("target_uV.npy", "prediction_uV.npy", "window_metadata.csv"):
            files[task + "/" + name] = sha(a / name)
    info_path = root / "prepared.json"
    write(info_path, {"evaluation_dir": str(fm_dir), "evaluation_files": files,
                     "checkpoint_sha256": "original_fm_digest"})
    write(slow_dir / "evaluation.json", {"completed": True, "group": "F", "context_seconds": 50,
          "target_used_at_inference": False, "prepared_sha256": sha(info_path),
          "B5_checkpoint_sha256": "original_fm_digest", "probe_checkpoint_sha256": "slow_f_digest"})
    tasks = {name: diagnose.load_task(name, fm_dir, slow_dir / "learned_slow_hybrid", qc_dir, True)
             for name in ("task1", "task2")}
    profile = {"fm_sha256": "original_fm_digest", "slow_sha256": "slow_f_digest", "reference_tolerance": 2e-5,
               "reference_qc_r": {name + "_r_missing11": diagnose.task_r(task, task.hybrid)
                                  for name, task in tasks.items()}}
    args = SimpleNamespace(repository_root=str(root), prepared_info=str(info_path), fm_evaluation_dir=None,
                           slow_evaluation_dir=str(slow_dir), integration_root=str(root / "not_created"),
                           qc_evaluation_dir=str(qc_dir), output=str(root / "diagnosis"), check_only=False,
                           leads=["V3"], no_plots=False, plot_records=1)
    for task in tasks.values():
        task.close()
    return args, None, profile


class CoreDiagnosticsTests(unittest.TestCase):
    def test_global_offset_scale_invariant_but_relative_slow_scale_matters(self):
        t = np.arange(60000) / 500
        fast = np.sin(2 * np.pi * 5 * t)
        slow = 2 * np.sin(2 * np.pi * .05 * t)
        target = fast + slow
        self.assertAlmostEqual(corr(4 * target + 999, target), 1)
        self.assertLess(corr(fast + 2 * slow, target), .99)
        self.assertAlmostEqual(corr(fast + slow, target), 1)

    def test_filter_full_record_not_each_window(self):
        x = np.zeros(60000)
        x[5000:] = 1
        s = moving_average(x)
        self.assertGreater(s[4999], 0)
        self.assertLess(s[5000], 1)
        np.testing.assert_allclose(x - s + s, x)

    def test_known_amplitude_and_timing_errors_are_detected_without_correcting_output(self):
        x = np.arange(3000)
        anchors = np.array([500, 1500, 2500])
        target = sum(100 * np.exp(-.5 * ((x - q) / 4) ** 2) for q in anchors)
        prediction = sum(150 * np.exp(-.5 * ((x - q - 6) / 4) ** 2) for q in anchors)
        original = prediction.copy()
        r = local_shape(prediction, target, anchors)
        self.assertEqual(r["n_eligible_beats"], 3)
        self.assertAlmostEqual(r["abs_peak_time_error_ms_median"], 12)
        self.assertAlmostEqual(r["peak_amplitude_ratio_median"], 1.5, places=5)
        self.assertGreater(r["qrs_r_best_lag_DIAGNOSTIC_ONLY_median"], r["qrs_r_unshifted_median"])
        np.testing.assert_array_equal(original, prediction)

    def test_empty_or_flat_I_does_not_invent_beats(self):
        self.assertEqual(len(qrs_candidates(np.zeros(60000))), 0)
        r = local_shape(np.zeros(60000), np.zeros(60000), [])
        self.assertEqual(r["n_eligible_beats"], 0)
        self.assertIsNone(r["qrs_r_unshifted_median"])

    def test_boundary_diagnostic_detects_steps_and_not_constant_record_bias(self):
        y = np.sin(np.arange(60000) / 20)
        p = y + (np.arange(60000) // 5000) % 2 * 100
        self.assertGreater(boundary_metrics(p, y)["seam_error_one_sample_jump_mae_uV"], 90)
        self.assertLess(boundary_metrics(y + 800, y)["centered_error_rms_near_seam_uV"], 1e-9)

    def test_metadata_alignment_is_by_identity_and_rejects_duplicates(self):
        a = [{"pair_id": "a", "target_record_id": "t", "start_sample_500hz": str(i),
              "subject_id": "s", "input_type": "ecg_machine_d6"} for i in (0, 5000)]
        np.testing.assert_array_equal(diagnose.align(a, a[::-1]), [1, 0])
        with self.assertRaises(ValueError):
            diagnose.align(a, [a[0], a[0]])


class FileDiagnosticsTests(unittest.TestCase):
    def test_existing_integration_evaluation_is_scored_and_optional_missing_arrays_are_marked(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, profile = fixture(Path(temp))
            fm = Path(temp) / "old_B5_evaluation"
            qc = Path(args.qc_evaluation_dir)
            slow = Path(args.slow_evaluation_dir) / "learned_slow_hybrid"
            with diagnose.ExitStack() as resources:
                tasks = {}
                for name in ("task1", "task2"):
                    tasks[name] = diagnose.load_task(name, fm, slow, qc, True)
                    resources.callback(tasks[name].close)
                    source = np.load(fm / name / "prediction_uV.npy", allow_pickle=False)
                    np.save(qc / name / "prediction_uV.npy", source)
                write(qc / "evaluation.json", {"completed": True, "fingerprint": {"sampling": SAMPLING}})
                rows = diagnose.integration_report(qc.parents[2], tasks)
                self.assertEqual(rows[0]["status"], "recomputed_from_existing_arrays_same_QC")
                self.assertAlmostEqual(rows[0]["task1_r"], diagnose.task_r(tasks["task1"], tasks["task1"].fm))
                (qc / "task1/prediction_uV.npy").unlink()
                rows = diagnose.integration_report(qc.parents[2], tasks)
                self.assertEqual(rows[0]["status"], "optional_arrays_missing_not_regenerated")

    def test_end_to_end_reads_legacy_paths_QC_reference_and_writes_lightweight_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, tasks, profile = fixture(root)
            # Deployment reference may now differ; the saved source remains beta=1.
            profile["reference_qc_r_beta1"] = dict(profile["reference_qc_r"])
            profile["reference_qc_r"] = {k: 0.123 for k in profile["reference_qc_r"]}
            source_hashes = {p: sha(p) for p in root.rglob("*.npy")}
            original_read = diagnose.read
            def mocked_read(path):
                return profile if Path(path).name == "b5_best.json" else original_read(path)
            with patch.object(diagnose, "read", side_effect=mocked_read):
                diagnose.run(args)
            out = Path(args.output)
            marker = read(out / "diagnostics.json")
            self.assertTrue(marker["completed"])
            self.assertFalse(marker["training_started"])
            self.assertFalse(marker["inference_started"])
            self.assertFalse(marker["models_loaded"])
            self.assertEqual(marker["n_unique_diagnostic_targets"], 2)
            rows = diagnose.csv_read(out / "component_record_lead.csv")
            self.assertFalse(any(r["target_record_id"] == "target_0" and r["lead"] == "V4" for r in rows))
            b = next(r for r in diagnose.csv_read(out / "beta_scores.csv") if float(r["beta"]) == 1)
            self.assertAlmostEqual(float(b["task1_r"]), profile["reference_qc_r_beta1"]["task1_r_missing11"])
            self.assertTrue(list((out / "plots").glob("*.png")))
            self.assertFalse(list(out.rglob("*.npy")))
            self.assertFalse(list(root.rglob("*.pt")))
            self.assertFalse((root / "results" / "B5-best").exists())
            for p, digest in source_hashes.items():
                self.assertEqual(sha(p), digest)

    def test_target_reference_and_QC_mismatch_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            args, tasks, profile = fixture(Path(temp))
            mask_path = Path(args.qc_evaluation_dir) / "task1/target_quality_mask.npy"
            mask = np.load(mask_path)
            mask[0, 3] = False
            np.save(mask_path, mask)
            with self.assertRaisesRegex(ValueError, "QC changes"):
                diagnose.load_task("task1", Path(args.prepared_info).parent / "old_B5_evaluation",
                                   Path(args.slow_evaluation_dir) / "learned_slow_hybrid",
                                   Path(args.qc_evaluation_dir), True)
            ref = Path(args.qc_evaluation_dir) / "task2/target_reference.json"
            r = read(ref)
            r["sha256"] = "changed"
            write(ref, r)
            with self.assertRaisesRegex(ValueError, "Target reference"):
                diagnose.target_file(ref.parent)

    def test_incomplete_window_sequence_is_rejected(self):
        rows = [{"pair_id": "p", "target_record_id": "t", "subject_id": "s", "input_type": "ecg_machine_d6",
                 "start_sample_500hz": str(i)} for i in range(0, 55000, 5000)]
        with self.assertRaisesRegex(ValueError, "twelve"):
            diagnose.pairs(rows)


if __name__ == "__main__":
    unittest.main()
