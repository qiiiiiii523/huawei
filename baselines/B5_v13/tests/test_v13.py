"""Synthetic CPU functionality checks; no training loop or optimizer steps."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from baselines.B5.config import ModelConfig
from baselines.B5.flow import sample
from baselines.B5.losses import flow_loss
from baselines.B5.metadata import encode_demographics
from baselines.B5.model import B5UNet
from baselines.B5_best import SAMPLING, SCORE_PROTOCOL, BUNDLE_PROTOCOL
from baselines.B5_best.common import write, sha, read
from baselines.B5_best.tests.test_hybrid import fixture
from baselines.B5_best.bundle import export_bundle, load_bundle
from baselines.B5.checkpoint import atomic_save
from baselines.B5_v13 import PROTOCOL, ARCHITECTURE_ID
from baselines.B5_v13.common import code_digest, load_study, load_experiment_checkpoint
from baselines.B5_v13.evaluation import ExperimentalPredictor
from baselines.B5_v13.model import B5V13, LocalV13, KERNELS, initialized_model
from baselines.B5_v13.prepare import zero_check
from baselines.B5_v13.compare import compare
from baselines.B5_best.validate import evaluate as score_arrays

torch.set_num_threads(2)


def small_bundle():
    config = ModelConfig(channels=(8, 16, 32), time_dim=16, metadata_dim=16)
    base = B5UNet(config)
    return {"fm": {"model_config": asdict(config), "ema": copy.deepcopy(base.state_dict())}}


def batch(length=64):
    fields = {k: torch.from_numpy(v[None]) for k, v in encode_demographics({"age": 45}).items()}
    return {"anchor": torch.randn(1, 1, length), "target": torch.randn(1, 12, length),
            "quality_mask": torch.ones(1, 12, dtype=torch.bool), **fields}


class V13Tests(unittest.TestCase):
    def test_zero_initialization_full_5000_window_all_times(self):
        result = zero_check(small_bundle())
        self.assertEqual(result["velocity_max_abs_difference"], 0.)
        self.assertGreater(result["new_branch_parameters"], 0)

    def test_only_v1_v3_receive_direct_velocity_correction(self):
        branch = LocalV13(8)
        with torch.no_grad():
            branch.project.bias.copy_(torch.tensor([2., -3.]))
        output = branch(torch.randn(2, 8, 77))
        self.assertTrue(torch.all(output[:, 5] == 2))
        self.assertTrue(torch.all(output[:, 7] == -3))
        others = [i for i in range(11) if i not in (5, 7)]
        self.assertTrue(torch.all(output[:, others] == 0))

    def test_original_flow_loss_propagates_gradient_into_new_head(self):
        model = initialized_model(small_bundle(), "v13", torch.device("cpu")).train()
        data = batch()
        # An explicitly bad V3 label must be safely excluded from the original FM objective.
        data["quality_mask"][:, 8] = False
        data["target"][:, 8] = float('nan')
        weights = {"huber": .1, "pcc": .1, "anchor": .02, "physiology": 0., "huber_delta": 1.}
        loss, _ = flow_loss(model, data, weights, torch.ones(12),
                            noise=torch.zeros(1, 11, 64), time=torch.tensor([.5]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        grad = model.local_v13.project.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad[0].abs().sum()), 0.)
        self.assertEqual(float(grad[1].abs().sum()), 0.)

    def test_ode_zero_branch_reproduces_baseline_without_target_input(self):
        bundle = small_bundle()
        a = initialized_model(bundle, "control", torch.device("cpu")).eval()
        b = initialized_model(bundle, "v13", torch.device("cpu")).eval()
        data = batch(40)
        inputs = {k: v for k, v in data.items() if k not in ("target", "quality_mask")}
        out_a = sample(a, inputs, ["record:0"], seed=42, steps=2, solver="heun", samples=2)
        out_b = sample(b, inputs, ["record:0"], seed=42, steps=2, solver="heun", samples=2)
        torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)

    def test_strict_initializer_rejects_unrelated_missing_keys(self):
        bundle = small_bundle()
        del bundle["fm"]["ema"]["output.weight"]
        with self.assertRaises(ValueError):
            initialized_model(bundle, "v13", torch.device("cpu"))

    def test_shared_and_experimental_checkpoint_roundtrip_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp, sp, ip, profile = fixture(root)
            bundle = export_bundle(fp, sp, ip, root / "components.pt", profile)
            study_path = root / "study.json"
            config = {"paths": {"preprocessing_config": str(Path(__file__).resolve().parents[3] /
                                                              "configs/preprocessing.yaml")},
                      "repository_root": str(root),
                      "loss": {"huber": .1, "pcc": .1, "anchor": .02, "physiology": 0., "huber_delta": 1.}}
            write(study_path, {"protocol": PROTOCOL, "score_protocol": SCORE_PROTOCOL,
                  "sampling": dict(SAMPLING), "beta": .75, "code_sha256": code_digest(),
                  "bundle_sha256": sha(root / "components.pt"), "data_config": config})
            model = initialized_model(bundle, "v13", torch.device("cpu"))
            cp = {"protocol": PROTOCOL, "score_protocol": SCORE_PROTOCOL,
                  "study_sha256": sha(study_path), "sampling_config": dict(SAMPLING),
                  "group": "v13", "architecture_id": ARCHITECTURE_ID, "kernels": list(KERNELS),
                  "model": model.state_dict(), "ema": model.state_dict()}
            atomic_save(root / "best.pt", cp)
            _, _, loaded, _, restored = load_experiment_checkpoint(
                root / "best.pt", study_path, torch.device("cpu"), verify_data=False)
            engine = ExperimentalPredictor(loaded, torch.device("cpu"), restored)
            self.assertEqual(engine.slow_beta, .75)
            self.assertFalse(any(p.requires_grad for p in engine.slow.parameters()))
            # Keep the end-to-end 120s predictor, interpolation and combination; mock only heavy FM sampling.
            def fake_sample(model, inputs, keys, **settings):
                self.assertIsInstance(model, B5V13)
                self.assertEqual(settings, dict(SAMPLING))
                return torch.zeros(len(keys), 12, 5000)
            with patch('baselines.B5_best.predict.sample', side_effect=fake_sample):
                output = engine.predict(np.zeros((1, 60000), np.float32), {}, "new_patient")
                engine.slow_beta = 1.
                beta1_output = engine.predict(np.zeros((1, 60000), np.float32), {}, "new_patient")
                engine.slow_beta = .75
            self.assertEqual(output.shape, (12, 60000))
            self.assertTrue(np.isfinite(output).all())
            # Mock FM is zero: this checks the ACTUAL output coefficient, not just report metadata.
            np.testing.assert_allclose(output[1:], .75 * beta1_output[1:], rtol=2e-6, atol=2e-5)
            cp["group"] = "control"
            atomic_save(root / "wrong.pt", cp)
            with self.assertRaises(ValueError):
                load_experiment_checkpoint(root / "wrong.pt", study_path, torch.device("cpu"), verify_data=False)

    def test_old_beta_one_study_is_rejected_before_reading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'study.json'
            write(path, {'protocol': PROTOCOL, 'score_protocol': SCORE_PROTOCOL,
                         'sampling': dict(SAMPLING), 'beta': 1.})
            with self.assertRaisesRegex(ValueError, 'beta=0.75'):
                load_study(path, verify_data=False)

    def test_saved_paired_scores_and_morphology_comparison(self):
        phase = np.arange(60000)
        raw = np.tile((100 * np.sin(phase / 25)).astype(np.float32), (12, 1))
        class Base:
            def __init__(self, rows): self.rows = rows
            def __getitem__(self, i):
                s = int(self.rows[i]['start_sample_500hz'])
                return SimpleNamespace(anchor_i_ecg=raw[:1, s:s + 5000])
        class Dataset:
            def __init__(self, task):
                devices = ['watch_ecg'] if task == 'task1' else ['ecg_machine_d6', 'body_scale_d6']
                self.rows = [{'pair_id': d, 'target_record_id': 'record', 'subject_id': 'patient',
                              'start_sample_500hz': str(s), 'input_type': d, 'expected_window_count': '12'}
                             for d in devices for s in range(0, 60000, 5000)]
                self.base = Base(self.rows)
                self.demographics = SimpleNamespace(get=lambda _: encode_demographics({}))
            def __len__(self): return len(self.rows)
            def __getitem__(self, i):
                s = int(self.rows[i]['start_sample_500hz'])
                return {'target_uV': raw[:, s:s + 5000], 'quality_mask': np.ones(12, bool),
                        'evaluation_metadata': self.rows[i]}
        class Engine:
            def __init__(self, error): self.error = error
            def predict_encoded(self, anchor, metadata, key, batch_size):
                output = raw.copy()
                output[[6, 8]] += self.error * np.sin(phase / 9)
                return output
        bundle = {'bundle_protocol': BUNDLE_PROTOCOL, 'slow_beta': .75,
                  'sampling': dict(SAMPLING), 'reference_qc_r': None,
                  'reference_tolerance': 2e-5, 'provenance': {}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group, error in [('control', 30), ('v13', 10)]:
                folder = root / group / 'evaluation'
                score = score_arrays(bundle, {t: Dataset(t) for t in ['task1', 'task2']},
                                     torch.device('cpu'), folder, 'synthetic', predictor=Engine(error))
                write(folder / 'experiment.json', {'group': group, 'epoch': 20, 'score': score})
                write(root / group / 'initialization.json', {'study_sha256': 'same-study'})
                for choice in ('best', 'last'):
                    (root / group / f'latest_{choice}_evaluation.txt').write_text(str(folder))
            compare(root, root / 'comparison', plots=False)
            result = read(root / 'comparison/comparison.json')
            self.assertGreater(result['best_delta_mean_r'], 0)
            self.assertTrue((root / 'comparison/morphology_summary.csv').exists())


if __name__ == "__main__":
    unittest.main()
