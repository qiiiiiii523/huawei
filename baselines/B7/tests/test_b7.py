from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from baselines.B7 import ARCHITECTURE_ID, CHECKPOINT_VERSION, GENERATED_LEADS
from baselines.B7.checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint
from baselines.B7.config import ModelConfig, LossConfig, SamplingConfig
from baselines.B7.data import DATA_PROTOCOL, WindowDataset, read_manifest
from baselines.B7.evaluate import aggregate, pearson
from baselines.B7.flow import integrate, keyed_noise, sample
from baselines.B7.inference import predict_record
from baselines.B7.leads import compose_d12
from baselines.B7.losses import flow_loss, reconstruction_losses
from baselines.B7.model import B7UNet
from baselines.B7.package import export
from baselines.B7.prepare import prepare, read_wfdb
from baselines.B7 import train as trainer


def small_config():
    return ModelConfig(channels=(8, 16, 32), time_dim=8, metadata_dim=8, dilations=(1,))


class RecordingVelocity(nn.Module):
    def __init__(self, velocity=0.):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(velocity))
        self.observed = []

    def forward(self, state, time, anchor, metadata=None):
        self.observed.append(anchor.clone())
        return torch.ones_like(state) * self.marker


def payload(model, sampling=None):
    return {"architecture_id": ARCHITECTURE_ID, "format_version": CHECKPOINT_VERSION,
            "generated_leads": GENERATED_LEADS, "model_config": model.config.to_dict(),
            "ema": model.state_dict(), "scales_uV": [100.] * 12,
            "sampling": asdict(sampling or SamplingConfig(steps=1, samples=1)),
            "loss": asdict(LossConfig()), "stage": "public10", "update": 0, "best_r": 0.,
            "initialization": {"kind": "random_B7", "B5_weights_used": False}}


def make_manifest(root):
    records = []
    t = np.arange(5000, dtype="float32") / 500
    for split in ("train", "validation"):
        target = np.stack([100 * np.sin(t * (2 + j / 10)) + j for j in range(12)])
        f = root / (split + ".npz")
        np.savez(f, anchor_uV=target[:1], target_uV=target, lead_mask=np.ones(12, dtype=bool))
        records.append({"record_id": split, "subject_id": split, "target_record_id": split,
                        "source": "synthetic", "group": "synthetic", "path": f.name,
                        "split": split, "input_reliable": True})
    path = root / "manifest.json"
    path.write_text(json.dumps({"protocol": DATA_PROTOCOL, "sample_rate": 500, "scales_uV": [100.] * 12,
                               "records": records}), encoding="utf-8")
    return path


class B7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default40_shape_direct_raw_I_and_seven_output(self):
        torch.manual_seed(3)
        model = B7UNet().eval()
        anchor = torch.linspace(-2, 2, 20000).reshape(1, 1, -1)
        original = anchor.clone()
        state = torch.randn(1, 7, 20000)
        observed = []
        hook = model.stem.register_forward_pre_hook(lambda m, args: observed.append(args[0].clone()))
        with torch.no_grad():
            prediction = model(state, torch.tensor([.5]), anchor)
        hook.remove()
        self.assertEqual(prediction.shape, (1, 7, 20000))
        self.assertTrue(torch.equal(observed[0][:, :1], original))
        self.assertTrue(torch.equal(observed[0][:, 1:], state))
        self.assertTrue(torch.equal(anchor, original))
        self.assertFalse(any("anchor_" in k or "condition_enc" in k for k in model.state_dict()))

    def test_same_network_supports_short_window(self):
        model = B7UNet(small_config()).eval()
        with torch.no_grad():
            self.assertEqual(model(torch.randn(2, 7, 5000), torch.tensor([0., 1.]), torch.randn(2, 1, 5000)).shape,
                             (2, 7, 5000))

    def test_heun_never_changes_I(self):
        model = RecordingVelocity(2.).eval()
        q = torch.randn(1, 1, 120)
        z = torch.randn(1, 7, 120)
        result = integrate(model, q, z, steps=4)
        torch.testing.assert_close(result, z + 2)
        self.assertEqual(len(model.observed), 8)
        self.assertTrue(all(torch.equal(x, q) for x in model.observed))
        self.assertEqual(q.shape[1], 1)

    def test_K16_is_actual_mean_and_noise_stable_by_key(self):
        model = RecordingVelocity(0.).eval()
        q = torch.randn(2, 1, 40)
        sc = SamplingConfig(samples=16, steps=1)
        expected = torch.stack([keyed_noise(["a", "b"], 40, 42, k, "cpu") for k in range(16)]).mean(0)
        torch.testing.assert_close(sample(model, q, ["a", "b"], config=sc), expected)
        torch.testing.assert_close(keyed_noise(["a", "b"], 40, 42, 0, "cpu").flip(0),
                                   keyed_noise(["b", "a"], 40, 42, 0, "cpu"))

    def test_edge_affects_MAE_but_not_central_PCC(self):
        y = torch.sin(torch.arange(20000) / 23)[None, None].repeat(1, 7, 1)
        valid = torch.ones_like(y, dtype=torch.bool)
        p = y.clone()
        p[:, :, :7500] += 10
        p[:, :, 12500:] -= 10
        parts = reconstruction_losses(p, y, valid)
        self.assertAlmostEqual(float(parts["mae"]), 7.5, places=5)
        self.assertLess(abs(float(parts["pcc"])), 1e-6)
        p[:, :, 7500:12500] = -y[:, :, 7500:12500]
        self.assertAlmostEqual(float(reconstruction_losses(p, y, valid)["pcc"]), 2., places=5)

    def test_II_weight_normalized_and_constant_failed_prediction_counted(self):
        y = torch.sin(torch.arange(5000) / 19)[None, None].repeat(1, 7, 1)
        valid = torch.ones_like(y, dtype=torch.bool)
        p = y.clone()
        p[:, 0] += 1
        self.assertAlmostEqual(float(reconstruction_losses(p, y, valid)["mae"]), 5 / 11, places=6)
        parts = reconstruction_losses(torch.zeros_like(y), y, valid)
        self.assertEqual(int(parts["pcc_valid_leads"]), 7)
        self.assertAlmostEqual(float(parts["pcc"]), 1.)

    def test_invalid_nan_and_padding_excluded_from_losses(self):
        y = torch.sin(torch.arange(5000) / 19)[None, None].repeat(1, 7, 1)
        valid = torch.ones_like(y, dtype=torch.bool)
        valid[:, 2] = False
        valid[:, :, :300] = False
        p = y.clone()
        p[~valid] = float("nan")
        y[~valid] = float("nan")
        parts = reconstruction_losses(p, y, valid)
        self.assertAlmostEqual(float(parts["mae"]), 0.)
        self.assertTrue(torch.isfinite(parts["pcc"]))

    def test_flow_backward_uses7_and_no_target_I(self):
        model = B7UNet(small_config())
        q = torch.randn(1, 1, 5000)
        original = q.clone()
        batch = {"anchor": q, "target": torch.randn(1, 7, 5000), "lead_mask": torch.ones(1, 7, dtype=torch.bool)}
        loss, parts = flow_loss(model, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.stem.weight.grad).all())
        self.assertTrue(torch.equal(q, original))
        self.assertEqual(set(parts), {"fm", "mae", "pcc", "mae_ii", "mae_chest", "pcc_valid_leads"})

    def test_record_stitch_copy_I_and_physical_lead_formulas(self):
        model = RecordingVelocity(0.).eval()
        anchor = torch.arange(60000).float()[None] / 10
        prediction = predict_record(model, anchor, [2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
                                    "record", config=SamplingConfig(samples=1, steps=1))
        self.assertEqual(prediction.shape, (12, 60000))
        self.assertEqual(len(model.observed), 24)  # 12 cores, two Heun evaluations.
        self.assertTrue(torch.equal(prediction[:1], anchor))
        torch.testing.assert_close(prediction[2], prediction[1] - anchor[0])
        torch.testing.assert_close(prediction[3], -(prediction[1] + anchor[0]) / 2)
        torch.testing.assert_close(prediction[4], anchor[0] - prediction[1] / 2)
        torch.testing.assert_close(prediction[5], prediction[1] - anchor[0] / 2)
        self.assertTrue(torch.equal(model.observed[0][0, 0, :7500], anchor[0, :1].expand(7500) / 2))

    def test_compose_d12_I_is_original(self):
        q = torch.tensor([[[3., 4.]]])
        y = torch.arange(14).reshape(1, 7, 2).float()
        a = compose_d12(q, y)
        self.assertTrue(torch.equal(a[:, :1], q))
        self.assertTrue(torch.equal(a[:, 6:], y[:, 1:]))

    def test_reject_B5_checkpoint_and_export_strict_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            torch.save({"architecture_id": "B5-U"}, root / "old.pt")
            with self.assertRaisesRegex(ValueError, "B5"):
                load_checkpoint(root / "old.pt")
            model = B7UNet(small_config()).eval()
            save_checkpoint(root / "b7.pt", payload(model))
            export(root / "b7.pt", root / "inference.pt")
            value = load_checkpoint(root / "inference.pt")
            self.assertNotIn("optimizer", value)
            other = model_from_checkpoint(value)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, other.state_dict()[k]))

    def test_patient_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_manifest(Path(d))
            m = json.loads(path.read_text())
            m["records"][1]["subject_id"] = "train"
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "Patient"):
                read_manifest(path)

    def test_train_random_start_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = make_manifest(root)
            config = {"stage": "public10", "model": small_config().to_dict(), "loss": asdict(LossConfig()),
                      "sampling": asdict(SamplingConfig(window_seconds=10, steps=1, samples=1)),
                      "training": {"seed": 4, "updates": 2, "validate_every": 1, "batch_size": 1,
                                   "learning_rate": .0001, "weight_decay": .01, "ema_decay": .9,
                                   "gradient_clip": 1., "boundary_probability": 0.}}
            cp = root / "cfg.json"
            cp.write_text(json.dumps(config))
            args = Namespace(config=str(cp), manifest=str(manifest), output=str(root / "full"),
                             device="cpu", resume=False, init_checkpoint=None)
            trainer.train(args)
            original = trainer.evaluate
            calls = 0
            def interrupt(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("simulated interruption")
                return original(*a, **kw)
            args.output = str(root / "interrupted")
            with patch.object(trainer, "evaluate", interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    trainer.train(args)
            args.resume = True
            trainer.train(args)
            a, b = load_checkpoint(root / "full/last.pt"), load_checkpoint(root / "interrupted/last.pt")
            best = load_checkpoint(root / "full/best.pt")
            best_r = load_checkpoint(root / "full/best_r.pt")
            self.assertEqual(best["best_r"], best_r["best_r"])
            self.assertEqual(best["update"], best_r["update"])
            for key in best["ema"]:
                self.assertTrue(torch.equal(best["ema"][key], best_r["ema"][key]))
            history = [json.loads(line) for line in (root / "full/history.jsonl").read_text().splitlines()]
            self.assertAlmostEqual(history[-1]["train"]["window_r"], 1 - history[-1]["train"]["pcc"])
            self.assertEqual(a["initialization"]["kind"], "random_B7")
            self.assertFalse(a["initialization"]["B5_weights_used"])
            self.assertEqual(a["best_r"], b["best_r"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)

    def test_evaluation_matches_undefined_rule_and_group_weights(self):
        self.assertTrue(np.isnan(pearson(np.ones(10), np.arange(10))))
        rows = [{"group": g, "target_record_id": "a", "r": score,
                 "leads": {str(j): {"r": score} for j in range(1, 12)}} for g, score in
                (("task1", .8), ("machine", .4), ("body", .2))]
        self.assertAlmostEqual(aggregate(rows, "huawei40")["r"], .55)

    def test_QC_lead_macro_is_not_pair_available_lead_macro(self):
        rows = [
            {"group": "public", "target_record_id": "a", "leads": {str(j): {"r": 1.} for j in range(1, 12)}},
            {"group": "public", "target_record_id": "b", "leads": {"1": {"r": 0.}}}
        ]
        self.assertAlmostEqual(aggregate(rows, "public10")["r"], 10.5 / 11)

    def test_QC_score_matches_existing_scorer_with_repeated_pairs(self):
        from ecg12gen.evaluate import evaluate_record_predictions
        rng = np.random.default_rng(8)
        time = np.arange(60000) / 500
        a = np.stack([np.sin(time * (j + 2)) for j in range(12)])
        b = np.stack([np.cos(time * (j + 3)) for j in range(12)])
        qc_a, qc_b = np.ones(12, dtype=bool), np.ones(12, dtype=bool)
        qc_b[2:6] = False
        predictions, targets, qualities, metadata, rows = [], [], [], [], []
        for i, (target_id, y, qc) in enumerate((("a", a, qc_a), ("a", a, qc_a), ("b", b, qc_b))):
            p = y + (i + 1) * .3 * rng.normal(size=y.shape)
            leads = {str(j): {"r": pearson(p[j], y[j])} for j in range(1, 12) if qc[j]}
            rows.append({"group": "public", "target_record_id": target_id, "leads": leads})
            for k in range(12):
                predictions.append(p[:, k*5000:(k+1)*5000])
                targets.append(y[:, k*5000:(k+1)*5000])
                qualities.append(qc)
                metadata.append({"pair_id": f"pair{i}", "target_record_id": target_id,
                                 "subject_id": target_id, "start_sample_500hz": str(k * 5000),
                                 "expected_window_count": "12"})
        existing, _ = evaluate_record_predictions(np.stack(predictions), np.stack(targets), "task1", metadata,
                                                   target_quality_mask=np.stack(qualities))
        self.assertAlmostEqual(aggregate(rows, "public10")["r"], existing["r_missing11"], places=12)

    def test_same_target_cannot_cross_patient_or_split(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_manifest(Path(d))
            manifest = json.loads(path.read_text())
            manifest['records'][1]['target_record_id'] = 'train'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Physical target'):
                read_manifest(path)

    def test_prepare_rejects_nonbinary_QC_before_writing_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = make_manifest(root)
            index = json.loads(path.read_text())['records']
            index[0]['lead_mask'] = [0.5] * 12
            ip = root/'bad_index.json'
            ip.write_text(json.dumps(index))
            args = Namespace(index=str(ip), output=str(root/'out'), public_source=None, root=None, scale_manifest=None)
            with self.assertRaisesRegex(ValueError, 'boolean/0-1'):
                prepare(args)
            self.assertFalse((root/'out').exists())

    def test_undefined_eval_pair_uses_established_skip_within_target_rule(self):
        rows = [
            {'group':'public', 'target_record_id':'a', 'leads':{str(j):{'r':None} for j in range(1,12)}},
            {'group':'public', 'target_record_id':'a', 'leads':{str(j):{'r':0.7} for j in range(1,12)}}
        ]
        self.assertAlmostEqual(aggregate(rows, 'public10')['r'], 0.7)
        with self.assertRaisesRegex(ValueError, 'No defined QC-eligible'):
            aggregate(rows[:1], 'public10')

    def test_full120_Heun16_K16_has_expected_calls_and_only7_states(self):
        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = nn.Parameter(torch.tensor(0.))
                self.calls = 0
                self.q = None

            def forward(self, state, time, anchor, metadata=None):
                if self.calls % 512 == 0:
                    self.q = anchor.clone()
                if state.shape != (1,7,20000) or not torch.equal(anchor,self.q):
                    raise AssertionError('I changed or seven-channel40s state violated')
                self.calls += 1
                return torch.zeros_like(state)
        model = Counter().eval()
        q = torch.sin(torch.arange(60000)/23)[None]
        output = predict_record(model, q, [100.]*12, 'full_K16', config=SamplingConfig())
        self.assertEqual(model.calls, 12 * 16 * 16 * 2)
        self.assertTrue(torch.equal(output[:1],q))
        self.assertEqual(output.shape,(12,60000))

    def test_prepare_uses_train_only_scale_and_freezes_for_next_stage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = make_manifest(root)
            index = json.loads(manifest.read_text())["records"]
            with np.load(root / "validation.npz") as a:
                np.savez(root / "val_large.npz", anchor_uV=a["anchor_uV"] * 1000,
                         target_uV=a["target_uV"] * 1000, lead_mask=a["lead_mask"])
            index[1]["path"] = "val_large.npz"
            ip = root / "index.json"
            ip.write_text(json.dumps(index))
            args = Namespace(index=str(ip), output=str(root / "prepared"), public_source=None, root=None, scale_manifest=None)
            prepare(args)
            prepared = read_manifest(root / "prepared/manifest.json")
            self.assertTrue(all(100 <= s < 300 for s in prepared["scales_uV"]))
            args.output, args.scale_manifest = str(root / "next"), str(root / "prepared/manifest.json")
            prepare(args)
            self.assertEqual(prepared["scales_uV"], read_manifest(root / "next/manifest.json")["scales_uV"])

    def test_wfdb_multifile_and_unsigned_checksum(self):
        from baselines.B7 import D12_LEADS
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = (np.arange(5000 * 12).reshape(5000, 12) % 1000 + 200).astype('<i2')
            (root / 'sample.dat').write_bytes(raw.tobytes())
            (root / 'sample.xyz').write_bytes(np.zeros((5000, 3), dtype='<i2').tobytes())
            lines = ['sample 15 500 5000']
            for j, label in enumerate(D12_LEADS):
                cs = int(raw[:, j].sum(dtype='int64')) % 65536
                lines.append(f'sample.dat 16 1000/mV 16 0 {raw[0,j]} {cs} 0 {label}')
            for label in ('vx', 'vy', 'vz'):
                lines.append(f'sample.xyz 16 1000/mV 16 0 0 0 0 {label}')
            (root/'sample.hea').write_text('\n'.join(lines))
            signal, fs, reader = read_wfdb(root/'sample.hea')
            np.testing.assert_allclose(signal, raw.T.astype('float32'))
            self.assertEqual(fs, 500)
            self.assertEqual(reader, 'WFDB_multifile_standard12_selected')

    def test_40s_backward_and_packaged_prediction_agree(self):
        torch.manual_seed(7)
        model = B7UNet(small_config())
        q = torch.sin(torch.arange(20000) / 40)[None, None]
        target = q.repeat(1, 7, 1) + torch.arange(7)[None, :, None] / 10
        batch = {"anchor": q, "target": target, "lead_mask": torch.ones(1, 7, dtype=torch.bool)}
        opt = torch.optim.AdamW(model.parameters(), lr=.0001)
        loss, _ = flow_loss(model, batch)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        opt.step()
        model.eval()
        sampling = SamplingConfig(samples=2, steps=1)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            save_checkpoint(root/'trained.pt', payload(model, sampling))
            export(root/'trained.pt', root/'inference.pt')
            loaded = model_from_checkpoint(load_checkpoint(root/'inference.pt'))
            anchor = q[0, :, :5000] * 100
            a = predict_record(model, anchor, [100.]*12, 'same', config=sampling)
            b = predict_record(loaded, anchor, [100.]*12, 'same', config=sampling)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_boundary_training_mask_excludes_repeated_target_padding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = make_manifest(root)
            m = read_manifest(path)
            t = np.arange(60000, dtype='float32')
            y = np.stack([t + j for j in range(12)])
            np.savez(root/'train.npz', anchor_uV=y[:1], target_uV=y, lead_mask=np.ones(12, dtype=bool))
            dataset = WindowDataset(m, 40, 1.)
            with patch('numpy.random.rand', side_effect=[0., 0.]):
                batch = dataset[0]
            self.assertFalse(batch['time_mask'][:7500].any())
            self.assertTrue(batch['time_mask'][7500:].all())
            self.assertTrue(torch.equal(batch['anchor'][0, :7500], torch.zeros(7500)))


if __name__ == "__main__":
    unittest.main()
