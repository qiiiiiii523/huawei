import unittest
import numpy as np
import torch
from baselines.B5_final.c3_joint import project, joint_prediction, correlation_loss, assert_disjoint
from baselines.B5_final.body_scope import transform


class C3JointTests(unittest.TestCase):
    def test_differentiable_forward_matches_deployed_transform(self):
        rng = np.random.default_rng(43)
        base = rng.normal(size=(12, 5000)).astype(np.float32)
        anchor = rng.normal(size=5000).astype(np.float32)
        residual = rng.normal(size=5000).astype(np.float32) * .1
        scale = 7.
        projected = project(base, anchor)
        z = torch.tensor(residual[None], requires_grad=True)
        result = joint_prediction(z, torch.zeros_like(z), torch.tensor(projected[None] / scale), torch.tensor([scale]))
        expected = transform(base, anchor, base[1] + scale * residual, projection=True, scope="five")[1:6]
        np.testing.assert_allclose(result.detach().numpy()[0] * scale, expected, atol=8e-7, rtol=1e-5)
        result.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0)

    def test_qc_masks_ignore_bad_leads_and_gradient_is_finite(self):
        torch.manual_seed(42)
        p = torch.randn(2, 5, 100, requires_grad=True)
        t = torch.randn_like(p)
        mask = torch.tensor([[True, True, False, False, True]] * 2)
        w = torch.ones(2)
        residual = torch.zeros(2, 100)
        a = correlation_loss(p, t, mask, w, residual)
        changed = t.clone(); changed[:, 2:4] *= 1000
        b = correlation_loss(p, changed, mask, w, residual)
        self.assertEqual(float(a.detach()), float(b.detach()))
        a.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(p.grad[:, 2:4].abs().sum()), 0)

    def test_rejects_train_validation_patient_leakage(self):
        rows = [{"split": "train", "subject_id": "a", "target_record_id": "one"},
                {"split": "validation", "subject_id": "a", "target_record_id": "two"}]
        with self.assertRaises(ValueError):
            assert_disjoint(rows)


if __name__ == "__main__":
    unittest.main()
