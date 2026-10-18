from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from baselines.B5_control_pcc.losses import objective, multiscale_shape_loss, scale_bands
from baselines.B5_control_pcc.prepare import multiscale_settings
from baselines.B5_control_pcc.common import multiscale_for_group
from baselines.B5_control_pcc import GROUPS

torch.set_num_threads(2)


def config():
    return {'weight': .1, 'shape_huber_weight': .1, 'std_floor': [[.001] * 11 for _ in range(3)]}


def wave():
    t = torch.arange(6000) / 50
    return (torch.sin(t * 3) + .5 * torch.sin(t * .7) + .3 * torch.sin(t * .1))[None, None].repeat(1, 11, 1)


class MultiscaleTests(unittest.TestCase):
    def test_group_configuration_is_zero_zero_positive_and_shared_otherwise(self):
        study = {'multiscale': config()}
        values = [multiscale_for_group(study, g) for g in GROUPS]
        self.assertEqual([v['weight'] for v in values], [0., 0., .1])
        for value in values:
            self.assertEqual(value['std_floor'], study['multiscale']['std_floor'])
        self.assertEqual(study['multiscale']['weight'], .1)

    def test_disabled_auxiliary_preserves_original_loss_and_gradient_exactly(self):
        p = (.6 * wave()).requires_grad_()
        batch = {'mask': torch.ones(1, 12, dtype=torch.bool), 'slow_target': wave(),
                 'fm_uV': torch.sin(torch.arange(60000) / 20)[None, None].repeat(1, 12, 1),
                 'target_uV': torch.cos(torch.arange(60000) / 30)[None, None].repeat(1, 12, 1)}
        original, _ = objective(p, batch, torch.ones(12), .5, .1, beta=.75)
        original_grad = torch.autograd.grad(original, p)[0]
        disabled, _ = objective(p, batch, torch.ones(12), .5, .1, beta=.75, multiscale={**config(), 'weight': 0.})
        disabled_grad = torch.autograd.grad(disabled, p)[0]
        torch.testing.assert_close(original, disabled, rtol=0, atol=0)
        torch.testing.assert_close(original_grad, disabled_grad, rtol=0, atol=0)
        enabled, parts = objective(p, batch, torch.ones(12), .5, .1, beta=.75, multiscale=config())
        torch.testing.assert_close(enabled, original + .1 * parts['multiscale_loss'])
        self.assertGreater(float(parts['multiscale_loss'].detach()), 0)

    def test_scale_loss_matches_truth_and_penalizes_shape_and_relative_amplitude(self):
        y = wave()
        mask = torch.ones(1, 11, dtype=torch.bool)
        perfect, _ = multiscale_shape_loss(y, y, mask, config())
        weak, _ = multiscale_shape_loss(.2 * y, y, mask, config())
        wrong, _ = multiscale_shape_loss(-y, y, mask, config())
        shifted, _ = multiscale_shape_loss(torch.roll(y, 100, -1), y, mask, config())
        self.assertLess(float(perfect), 1e-6)
        self.assertGreater(float(weak), float(perfect) + .01)
        self.assertGreater(float(wrong), float(weak))
        self.assertGreater(float(shifted), float(perfect) + .01)

    def test_qc_and_flat_labels_produce_no_bad_lead_gradient(self):
        y, p = wave(), (.3 * wave()).requires_grad_()
        mask = torch.ones(1, 11, dtype=torch.bool)
        mask[:, 3] = False
        y[:, 4] = 2
        loss, parts = multiscale_shape_loss(p, y, mask, config())
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(p.grad[:, 3:5].abs().sum()), 0.)
        self.assertGreater(float(p.grad[:, :3].abs().sum()), 0.)
        self.assertLess(float(parts['long_gt15s_eligible_fraction']), 1.)
        flat = torch.zeros_like(y, requires_grad=True)
        zero, _ = multiscale_shape_loss(flat, torch.zeros_like(y), mask, config())
        zero.backward()
        self.assertEqual(float(zero.detach()), 0.)
        self.assertTrue(torch.isfinite(flat.grad).all())

    def test_centering_is_loss_only_and_bands_reconstruct_full_curve(self):
        y = wave()
        bands = scale_bands(y)
        torch.testing.assert_close(sum(bands), y)
        p = .5 * y
        before = p.clone()
        mask = torch.ones(1, 11, dtype=torch.bool)
        a, _ = multiscale_shape_loss(p, y, mask, config())
        b, _ = multiscale_shape_loss(p + 3, y - 2, mask, config())
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(p, before, rtol=0, atol=0)

    def test_thresholds_use_only_supplied_training_targets_and_target_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            y = wave()[0].numpy()
            mask = np.ones(12, bool)
            mask[4] = False
            np.savez(root / 'train.npz', slow_target=y, mask=mask)
            np.savez(root / 'validation.npz', slow_target=y * 100000, mask=np.ones(12, bool))
            settings = multiscale_settings(root, [{'file': 'train.npz'}], .1)
            bands = scale_bands(torch.from_numpy(y)[None])
            expected = np.array([b[0].std(-1, unbiased=False).numpy() for b in bands])
            actual = np.array(settings['train_median_std'])
            np.testing.assert_allclose(actual[:, :3], expected[:, :3], rtol=2e-5, atol=2e-6)
            np.testing.assert_array_equal(actual[:, 3], 0.)
            self.assertEqual(settings['floor_source'], 'TRAIN-target-QC-only')
            self.assertEqual(np.array(settings['std_floor']).shape, (3, 11))


if __name__ == '__main__':
    unittest.main()
