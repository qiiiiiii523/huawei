import unittest
import numpy as np
import torch
from baselines.B5_final.shared_neural import Correction, objective


class SharedNeuralTests(unittest.TestCase):
    def test_all_heads_zero_init_and_matched_common_weights(self):
        models = []
        for rank in (0, 2, 3):
            torch.manual_seed(42)
            model = Correction(rank, 11, np.eye(6))
            result = model(torch.randn(2, 3, 40), torch.randn(2, 11))
            self.assertTrue(all(torch.count_nonzero(v) == 0 for v in result))
            models.append(model)
        for other in models[1:]:
            for key, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, other.state_dict()[key]))
        counts = [sum(p.numel() for p in m.parameters()) for m in models]
        self.assertLess(max(counts) / min(counts), 1.01)

    def test_real_shared_and_private_backward_with_masked_targets(self):
        torch.manual_seed(1)
        model = Correction(2, 11, np.eye(6))
        delta, shared, private = model(torch.randn(2, 3, 40), torch.randn(2, 11))
        target, previous = torch.randn_like(delta), torch.randn_like(delta)
        error = target - previous
        mask = torch.ones(2, 6, dtype=torch.bool); mask[:, 1] = False
        loss = objective(delta, shared, private, error, previous, target, mask, torch.ones(2))
        changed = target.clone(); changed[:, 1] *= 1000
        changed_error = changed - previous
        second = objective(delta, shared, private, changed_error, previous, changed, mask, torch.ones(2))
        self.assertEqual(float(loss.detach()), float(second.detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(model.shared_out.weight.grad).all())
        self.assertGreater(float(model.shared_out.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.private_out.weight.grad.abs().sum()), 0)

    def test_network_never_normalizes_over_time(self):
        # Nonzero heads expose any unwanted far-away dependency through normalization.
        torch.manual_seed(2)
        model = Correction(2, 11, np.eye(6)).eval()
        torch.nn.init.normal_(model.shared_out.weight, std=.1)
        torch.nn.init.normal_(model.private_out.weight, std=.1)
        x = torch.randn(1, 3, 600); m = torch.randn(1, 11)
        changed = x.clone(); changed[..., 400:] += 100
        with torch.no_grad():
            first = model(x, m)[0]
            second = model(changed, m)[0]
        torch.testing.assert_close(first[..., :100], second[..., :100], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
