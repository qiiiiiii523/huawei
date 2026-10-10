"""CPU reference equivalence, plus CUDA deterministic backward when available."""
import unittest

import torch
import torch.nn.functional as F

from baselines.B6.resampling import resize_linear_1d


class ResamplingTests(unittest.TestCase):
    def test_forward_and_gradient_match_native_double(self):
        for source, destination in ((250,1250),(1250,5000),(13,27),(250,7),(1,5),(5,1),(17,17)):
            with self.subTest(source=source,destination=destination):
                torch.manual_seed(42)
                actual = torch.randn(2,3,source,dtype=torch.float64,requires_grad=True)
                reference = actual.detach().clone().requires_grad_(True)
                output = resize_linear_1d(actual,destination)
                expected = F.interpolate(reference,size=destination,mode='linear',align_corners=False)
                torch.testing.assert_close(output,expected,rtol=1e-10,atol=1e-10)
                upstream = torch.randn_like(output)
                output.backward(upstream); expected.backward(upstream)
                torch.testing.assert_close(actual.grad,reference.grad,rtol=1e-10,atol=1e-10)

    def test_forward_float32_tolerance(self):
        for source,destination in ((250,1250),(1250,5000),(7,19)):
            x = torch.randn(2,3,source)
            output = resize_linear_1d(x,destination)
            reference = F.interpolate(x,size=destination,mode='linear',align_corners=False)
            # Native float coordinates and double coordinate construction differ
            # slightly at fractional locations, without changing interpolation semantics.
            torch.testing.assert_close(output,reference,rtol=2e-4,atol=2e-4)

    def test_gradcheck_downsample_and_edges(self):
        for source,destination in ((3,9),(9,3),(1,3)):
            x = torch.randn(1,2,source,dtype=torch.float64,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda y:resize_linear_1d(y,destination),(x,)))

    def test_invalid_lengths_and_types(self):
        for x,length in ((torch.ones(1,2),3),(torch.ones(1,1,4),0),(torch.ones(1,1,4),2.5),
                         (torch.ones(1,1,4,dtype=torch.long),3)):
            with self.assertRaises(ValueError):
                resize_linear_1d(x,length)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available; run this regression on the training server')
    def test_cuda_backward_strict_and_repeatable(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            source = torch.randn(2,3,250,device='cuda')
            upstream = torch.randn(2,3,1250,device='cuda')
            gradients = []
            for _ in range(2):
                x = source.detach().clone().requires_grad_(True)
                resize_linear_1d(x,1250).backward(upstream)
                gradients.append(x.grad.detach())
            self.assertTrue(torch.isfinite(gradients[0]).all())
            torch.testing.assert_close(gradients[0],gradients[1],rtol=0,atol=0)
        finally:
            torch.use_deterministic_algorithms(previous)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available; run this regression on the training server')
    def test_cuda_model_backward_strict(self):
        from baselines.B6.model import B6AxialFlow
        from baselines.B6.runtime import seed_all
        from baselines.B6.tests.test_b6 import tiny_config, inputs
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            seed_all(42,True)
            model = B6AxialFlow(tiny_config()).cuda().train()
            data = {k:v.cuda() for k,v in inputs(1,5000).items()}
            state = torch.randn(1,11,5000,device='cuda')
            velocity,anchor = model(state,torch.tensor([.5],device='cuda'),**data)
            (velocity.square().mean()+anchor.square().mean()).backward()
            self.assertTrue(torch.isfinite(model.decoder.head.weight.grad).all())
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == '__main__':
    unittest.main()
