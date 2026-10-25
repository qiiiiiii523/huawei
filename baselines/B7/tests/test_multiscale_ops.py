"""Regression for strict deterministic CUDA training of the MS velocity head."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.nn import functional as F

from baselines.B7.config import LossConfig
from baselines.B7.checkpoint import save_checkpoint, load_checkpoint, model_from_checkpoint
from baselines.B7.losses import flow_loss
from baselines.B7.model import B7UNet
from baselines.B7.package import export
from baselines.B7.multiscale_ops import pool_control_features, interpolate_control_points
from baselines.B7.tests.test_b7 import small_config, payload
from baselines.B7.tests.test_pcc11 import example


class MultiscaleOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pool_forward_and_gradient_match_original_bins(self):
        for length, count in ((1250,21),(5000,81),(7,4),(3,5)):
            a=torch.randn(2,3,length,dtype=torch.float64,requires_grad=True)
            b=a.detach().clone().requires_grad_()
            expected=F.adaptive_avg_pool1d(a,count)
            actual=pool_control_features(b,count)
            torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
            upstream=torch.randn_like(expected)
            expected.backward(upstream);actual.backward(upstream)
            torch.testing.assert_close(a.grad,b.grad,atol=1e-12,rtol=1e-12)

    def test_interpolation_forward_gradient_and_exact_endpoints(self):
        for count,length in ((21,5000),(81,20000),(2,9),(4,3)):
            a=torch.randn(2,7,count,dtype=torch.float64,requires_grad=True)
            b=a.detach().clone().requires_grad_()
            expected=F.interpolate(a,size=length,mode='linear',align_corners=True)
            actual=interpolate_control_points(b,length)
            torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
            torch.testing.assert_close(actual[...,0],b[...,0],rtol=0,atol=0)
            torch.testing.assert_close(actual[...,-1],b[...,-1],rtol=0,atol=0)
            upstream=torch.randn_like(expected)
            expected.backward(upstream);actual.backward(upstream)
            torch.testing.assert_close(a.grad,b.grad,atol=1e-10,rtol=1e-10)

    def test_MS_training_does_not_call_incompatible_kernels(self):
        old_interpolate=F.interpolate
        def reject_linear(*args,**kwargs):
            if kwargs.get('mode')=='linear':
                raise AssertionError('CUDA-incompatible linear backward kernel')
            return old_interpolate(*args,**kwargs)
        model=B7UNet(replace(small_config(),output_head='multiscale',head_channels=8))
        with patch.object(F,'adaptive_avg_pool1d',side_effect=AssertionError('adaptive pooling kernel')), \
             patch.object(F,'interpolate',side_effect=reject_linear):
            loss,_=flow_loss(model,example(),LossConfig(pcc_leads=11,low_frequency_weight=1))
            loss.backward()
        self.assertGreater(model.coarse_head[-1].weight.grad.abs().sum().item(),0)

    def test_deterministic_matrix_backward_repeats_exactly(self):
        old=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            x=torch.randn(2,7,1250,requires_grad=True)
            result=[]
            for _ in range(2):
                x.grad=None
                y=interpolate_control_points(pool_control_features(x,21),5000)
                y.square().mean().backward()
                result.append((y.detach().clone(),x.grad.clone()))
            for a,b in zip(*result):torch.testing.assert_close(a,b,rtol=0,atol=0)
        finally:
            torch.use_deterministic_algorithms(old)

    def test_export_keeps_head_and_LF_config_and_predictions(self):
        with tempfile.TemporaryDirectory()as tmp:
            for head in ('linear','split','multiscale'):
                model=B7UNet(replace(small_config(),output_head=head,head_channels=8)).eval()
                value=payload(model)
                value['loss']['low_frequency_weight']=1. if head!='split'else 0.
                source=Path(tmp)/(head+'.pt');destination=Path(tmp)/(head+'_export.pt')
                save_checkpoint(source,value);before=source.read_bytes()
                export(source,destination)
                restored=load_checkpoint(destination)
                self.assertEqual(restored['model_config']['output_head'],head)
                self.assertEqual(restored['loss'],value['loss'])
                self.assertFalse(restored['target_used_at_inference'])
                self.assertNotIn('optimizer',restored)
                other=model_from_checkpoint(restored)
                z=torch.randn(1,7,5000);q=torch.randn(1,1,5000);t=torch.tensor([.4])
                with torch.no_grad():
                    torch.testing.assert_close(model(z,t,q),other(z,t,q),rtol=0,atol=0)
                self.assertEqual(before,source.read_bytes())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA runtime is unavailable in local CPU environment')
    def test_CUDA_strict_deterministic_full_MS_backward(self):
        old=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            model=B7UNet(replace(small_config(),output_head='multiscale',head_channels=8)).cuda()
            batch={k:v.cuda() for k,v in example().items()}
            loss,_=flow_loss(model,batch,LossConfig(pcc_leads=11,low_frequency_weight=1))
            loss.backward()
            with torch.no_grad():model.coarse_head[-1].weight.add_(.01)
            model.zero_grad(set_to_none=True)
            loss,_=flow_loss(model,batch,LossConfig(pcc_leads=11,low_frequency_weight=1))
            loss.backward()
            torch.cuda.synchronize()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters()))
        finally:
            torch.use_deterministic_algorithms(old)


if __name__=='__main__':unittest.main()
