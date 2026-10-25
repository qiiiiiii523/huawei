"""Seven-resolution U-Net, exact skip sizes, heads, serialization and recovery."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B7 import train as trainer
from baselines.B7.checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint
from baselines.B7.config import ModelConfig, LossConfig, SamplingConfig
from baselines.B7.data import read_manifest
from baselines.B7.evaluate import evaluate
from baselines.B7.flow import sample
from baselines.B7.losses import flow_loss
from baselines.B7.model import B7UNet
from baselines.B7.package import export
from baselines.B7.pretrain import run
from baselines.B7.tests.test_b7 import small_config, payload
from baselines.B7.tests.test_pcc11 import example
from baselines.B7.tests.test_pprime import fixture, training_config


def deep_config(head='linear'):
    return replace(small_config(), channels=(8,16,32,32,32,32,32),
                   output_head=head, head_channels=8)


class DeepUNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_seven_scales_and_rejects_invalid_widths(self):
        root=Path(__file__).resolve().parents[3]
        cfg=json.loads((root/'configs/b7_pprime.json').read_text())
        self.assertEqual(cfg['model']['channels'],[112,224,448,448,448,448,448])
        for widths in ((8,16),(8,)*9,(8,16,32.)):
            with self.assertRaises(ValueError):ModelConfig(channels=widths)
        self.assertEqual(len(ModelConfig(channels=(8,)*8).channels),8)

    def test_full10s_six_downs_and_six_symmetric_skips(self):
        model=B7UNet(deep_config())
        lengths=[];decoded=[];observed=[]
        hooks=[model.stem.register_forward_pre_hook(lambda m,a:observed.append(a[0].detach().clone()))]
        for module in [model.down1,model.down2,*model.extra_downs]:
            hooks.append(module.register_forward_hook(lambda m,a,o:lengths.append(o.shape[-1])))
        for module in [*model.extra_merges,model.merge1,model.merge0]:
            hooks.append(module.register_forward_hook(lambda m,a,o:decoded.append(o.shape[-1])))
        b=example();original=b['anchor'].clone()
        loss,_=flow_loss(model,b,LossConfig(pcc_leads=11));loss.backward()
        for hook in hooks:hook.remove()
        self.assertEqual(lengths,[2500,1250,625,312,156,78])
        self.assertEqual(decoded,[156,312,625,1250,2500,5000])
        self.assertEqual(len(model.extra_encoders),4)
        self.assertEqual(len(model.extra_decoders),4)
        torch.testing.assert_close(observed[0][:,:1],original,rtol=0,atol=0)
        self.assertTrue(torch.equal(b['anchor'],original))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters()))

    def test_odd_and_long_shapes_all_heads_and_too_short_input(self):
        for head in ('linear','split','multiscale'):
            model=B7UNet(deep_config(head))
            for length in (5000,5001,20000):
                with torch.no_grad():out=model(torch.randn(1,7,length),torch.tensor([.5]),torch.randn(1,1,length))
                self.assertEqual(out.shape,(1,7,length))
                self.assertTrue(torch.isfinite(out).all())
            with self.assertRaisesRegex(ValueError,'T>=64'):
                model(torch.randn(1,7,63),torch.tensor([.5]),torch.randn(1,1,63))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA runtime is unavailable in local CPU environment')
    def test_CUDA_seven_scale_all_four_variants_strict_backward(self):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        old=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for head,lf in [('linear',0.),('linear',1.),('multiscale',1.),('split',0.)]:
                model=B7UNet(deep_config(head)).cuda()
                batch={k:v.cuda()for k,v in example().items()}
                loss,_=flow_loss(model,batch,LossConfig(pcc_leads=11,low_frequency_weight=lf))
                loss.backward();torch.cuda.synchronize()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters()))
                del model,batch,loss
        finally:
            torch.use_deterministic_algorithms(old)

    def test_shared_initialization_and_zero_start_MS(self):
        models=[];states=[]
        for head in ('linear','split','multiscale'):
            torch.manual_seed(42);models.append(B7UNet(deep_config(head)))
            states.append(torch.get_rng_state())
        for state in states[1:]:self.assertTrue(torch.equal(state,states[0]))
        for name,value in models[0].state_dict().items():
            if not name.startswith('output.'):
                for model in models[1:]:self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
        z=torch.randn(1,7,5000);q=torch.randn(1,1,5000);t=torch.tensor([.5])
        with torch.no_grad():torch.testing.assert_close(models[0](z,t,q),models[2](z,t,q),rtol=0,atol=0)

    def test_all_heads_backward_and_private_deep_path_gradients(self):
        for head in ('linear','split','multiscale'):
            model=B7UNet(deep_config(head));b=example()
            loss,_=flow_loss(model,b,LossConfig(pcc_leads=11,low_frequency_weight=1))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters()))
            for down in model.extra_downs:self.assertGreater(down.weight.grad.abs().sum().item(),0)
            for up in model.extra_ups:self.assertGreater(up.weight.grad.abs().sum().item(),0)
            if head=='multiscale':
                self.assertGreater(model.coarse_head[-1].weight.grad.abs().sum().item(),0)
                with torch.no_grad():model.coarse_head[-1].weight.add_(.01)
                model.zero_grad(set_to_none=True)
                loss,_=flow_loss(model,b,LossConfig(pcc_leads=11,low_frequency_weight=1));loss.backward()
                self.assertGreater(model.coarse_head[0].weight.grad.abs().sum().item(),0)

    def test_deep_checkpoint_export_reload_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory()as tmp:
            for head in ('linear','split','multiscale'):
                model=B7UNet(deep_config(head)).eval()
                source=Path(tmp)/(head+'.pt');destination=Path(tmp)/(head+'_export.pt')
                save_checkpoint(source,payload(model));export(source,destination)
                restored=model_from_checkpoint(load_checkpoint(destination))
                q=torch.randn(1,1,120)
                sc=SamplingConfig(window_seconds=10,steps=1,samples=16,sample_batch_size=8)
                torch.testing.assert_close(sample(model,q,['record'],config=sc),sample(restored,q,['record'],config=sc),rtol=0,atol=0)
                with self.assertRaises(RuntimeError):model.load_state_dict(B7UNet(small_config()).state_dict(),strict=True)

    def test_deep_actual_full10s_K16_scoring(self):
        with tempfile.TemporaryDirectory()as tmp:
            manifest=read_manifest(fixture(Path(tmp)))
            sc=SamplingConfig(window_seconds=10,steps=1,samples=16,sample_batch_size=8,validation_batch_size=2)
            for head in ('linear','split','multiscale'):
                score,rows=evaluate(B7UNet(deep_config(head)).eval(),manifest,sc,'public10')
                self.assertTrue(np.isfinite(score['r']))
                self.assertEqual(len(rows),2)
                self.assertEqual(set(score['groups']),{'ptbxl','chapman'})

    def test_deep_exact_interrupted_resume_all_four_variants(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);manifest=fixture(root)
            for head,lf in [('linear',0.),('linear',1.),('multiscale',1.),('split',0.)]:
                name=f'{head}_{lf}';cfg=training_config(head,lf);cfg['model']=deep_config(head).to_dict()
                config=root/(name+'.json');config.write_text(json.dumps(cfg))
                args=Namespace(config=config,manifest=manifest,output=root/(name+'_full'),device='cpu',resume=False)
                with patch.object(trainer,'evaluate',return_value=({'r':.5},[])):run(args)
                full=load_checkpoint(args.output/'last.pt')
                args.output=root/(name+'_resume');save=trainer.save_checkpoint
                def interrupt(path,value):
                    save(path,value)
                    if Path(path).name=='last.pt'and value['update']==1:raise RuntimeError('interrupt test')
                with patch.object(trainer,'save_checkpoint',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'interrupt test'):run(args)
                args.resume=True
                with patch.object(trainer,'evaluate',return_value=({'r':.5},[])):run(args)
                restored=load_checkpoint(args.output/'last.pt')
                for field in ('model','ema'):
                    for name,value in full[field].items():self.assertTrue(torch.equal(value,restored[field][name]))
                for key,state in full['optimizer']['state'].items():
                    for field,value in state.items():self.assertTrue(torch.equal(value,restored['optimizer']['state'][key][field]))
                self.assertEqual(restored['last_validated_update'],2)


if __name__=='__main__':unittest.main()
