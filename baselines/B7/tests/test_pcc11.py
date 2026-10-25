"""Physical scaling, derived-lead gradients, true targets, masks and recovery."""
from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate

from baselines.B7 import GENERATED_INDICES
from baselines.B7 import train as trainer
from baselines.B7.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.data import WindowDataset, read_manifest
from baselines.B7.leads import compose_d12
from baselines.B7.losses import flow_loss, reconstruction_losses, reconstructed_pcc11
from baselines.B7.model import B7UNet
from baselines.B7.pipeline import plan, fingerprint
from baselines.B7.tests.test_b7 import make_manifest, payload, small_config


def example(length=5000):
    t = torch.arange(length)/500
    anchor = (140*torch.sin(2*torch.pi*t))[None,None]
    physical = torch.stack([160*torch.sin(2*torch.pi*t + .3*j) + 30*torch.sin(6*torch.pi*t)
                            for j in range(7)])[None]
    target = compose_d12(anchor, physical)
    scales = torch.tensor([[278.,330.,244.,280.,205.,254.,346.,649.,646.,576.,498.,434.]])
    y = physical / scales[:, GENERATED_INDICES, None]
    batch = dict(anchor=anchor/scales[:, :1, None],target=y,
                 lead_mask=torch.ones(1,7,dtype=torch.bool),time_mask=torch.ones(1,length,dtype=torch.bool),
                 target_d12_uV=target,lead_mask_d12=torch.ones(1,12,dtype=torch.bool),scales_uV=scales)
    return batch


class PCC11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_physical_recovery_matches_independent_pearson_and_ignores_I_target(self):
        b=example()
        p=b['target'].clone();p[:,0] += .15*torch.cos(torch.arange(5000)/17)
        # Deliberately non-formula real III: supervision must preserve it.
        b['target_d12_uV'][:,2] = 70*torch.cos(torch.arange(5000)/77)
        physical=compose_d12(b['anchor']*b['scales_uV'][:,:1,None],p*b['scales_uV'][:,GENERATED_INDICES,None])
        correlations=[]
        for j in range(1,12):
            a=physical[0,j].numpy().astype('float64');y=b['target_d12_uV'][0,j].numpy().astype('float64')
            a-=a.mean();y-=y.mean();correlations.append(float(a@y/np.sqrt((a@a)*(y@y))))
        result=reconstructed_pcc11(p,b['anchor'],b,b['time_mask'],LossConfig(pcc_leads=11))
        self.assertAlmostEqual(float(result['pcc']),1-np.mean(correlations),places=6)
        self.assertEqual(float(result['pcc_valid_leads']),11)
        b['target_d12_uV'][:,0]=float('nan')
        again=reconstructed_pcc11(p,b['anchor'],b,b['time_mask'],LossConfig(pcc_leads=11))
        torch.testing.assert_close(result['pcc'],again['pcc'],rtol=0,atol=0)

    def test_derived_only_supervision_backpropagates_into_II(self):
        for lead in (2,3,4,5):
            b=example();b['lead_mask_d12'].zero_();b['lead_mask_d12'][:,lead]=True
            b['lead_mask'].zero_()
            b['target_d12_uV'][:,[j for j in range(12)if j!=lead]]=float('nan')
            p=b['target'].clone();p[:,0] += .1*torch.cos(torch.arange(5000)/19);p.requires_grad_()
            anchor=b['anchor'].clone()
            result=reconstructed_pcc11(p,b['anchor'],b,b['time_mask'],LossConfig(pcc_leads=11))
            result['pcc'].backward()
            self.assertEqual(float(result['pcc_valid_leads']),1)
            self.assertGreater(float(p.grad[:,0].abs().sum()),0)
            self.assertEqual(float(p.grad[:,1:].abs().sum()),0)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertTrue(torch.equal(anchor,b['anchor']))

    def test_long_edges_and_padding_excluded_from_PCC_but_not_real_MAE(self):
        b=example(20000);p=b['target'].clone()
        p[:,:,:7500] += torch.sin(torch.arange(7500)/11)[None,None]
        p[:,:,12500:] -= torch.cos(torch.arange(7500)/9)[None,None]
        parts=reconstruction_losses(p,b['target'],torch.ones_like(p,dtype=torch.bool),LossConfig(pcc_leads=11))
        self.assertGreater(float(parts['mae']),0)
        r=reconstructed_pcc11(p,b['anchor'],b,b['time_mask'],LossConfig(pcc_leads=11))
        self.assertLess(abs(float(r['pcc'])),1e-6)
        # A boundary window has only a real subinterval inside its central core.
        b['time_mask'][:,:9000]=False
        b['target_d12_uV'][:,:,:9000]=float('nan')
        p[:,:,:9000]=float('nan')
        r=reconstructed_pcc11(p,b['anchor'],b,b['time_mask'],LossConfig(pcc_leads=11))
        self.assertTrue(torch.isfinite(r['pcc']))
        self.assertLess(abs(float(r['pcc'])),1e-6)

    def test_zero_variance_and_no_eligible_leads_are_finite(self):
        b=example();b['target_d12_uV'].fill_(1.)
        p=torch.zeros_like(b['target'],requires_grad=True)
        r=reconstructed_pcc11(p,b['anchor'],b,b['time_mask'],LossConfig(pcc_leads=11))
        self.assertEqual(float(r['pcc']),0)
        self.assertEqual(float(r['pcc_valid_leads']),0)
        r['pcc'].backward();self.assertTrue(torch.isfinite(p.grad).all())
        b['lead_mask_d12'].zero_();b['lead_mask'].zero_();b['target_d12_uV'].fill_(float('nan'))
        r=reconstructed_pcc11(p,b['anchor'],b,b['time_mask'],LossConfig(pcc_leads=11))
        self.assertEqual(float(r['pcc']),0)

    def test_FM_MAE_identical_and_only_PCC_changes(self):
        torch.manual_seed(42);b=example();model=B7UNet(small_config())
        noise=torch.randn_like(b['target']);t=torch.tensor([.2]);original=b['anchor'].clone()
        _,old=flow_loss(model,b,LossConfig(),noise=noise,time=t)
        loss,new=flow_loss(model,b,LossConfig(pcc_leads=11),noise=noise,time=t)
        for key in ('fm','mae','mae_ii','mae_chest'):
            torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
        self.assertEqual(float(new['pcc_valid_leads']),11)
        self.assertNotEqual(float(old['pcc']),float(new['pcc']))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters()))
        self.assertTrue(torch.equal(original,b['anchor']))

    def test_invalid_or_missing_physical_context_rejected(self):
        b=example();cfg=LossConfig(pcc_leads=11)
        for key in ('target_d12_uV','lead_mask_d12','scales_uV'):
            invalid={k:v for k,v in b.items()if k!=key}
            with self.assertRaisesRegex(ValueError,'PCC11 requires'):
                reconstructed_pcc11(b['target'],b['anchor'],invalid,b['time_mask'],cfg)
        b['scales_uV'][0,0]=0
        with self.assertRaisesRegex(ValueError,'positive frozen'):
            reconstructed_pcc11(b['target'],b['anchor'],b,b['time_mask'],cfg)
        b=example();b['lead_mask'][0,0]=False
        with self.assertRaisesRegex(ValueError,'QC disagree'):
            reconstructed_pcc11(b['target'],b['anchor'],b,b['time_mask'],cfg)
        for invalid in (8,12,True,11.):
            with self.assertRaises(ValueError):LossConfig(pcc_leads=invalid)

    def test_loader_preserves_real12_labels_and_cache(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);manifest=make_manifest(root);data=read_manifest(manifest)
            before=file_sha256(root/'train.npz')
            legacy=WindowDataset(data,10,include_d12=False)[0]
            self.assertNotIn('target_d12_uV',legacy)
            new=WindowDataset(data,10,include_d12=True)[0]
            with np.load(root/'train.npz')as f:
                self.assertTrue(np.array_equal(new['target_d12_uV'].numpy(),f['target_uV']))
            self.assertEqual(file_sha256(root/'train.npz'),before)
            self.assertEqual(new['target'].shape,(7,5000))
            self.assertEqual(new['target_d12_uV'].shape,(12,5000))
            self.assertEqual(new['lead_mask_d12'].shape,(12,))
            self.assertEqual(default_collate([new,new])['scales_uV'].shape,(2,12))

    def test_new_loss_initializes_old_EMA_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);manifest=make_manifest(root);model=B7UNet(small_config())
            source=payload(model);source['update']=10000;source['best_r']=.74
            source['loss'].pop('pcc_leads') # Actual old checkpoint schema.
            save_checkpoint(root/'old.pt',source);old_sha=file_sha256(root/'old.pt')
            cfg=dict(stage='public10',model=small_config().to_dict(),loss=asdict(LossConfig(pcc_leads=11)),
                sampling=asdict(SamplingConfig(window_seconds=10,steps=1,samples=1)),
                training=dict(seed=4,updates=2,validate_every=2,batch_size=1,learning_rate=3e-5,
                    weight_decay=.01,ema_decay=.9,gradient_clip=1.,boundary_probability=0.,
                    validation_mode='final_only',checkpoint_every=1))
            config=root/'config.json';config.write_text(json.dumps(cfg))
            args=Namespace(config=config,manifest=manifest,output=root/'full',device='cpu',resume=False,init_checkpoint=root/'old.pt')
            metrics=({'r':.75},[])
            with patch.object(trainer,'evaluate',return_value=metrics)as evaluate:
                trainer.train(args);self.assertEqual(evaluate.call_count,1)
            initial=load_checkpoint(root/'full/initial.pt')
            self.assertEqual(initial['update'],0)
            self.assertEqual(initial['optimizer']['state'],{})
            for k,v in source['ema'].items():self.assertTrue(torch.equal(v,initial['ema'][k]))
            self.assertEqual(initial['initialization']['source_update'],10000)
            args.output=root/'resume'
            save=trainer.save_checkpoint
            def interrupt(path,value):
                save(path,value)
                if Path(path).name=='last.pt'and value['update']==1:raise RuntimeError('test interrupt')
            with patch.object(trainer,'save_checkpoint',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'test interrupt'):trainer.train(args)
            args.resume=True;args.init_checkpoint=None
            with patch.object(trainer,'evaluate',return_value=metrics):trainer.train(args)
            a,b=[load_checkpoint(root/p/'last.pt')for p in ('full','resume')]
            for k in a['model']:
                self.assertTrue(torch.equal(a['model'][k],b['model'][k]))
                self.assertTrue(torch.equal(a['ema'][k],b['ema'][k]))
            for i,state in a['optimizer']['state'].items():
                for k,v in state.items():self.assertTrue(torch.equal(v,b['optimizer']['state'][i][k]))
            self.assertEqual(old_sha,file_sha256(root/'old.pt'))

    def test_M2L_configs_and_stage_dependencies(self):
        repository=Path(__file__).resolve().parents[3]
        for stage in ('public10','public40','huawei40'):
            cfg=json.loads((repository/'configs/u7_m2_l'/f'b7_{stage}.json').read_text())
            old=json.loads((repository/'configs/u7_m2'/f'b7_{stage}.json').read_text())
            mc,lc,sc=trainer.validate_settings(cfg)
            self.assertEqual(cfg['model'],old['model'])
            self.assertEqual(lc.pcc_leads,11)
            self.assertEqual((lc.ii_weight,lc.mae_weight,lc.pcc_weight),(5,1,.1))
            self.assertEqual(cfg['training']['updates'],5000 if stage=='public10'else 1000)
            self.assertEqual(cfg['training']['batch_size'],64 if stage=='public10'else 32)
            self.assertEqual(cfg['training']['validation_mode'],'final_only')
            self.assertEqual((sc.samples,sc.steps),(16,16))
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);source=root/'old.pt';source.write_bytes(b'fixed old EMA')
            args=Namespace(output=root/'new',data_root=root,metadata=root,ptb_root=root,
                reuse_data_from=root/'cache',gpu_p='0',gpu_a='2',gpu_b='0',
                config_dir=repository/'configs/u7_m2_l',init_p_checkpoint=source)
            jobs={j.name:j for j in plan(args)}
            self.assertEqual(jobs['P'].arguments[jobs['P'].arguments.index('--init-checkpoint')+1],str(source))
            for child,parent in [('A','P'),('Q','P'),('B','Q')]:
                command=jobs[child].arguments
                self.assertEqual(command[command.index('--init-checkpoint')+1],str(root/'new'/parent/'best_r.pt'))
                self.assertEqual(jobs[child].dependencies,(parent,))
            old_hash=fingerprint(args);source.write_bytes(b'changed old EMA')
            self.assertNotEqual(old_hash,fingerprint(args))


if __name__=='__main__':unittest.main()
