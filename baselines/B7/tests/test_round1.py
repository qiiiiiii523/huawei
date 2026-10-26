from argparse import Namespace
from dataclasses import asdict
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import numpy as np
import torch
from baselines.B7 import GENERATED_INDICES
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.data import read_manifest, load_record
from baselines.B7.evaluate import aggregate, limb_diagnostics
from baselines.B7 import round1
from baselines.B7.huawei10 import build
from baselines.B7.limb_losses import physical_losses
from baselines.B7.round1 import configurations, check_configs, verify_stage, lock_run
from baselines.B7.train import validation_due
from baselines.B7 import train as trainer
from baselines.B7.checkpoint import load_checkpoint
from baselines.B7.tests.test_b7 import small_config, make_manifest
from baselines.B7.tests.test_pcc11 import example
from baselines.B7.losses import flow_loss
from baselines.B7.model import B7UNet


class RoundOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_named_configs_budget_and_validation_nodes(self):
        for variant in ('C0', 'C1', 'C2', 'C3', 'C4'):
            cfg = configurations(variant); check_configs(variant, cfg)
            self.assertEqual([s for s in range(1,20001) if validation_due(cfg['P']['training'],s)], [10000,20000])
            self.assertEqual([s for s in range(1,4001) if validation_due(cfg['A']['training'],s)], [1000,2000,3000,4000])
            self.assertEqual(cfg['P']['model'], cfg['A']['model'])
        cfg = configurations('C2'); cfg['P']['loss']['pcc_weight'] = .5
        with self.assertRaises(ValueError): check_configs('C2', cfg)

    def test_derived_loss_physical_scaling_and_gradient_only_II(self):
        b=example();p=b['target'].clone();p[:,0] += .2;p.requires_grad_()
        cfg=LossConfig(pcc_leads=11,limb_wave_weight=.1)
        r=physical_losses(p,b['anchor'],b,b['time_mask'],cfg)
        # A constant II error propagates identically to III/aVF in physical uV.
        expected=[]
        for j in (2,5):
            expected.append(.2*float(b['scales_uV'][0,1])/max(float(b['target_d12_uV'][0,j].std(unbiased=False)),20.))
        self.assertAlmostEqual(float(r['limb_wave']),np.mean(expected),places=5)
        r['limb_wave'].backward()
        self.assertGreater(float(p.grad[:,0].abs().sum()),0)
        self.assertEqual(float(p.grad[:,1:].abs().sum()),0)

    def test_projection_responds_to_coefficient_not_orthogonal_shape(self):
        b=example();p=b['target'].clone()
        a=b['anchor'][:,0]*b['scales_uV'][:,0,None]
        ac=a-a.mean(-1,keepdim=True)
        delta=torch.cos(torch.arange(5000)/13)[None]
        delta=delta-delta.mean(-1,keepdim=True)
        delta=delta-(delta*ac).sum(-1,keepdim=True)/ac.square().sum(-1,keepdim=True)*ac
        p[:,0]+=delta/b['scales_uV'][:,1,None]
        cfg=LossConfig(pcc_leads=11,projection_weight=.1)
        self.assertLess(float(physical_losses(p,b['anchor'],b,b['time_mask'],cfg)['projection']),1e-6)
        p[:,0]+=.5*a/b['scales_uV'][:,1,None];p.requires_grad_()
        r=physical_losses(p,b['anchor'],b,b['time_mask'],cfg)
        expected=.5*float(ac.std(unbiased=False))/max(float(b['target_d12_uV'][:,1].std(unbiased=False)),20.)
        self.assertAlmostEqual(float(r['projection']),expected,places=5)
        r['projection'].backward();self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(p.grad[:,1:].abs().sum()),0)

    def test_invalid_nan_padding_constant_and_small_anchor_are_safe(self):
        b=example();p=b['target'].clone().requires_grad_()
        b['lead_mask_d12'][:,[1,2,5]]=False;b['lead_mask'][:,0]=False
        b['target_d12_uV'][:,[1,2,5]]=float('nan')
        cfg=LossConfig(pcc_leads=11,limb_wave_weight=.1,projection_weight=.1)
        r=physical_losses(p,b['anchor'],b,b['time_mask'],cfg)
        self.assertEqual(float(r['limb_wave']),0);self.assertEqual(float(r['projection']),0)
        (r['limb_wave']+r['projection']).backward();self.assertTrue(torch.isfinite(p.grad).all())
        b=example();b['anchor'].zero_();p=b['target'].clone().requires_grad_()
        r=physical_losses(p,b['anchor'],b,b['time_mask'],cfg)
        self.assertEqual(float(r['projection_valid']),0)
        b=example();b['time_mask'][:,:250]=False;b['target_d12_uV'][:,:,:250]=float('nan')
        r=physical_losses(b['target'],b['anchor'],b,b['time_mask'],cfg)
        self.assertLess(float(r['limb_wave']),1e-6)
        self.assertLess(float(r['projection']),1e-6)

    def test_new_terms_are_added_without_changing_FM_or_MAE7(self):
        b=example();model=B7UNet(small_config())
        noise=torch.zeros_like(b['target']);time=torch.tensor([.5])
        base,parts=flow_loss(model,b,LossConfig(pcc_leads=11),noise=noise,time=time)
        for cfg,key in ((LossConfig(pcc_leads=11,limb_wave_weight=.1),'limb_wave'),
                        (LossConfig(pcc_leads=11,projection_weight=.1),'projection')):
            new,extra=flow_loss(model,b,cfg,noise=noise,time=time)
            torch.testing.assert_close(new,base+.1*extra[key])
            for oldkey in ('fm','mae','pcc'):torch.testing.assert_close(parts[oldkey],extra[oldkey])
            new.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None));model.zero_grad()

    def test_virtual_huawei_windows_preserve_patients_pairs_and_QC(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pub=make_manifest(root);data=json.loads(pub.read_text());h=copy.deepcopy(data)
            for row in h['records']:
                row['group']='task1';row['source']='huawei'
                with np.load(root/row['path']) as old:
                    y=np.tile(old['target_uV'],(1,2))
                a=y[:1].copy()
                y[6,5000:]=0
                dest=root/(row['split']+'_long.npz');np.savez(dest,anchor_uV=a,target_uV=y,lead_mask=np.ones(12,bool))
                row['path']=str(dest)
            h['scales_uV']=[200.]*12
            source=root/'huawei.json';source.write_text(json.dumps(h));out=root/'views/manifest.json'
            value=build(source,pub,out);self.assertEqual(len(value['records']),4)
            self.assertEqual(value['scales_uV'],[100.]*12)
            self.assertEqual(value['parent_scales_uV'],[200.]*12)
            m=read_manifest(out);x=m['records'][1];a,y,mask,_=load_record(x)
            self.assertEqual(a.shape,(1,5000));self.assertEqual(y.shape,(12,5000));self.assertFalse(mask[6])
            self.assertEqual(x['subject_id'],'train');self.assertEqual(x['split'],'train')
            self.assertTrue(x['target_record_id'].endswith(':10s=5000'))
            self.assertEqual(len(list(root.glob('*.npz'))),4)
            with self.assertRaises(FileExistsError):build(source,pub,out)
            x['window_stop']=10001
            with self.assertRaises(ValueError):load_record(x)

    def test_huawei10_uses_original_group_weights(self):
        rows=[]
        for group,r in [('task1',.8),('machine',.4),('body',.2)]:
            rows.append({'group':group,'target_record_id':group,'leads':{str(j):{'r':r} for j in range(1,12)}})
        self.assertAlmostEqual(aggregate(rows,'huawei10')['r'],.55)

    def test_descriptive_diagnostics_do_not_modify_waveforms(self):
        b=example();anchor=(b['anchor']*b['scales_uV'][:,:1,None])[0].numpy();y=b['target_d12_uV'][0].numpy()
        p=y.copy();p[1]=.7*y[1];original=p.copy()
        d=limb_diagnostics(anchor,p,y,np.ones(12,bool))
        self.assertAlmostEqual(d['II']['std_ratio'],.7,places=6)
        self.assertTrue(np.array_equal(p,original))
        self.assertGreater(d['I_II']['projection_component_error'],0)

    def test_round_runner_gates_A_after_P_success_and_blocks_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pub=root/'pub.json';hua=root/'hua.json';raw=root/'raw.npz'
            for f in (pub,hua,raw):f.write_bytes(b'identity')
            fake_manifest={'records':[{'resolved_path':str(raw)}]}
            args=Namespace(variant='C0',public_manifest=str(pub),huawei_manifest=str(hua),output=str(root/'run'),
                           gpu_p='0',gpu_a='2',device='cpu',check_only=False)
            commands=[]
            def child(command,**kwargs):
                commands.append((command,kwargs['env']['CUDA_VISIBLE_DEVICES']))
                folder=Path(command[command.index('--output')+1]);folder.mkdir();(folder/'completed.json').write_text('{}')
                return Mock(wait=Mock(return_value=0))
            with patch.object(round1,'preflight',return_value=(fake_manifest,fake_manifest)),patch.object(round1.subprocess,'Popen',side_effect=child),patch.object(round1,'verify_stage',return_value={'best_r':.5}):
                round1.run(args)
            self.assertEqual(len(commands),2);self.assertNotIn('--init-checkpoint',commands[0][0])
            self.assertIn('--init-checkpoint',commands[1][0]);self.assertTrue(commands[1][0][-1].endswith('P\\best_r.pt') or commands[1][0][-1].endswith('P/best_r.pt'))
            self.assertEqual([c[1] for c in commands],['0','2'])
            self.assertTrue(json.loads((root/'run/status.json').read_text())['success'])
            args.output=str(root/'failure')
            with patch.object(round1,'preflight',return_value=(fake_manifest,fake_manifest)),patch.object(round1.subprocess,'Popen',return_value=Mock(wait=Mock(return_value=1))) as proc:
                with self.assertRaises(RuntimeError):round1.run(args)
            self.assertEqual(proc.call_count,1)
            self.assertEqual(json.loads((root/'failure/status.json').read_text())['jobs'],{'P':'failed','A':'blocked'})

    def test_duplicate_run_lock_releases_after_exit(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'run.lock'
            with lock_run(path):
                with self.assertRaises(OSError):
                    with lock_run(path):pass
            with lock_run(path):pass

    def test_no_initial_P_validation_and_resume_after_early_stop(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest=make_manifest(root)
            cfg={'stage':'public10','model':small_config().to_dict(),'loss':asdict(LossConfig()),
                 'sampling':asdict(SamplingConfig(window_seconds=10,steps=1,samples=1)),
                 'training':{'seed':42,'updates':4,'validate_every':2,'batch_size':1,'learning_rate':1e-4,'weight_decay':.01,
                    'ema_decay':.999,'gradient_clip':1.,'boundary_probability':0.,'checkpoint_every':1,'initial_validation':False}}
            path=root/'cfg.json';path.write_text(json.dumps(cfg));out=root/'P';calls=[]
            def evaluate(*args):calls.append(1);return {'r':.5},[]
            args=Namespace(manifest=str(manifest),config=str(path),output=str(out),device='cpu',resume=False,init_checkpoint=None)
            with patch.object(trainer,'evaluate',side_effect=evaluate):trainer.train(args)
            self.assertEqual(len(calls),2)
            self.assertFalse((out/'validation_0000000.json').exists())
            self.assertEqual(verify_stage(out,json.loads(path.read_text()),manifest)['updates'],4)
            cfg['training'].update(initial_validation=True,early_stop_patience=1,early_stop_min_update=2,early_stop_min_delta=.0001)
            path.write_text(json.dumps(cfg));args.output=str(root/'A')
            with patch.object(trainer,'evaluate',return_value=({'r':.5},[])):trainer.train(args)
            done=json.loads((Path(args.output)/'completed.json').read_text());self.assertEqual(done['update'],2);self.assertEqual(done['best_update'],0)
            args.resume=True
            with patch.object(trainer,'evaluate',side_effect=AssertionError('No extra validation after stopped run')):trainer.train(args)
            self.assertEqual(load_checkpoint(Path(args.output)/'last.pt')['update'],2)

    def test_P_best_EMA_initializes_Huawei10_with_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest=make_manifest(root)
            cfg={'stage':'public10','model':small_config().to_dict(),'loss':asdict(LossConfig(pcc_leads=11)),
                 'sampling':asdict(SamplingConfig(window_seconds=10,steps=1,samples=1)),
                 'training':{'seed':42,'updates':2,'validate_every':2,'batch_size':1,'learning_rate':1e-4,'weight_decay':.01,
                    'ema_decay':.999,'gradient_clip':1.,'boundary_probability':0.,'checkpoint_every':1,'initial_validation':False}}
            path=root/'cfg.json';path.write_text(json.dumps(cfg));args=Namespace(manifest=str(manifest),config=str(path),output=str(root/'P'),device='cpu',resume=False,init_checkpoint=None)
            with patch.object(trainer,'evaluate',return_value=({'r':.5},[])):trainer.train(args)
            source=root/'P/best_r.pt';cfg['stage']='huawei10';cfg['training'].update(learning_rate=1e-5,initial_validation=True)
            cfg['loss']['limb_wave_weight']=.1;path.write_text(json.dumps(cfg));args.output=str(root/'A');args.init_checkpoint=str(source)
            with patch.object(trainer,'evaluate',return_value=({'r':.6},[])):trainer.train(args)
            initial=load_checkpoint(root/'A/initial.pt');old=load_checkpoint(source)
            for k,v in old['ema'].items():torch.testing.assert_close(initial['model'][k],v,rtol=0,atol=0)
            self.assertEqual(initial['optimizer']['state'],{})
            self.assertTrue(initial['initialization']['optimizer_reset'])
            self.assertEqual(verify_stage(root/'A',json.loads(path.read_text()),manifest,source)['best_update'],0)


if __name__=='__main__':unittest.main()
