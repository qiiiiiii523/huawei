"""P-only scope, physical LF scale, masks, head gradients, inference and recovery."""
from argparse import Namespace
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B7 import train as trainer
from baselines.B7.checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint
from baselines.B7.config import LossConfig, ModelConfig, SamplingConfig
from baselines.B7.data import read_manifest
from baselines.B7.evaluate import evaluate, SCORE_PROTOCOL
from baselines.B7.flow import sample
from baselines.B7.leads import compose_d12
from baselines.B7.losses import flow_loss
from baselines.B7.low_frequency import cosine_basis, low_frequency_loss, project_slow
from baselines.B7.model import B7UNet, control_point_count
from baselines.B7.pretrain import check_public_run, run
from baselines.B7.public_qc import PUBLIC_QC_VERSION
from baselines.B7.tests.test_b7 import make_manifest, payload, small_config
from baselines.B7.tests.test_pcc11 import example


def fixture(root):
    path = make_manifest(root)
    value = json.loads(path.read_text())
    originals = value['records']
    rows = []
    for source in ('ptbxl', 'chapman'):
        for row in originals:
            row = dict(row)
            record_id = source + '_' + row['record_id']
            with np.load(root / row['path']) as f:
                target = f['target_uV'].copy()
            target += len(rows) * .01
            filename = record_id + '.npz'
            np.savez(root / filename, anchor_uV=target[:1], target_uV=target,
                     lead_mask=np.ones(12, dtype=bool))
            row.update(record_id=record_id, subject_id=record_id, target_record_id=record_id,
                       source=source, group=source, path=filename,
                       public_qc={'waveform_sha256': hashlib.sha256(target.tobytes()).hexdigest()})
            rows.append(row)
    value.update(records=rows, public_qc_version=PUBLIC_QC_VERSION)
    path.write_text(json.dumps(value))
    return path


def training_config(head='linear', lf=0.):
    return dict(stage='public10', model=replace(small_config(), output_head=head, head_channels=8).to_dict(),
                loss=asdict(LossConfig(pcc_leads=11, low_frequency_weight=lf)),
                sampling=asdict(SamplingConfig(window_seconds=10, samples=1, steps=1)),
                training=dict(seed=42, updates=2, validate_every=2, batch_size=1,
                              learning_rate=1e-4, weight_decay=.01, ema_decay=.999,
                              gradient_clip=1., boundary_probability=0., checkpoint_every=1,
                              validation_mode='final_only'))


class PPrimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_inherits_latest_M2_budget_and_500Hz(self):
        root = Path(__file__).resolve().parents[3]
        c = json.loads((root / 'configs/b7_pprime.json').read_text())
        original = json.loads((root / 'configs/u7_m2/b7_public10.json').read_text())
        mc, lc, sc = trainer.validate_settings(c)
        self.assertEqual(c['training'], original['training'])
        self.assertEqual(c['sampling'], original['sampling'])
        self.assertEqual(c['training']['updates'], 20000)
        self.assertEqual((sc.sample_rate, sc.window_seconds, sc.core_seconds), (500,10,10))
        self.assertEqual((lc.pcc_leads, lc.ii_weight, lc.pcc_weight), (11,5,.1))
        self.assertIn((mc.output_head, lc.low_frequency_weight),
                      [('linear',0.),('linear',1.),('multiscale',1.),('split',0.)])
        for key in ('channels','dilations','time_dim','metadata_dim'):
            self.assertEqual(c['model'][key], original['model'][key])

    def test_physical_DCT_modes_DC_and_fast_components(self):
        for length in (5000,20000):
            basis = cosine_basis(length, .5, 'cpu', torch.float32)
            self.assertEqual(len(basis), int(2*length*.5/500))
            torch.testing.assert_close(basis @ basis.T, torch.eye(len(basis)), atol=2e-6, rtol=0)
            n = torch.arange(length, dtype=torch.float64) + .5
            slow = torch.cos(torch.pi * n / length * 4).float()  # .2Hz on10s; .05Hz on40s
            fast = torch.cos(torch.pi * n / length * int(2*length*5/500)).float()
            torch.testing.assert_close(project_slow(slow+fast+3., .5), slow, atol=2e-6, rtol=0)
            self.assertEqual(project_slow(torch.ones(length)*4, .5).abs().max().item(), 0.)

    def test_LF_masks_nan_weighting_and_gradient(self):
        y = torch.zeros(1,7,5000)
        wave = torch.cos(torch.pi*(torch.arange(5000)+.5)*4/5000)
        p = torch.zeros_like(y); p[:,0] = wave
        p.requires_grad_()
        mask = torch.ones_like(p, dtype=torch.bool)
        r = low_frequency_loss(p,y,mask,LossConfig(low_frequency_weight=1))
        self.assertAlmostEqual(r['low_frequency_mae'].item(), wave.abs().mean().item()*5/11, places=6)
        r['low_frequency_mae'].backward()
        self.assertGreater(p.grad[:,0].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(p.grad).all())
        mask[:,1] = False
        p = p.detach(); p[:,1] = float('nan'); y[:,1] = 1e30
        a = low_frequency_loss(p,y,mask,LossConfig())
        p[:,1] = -1e30
        b = low_frequency_loss(p,y,mask,LossConfig())
        torch.testing.assert_close(a['low_frequency_mae'],b['low_frequency_mae'],rtol=0,atol=0)
        mask.zero_(); p.fill_(float('nan')); p.requires_grad_()
        z = low_frequency_loss(p,y,mask,LossConfig())['low_frequency_mae']
        self.assertEqual(z.item(),0);z.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_LF_never_projects_across_masked_gaps(self):
        # Distinct constant offsets in separate valid regions are both DC, not a
        # slow curve to connect across a gap. A zero-fill/global filter would fail.
        p = torch.zeros(1,7,5000); y = p.clone();mask = torch.zeros_like(p,dtype=torch.bool)
        mask[:,:,:1500] = True;mask[:,:,3000:] = True
        p[:,:,:1500] = 10;p[:,:,3000:] = -4;p[:,:,1500:3000] = float('nan')
        r = low_frequency_loss(p,y,mask,LossConfig())
        self.assertEqual(r['low_frequency_mae'].item(),0)
        self.assertEqual(r['low_frequency_valid_points'].item(),3500*7)

    def test_LF_adds_only_its_term_without_extra_model_calls(self):
        b = example();model = B7UNet(small_config());z = torch.randn_like(b['target']);t = torch.tensor([.1])
        calls=[];hook=model.register_forward_hook(lambda *a: calls.append(1))
        old, a = flow_loss(model,b,LossConfig(pcc_leads=11),noise=z,time=t)
        new, c = flow_loss(model,b,LossConfig(pcc_leads=11,low_frequency_weight=1),noise=z,time=t)
        hook.remove();self.assertEqual(len(calls),2)
        for key in a:torch.testing.assert_close(a[key],c[key],rtol=0,atol=0)
        torch.testing.assert_close(new,old+c['low_frequency_mae'])
        new.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_heads_preserve_trunk_rng_clean_I_and_5000_or_20000_samples(self):
        reference=None;rng=None
        for head in ('linear','split','multiscale'):
            torch.manual_seed(123)
            model=B7UNet(replace(small_config(),output_head=head,head_channels=8))
            if reference is None:reference=model.state_dict();rng=torch.get_rng_state()
            else:
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for key,value in reference.items():
                    if not key.startswith('output.'):
                        self.assertTrue(torch.equal(value,model.state_dict()[key]),key)
            for length in (5000,20000):
                anchor=torch.randn(1,1,length);original=anchor.clone();state=torch.randn(1,7,length)
                observed=[];hook=model.stem.register_forward_pre_hook(lambda m,a:observed.append(a[0].detach().clone()))
                with torch.no_grad():out=model(state,torch.tensor([.4]),anchor)
                hook.remove()
                self.assertEqual(out.shape,(1,7,length))
                self.assertTrue(torch.equal(observed[0][:,:1],original))
                self.assertTrue(torch.equal(anchor,original))

    def test_split_head_order_and_private_gradients(self):
        model=B7UNet(replace(small_config(),output_head='split',head_channels=8))
        ii=[];chest=[]
        a=model.ii_head.register_forward_hook(lambda m,i,o:ii.append(o))
        b=model.chest_head.register_forward_hook(lambda m,i,o:chest.append(o))
        out=model(torch.randn(1,7,5000),torch.tensor([.2]),torch.randn(1,1,5000))
        a.remove();b.remove()
        torch.testing.assert_close(out[:,:1],ii[0]);torch.testing.assert_close(out[:,1:],chest[0])
        out[:,:1].square().mean().backward(retain_graph=True)
        self.assertGreater(model.ii_head[0].weight.grad.abs().sum().item(),0)
        self.assertEqual(model.chest_head[0].weight.grad.abs().sum().item(),0)
        model.zero_grad(set_to_none=True);out[:,1:].square().mean().backward()
        self.assertGreater(model.chest_head[0].weight.grad.abs().sum().item(),0)

    def test_MS_zero_initialization_knots_and_gradient(self):
        self.assertEqual(control_point_count(5000,.5),21)
        self.assertEqual(control_point_count(20000,.5),81)
        torch.manual_seed(42);base=B7UNet(small_config())
        torch.manual_seed(42);ms=B7UNet(replace(small_config(),output_head='multiscale',head_channels=8))
        z=torch.randn(1,7,5000);q=torch.randn(1,1,5000);t=torch.tensor([.5])
        torch.testing.assert_close(base(z,t,q),ms(z,t,q),rtol=0,atol=0)
        ms(z,t,q).square().mean().backward()
        self.assertGreater(ms.coarse_head[-1].weight.grad.abs().sum().item(),0)
        with torch.no_grad():ms.coarse_head[-1].weight.add_(.01)
        ms.zero_grad(set_to_none=True);ms(z,t,q).square().mean().backward()
        self.assertGreater(ms.coarse_head[0].weight.grad.abs().sum().item(),0)

    def test_checkpoint_roundtrip_all_heads_and_same_K16(self):
        with tempfile.TemporaryDirectory()as tmp:
            for head in ('linear','split','multiscale'):
                model=B7UNet(replace(small_config(),output_head=head,head_channels=8)).eval()
                path=Path(tmp)/f'{head}.pt';save_checkpoint(path,payload(model))
                loaded=model_from_checkpoint(load_checkpoint(path))
                q=torch.randn(1,1,120);original=q.clone()
                sc=SamplingConfig(window_seconds=10,steps=1,samples=16,sample_batch_size=4)
                torch.testing.assert_close(sample(model,q,['record'],config=sc),sample(loaded,q,['record'],config=sc),rtol=0,atol=0)
                self.assertTrue(torch.equal(q,original))

    def test_actual_full10s_K16_scoring_all_heads(self):
        # Real inference/scoring, not the mocked validation used by recovery tests.
        # Keep Heun at1 step here; production remains Heun16 in the frozen config.
        with tempfile.TemporaryDirectory()as tmp:
            m=read_manifest(fixture(Path(tmp)))
            sc=SamplingConfig(window_seconds=10,steps=1,samples=16,sample_batch_size=8,
                              validation_batch_size=2)
            for head in ('linear','split','multiscale'):
                model=B7UNet(replace(small_config(),output_head=head,head_channels=8)).eval()
                score,rows=evaluate(model,m,sc,'public10')
                self.assertEqual(score['score_protocol'],SCORE_PROTOCOL)
                self.assertEqual(score['pairs'],2)
                self.assertEqual(set(score['groups']),{'ptbxl','chapman'})
                self.assertTrue(np.isfinite(score['r']))
                self.assertTrue(all(set(row['leads'])=={str(j)for j in range(1,12)}for row in rows))

    def test_P_only_rejects_old_cache_wrong_sources_lengths_and_non_P(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=fixture(root);m=read_manifest(p);c=training_config()
            check_public_run(c,m)
            with self.assertRaisesRegex(ValueError,'QC-v2'):
                check_public_run(c,{k:v for k,v in m.items() if k!='public_qc_version'})
            for row in m['records']:row['source']='ptbdb'
            with self.assertRaisesRegex(ValueError,'PTB-XL and Chapman'):check_public_run(c,m)
            m=read_manifest(p);row=m['records'][0]
            np.savez(row['resolved_path'],anchor_uV=np.ones((1,4000)),target_uV=np.ones((12,4000)),lead_mask=np.ones(12,bool))
            with self.assertRaisesRegex(ValueError,'5000-point'):check_public_run(c,m)
            c['stage']='public40';c['sampling']['window_seconds']=40
            with self.assertRaisesRegex(ValueError,'public10 only'):check_public_run(c,m)

    def test_bad_model_and_LF_settings_rejected(self):
        for setting in ({'output_head':'cascade'},{'head_channels':0},{'coarse_interval_seconds':0},
                        {'coarse_interval_seconds':float('nan')}):
            with self.assertRaises(ValueError):ModelConfig(**setting)
        for setting in ({'low_frequency_weight':-1},{'low_frequency_cutoff_hz':0},
                        {'low_frequency_cutoff_hz':250},{'low_frequency_cutoff_hz':float('nan')}):
            with self.assertRaises(ValueError):LossConfig(**setting)

    def test_P_only_random_training_resume_and_final_validation_each_variant(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);manifest=fixture(root)
            for head,lf in [('linear',0.),('linear',1.),('multiscale',1.),('split',0.)]:
                name=f'{head}_{lf}';cfg=training_config(head,lf);config=root/(name+'.json');config.write_text(json.dumps(cfg))
                args=Namespace(config=config,manifest=manifest,output=root/(name+'_full'),device='cpu',resume=False)
                with patch.object(trainer,'evaluate',return_value=({'r':.5},[]))as evaluation:
                    run(args);self.assertEqual(evaluation.call_count,1)
                initial=load_checkpoint(args.output/'initial.pt')
                self.assertEqual(initial['initialization']['kind'],'random_B7')
                self.assertEqual(initial['optimizer']['state'],{})
                initial_path=args.output/'initial.pt';initial_hash=hashlib.sha256(initial_path.read_bytes()).hexdigest()
                full=load_checkpoint(args.output/'last.pt')
                self.assertEqual((full['update'],full['last_validated_update'],full['pending_validation']),(2,2,False))
                args.output=root/(name+'_resume');original_save=trainer.save_checkpoint
                def interrupt(path,value):
                    original_save(path,value)
                    if Path(path).name=='last.pt' and value['update']==1:raise RuntimeError('interrupt test')
                with patch.object(trainer,'save_checkpoint',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'interrupt test'):run(args)
                args.resume=True
                with patch.object(trainer,'evaluate',return_value=({'r':.5},[]))as evaluation:
                    run(args);self.assertEqual(evaluation.call_count,1)
                recovered=load_checkpoint(args.output/'last.pt')
                for key in full['model']:
                    self.assertTrue(torch.equal(full['model'][key],recovered['model'][key]))
                    self.assertTrue(torch.equal(full['ema'][key],recovered['ema'][key]))
                for key,state in full['optimizer']['state'].items():
                    for k,v in state.items():self.assertTrue(torch.equal(v,recovered['optimizer']['state'][key][k]))
                self.assertEqual(hashlib.sha256(initial_path.read_bytes()).hexdigest(),initial_hash)
                # Strict identity still rejects changing settings on an existing run.
                cfg['training']['learning_rate']=2e-4;config.write_text(json.dumps(cfg))
                with self.assertRaisesRegex(ValueError,'Resume changed'):run(args)


if __name__=='__main__':unittest.main()
