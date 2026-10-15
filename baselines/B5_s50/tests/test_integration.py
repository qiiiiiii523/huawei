"""Synthetic functional checks only. No optimizer update or ECG training."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5 import ARCHITECTURE_ID,CONDITION_SCHEMA,PREPROCESSING_VERSION
from baselines.B5.checkpoint import scales_digest,pack_rng,restore_rng
from baselines.B5.config import ModelConfig
from baselines.B5.flow import sample
from baselines.B5.losses import flow_loss
from baselines.B5.model import B5UNet
from baselines.B5.metadata import encode_demographics
from baselines.B5_s50 import PROTOCOL,SCORE_PROTOCOL,INPUTS
from baselines.B5_s50.common import write,read,sha,state_hash,csv_write
from baselines.B5_s50.model import make_model,BoundModel
from baselines.B5_s50.slow import SlowCNN,predict_native,slow_window,load_slow
from baselines.B5_s50.data import ConditionedDataset,verify_inputs
from baselines.B5_s50.prepare import prepare,load_shared
from baselines.B5_s50.runtime import run,ema_evaluation,update_best,CANDIDATES
from baselines.B5_s50.checkpoint import load
from baselines.B5_s50.check import check
from baselines.B5_s50.predict import predict_record
from baselines.B5_s50.evaluate import evaluate
from baselines.B5_s50.compare import compare

torch.set_num_threads(2)
CONFIG=ModelConfig(channels=(8,16,24),time_dim=16,metadata_dim=16,dilations=(1,))
LOSS={'huber':.5,'pcc':.1,'anchor':.1,'physiology':.01,'huber_delta':1.}


def inputs(length=64,batch=2):
    return {'anchor':torch.linspace(-1,1,length)[None,None].repeat(batch,1,1),
        'numeric':torch.full((batch,3),.5),'sex':torch.zeros(batch,dtype=torch.long),
        'field_mask':torch.ones(batch,4,dtype=torch.bool),'age_topcoded':torch.zeros(batch,1)}


class Windows:
    def __init__(self,target_id,devices):
        self.meta=encode_demographics({'age':45,'sex':'male','height':170,'weight':65})
        self.full=(np.sin(np.arange(60000,dtype=np.float32)/70)*.5+np.arange(60000,dtype=np.float32)/60000)[None]
        self.rows=[{'pair_id':'pair_'+d,'target_record_id':target_id,'subject_id':target_id+'_patient',
            'start_sample_500hz':str(start),'expected_window_count':'12','input_type':d}
            for d in devices for start in range(0,60000,5000)]
        self.manifest_digest='synthetic_'+target_id+'_'+str(len(self.rows))

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];start=int(row['start_sample_500hz'])
        anchor=self.full[:,start:start+5000].copy()
        target=np.repeat(anchor,12,axis=0)*np.arange(1,13,dtype=np.float32)[:,None]
        target[9]=np.cos(np.arange(start,start+5000,dtype=np.float32)/45)*2
        mask=np.ones(12,bool);mask[9]=False
        return {'anchor':anchor,**self.meta,'target':target,'quality_mask':mask,
            'key':row['target_record_id']+':'+str(start),'evaluation_metadata':row}


def fake_sample(model,data,keys,**settings):
    """Deterministic stand-in used only to test file/score plumbing at full size."""
    assert not model.training
    return data['anchor'].repeat(1,12,1)*torch.arange(1,13,device=data['anchor'].device)[None,:,None]


class Fixture:
    def __init__(self,root):
        self.root=Path(root)
        self.prepared=self.root/'prepared';self.prepared.mkdir()
        self.original=self.root/'original';self.original.mkdir()
        self.shared=self.root/'shared'
        self.train=Windows('train',('ecg_machine_d6',))
        self.val={'task1':Windows('validation',('watch',)),
                  'task2':Windows('validation',('ecg_machine_d6','body_scale_d6'))}
        self.scale=np.arange(1,13,dtype=np.float32)*100
        model=B5UNet(CONFIG)
        scales={'d12':self.scale.tolist(),'ecg_machine_i':[float(self.scale[0])]}
        training={'seed':42,'batch_size':8,'gradient_accumulation':1,'learning_rate':5e-5,'weight_decay':.01,
            'ema_decay':.999,'amp':False,'gradient_clip':1.,'workers':0,'deterministic':True,'epochs':100,'validate_every':5}
        cfg={'model':asdict(CONFIG),'loss':LOSS,'training':training,'validation':{'batch_size':8},'repository_root':str(self.root)}
        source=self.root/'source.csv';source.write_text('synthetic_source\n',encoding='utf-8')
        manifests={'train':self.train.manifest_digest,**{k:d.manifest_digest for k,d in self.val.items()},'source':sha(source)}
        fm={'architecture_id':ARCHITECTURE_ID,'architecture_hash':CONFIG.fingerprint,'condition_schema':CONDITION_SCHEMA,
            'preprocessing_version':PREPROCESSING_VERSION,'model_config':asdict(CONFIG),'scales':scales,'scales_sha256':scales_digest(scales),
            'model':model.state_dict(),'ema':model.state_dict(),'cache_protocol':'independent-record-context-v1','stage':'finetune',
            'training_config':training,'loss_config':LOSS,'config':cfg,'manifests':manifests,'epoch':44}
        self.fm=self.root/'best.pt';torch.save(fm,self.fm)
        metadata={'train':[{'target_record_id':'train','subject_id':'train_patient'}],
            'validation':[{'target_record_id':'validation','subject_id':'validation_patient','shared_with_task1':True}],
            'pairs':{task:[{'pair_id':d.rows[i]['pair_id'],'target_record_id':'validation','input_type':d.rows[i]['input_type'],
                            'indices':list(range(i,i+12))} for i in range(0,len(d),12)] for task,d in self.val.items()}}
        write(self.prepared/'records.json',metadata)
        for split,dataset in (('train',self.train),('validation',self.val['task1'])):
            arrays={'anchor':dataset.full.reshape(1,6000,10).mean(-1)[None]}
            arrays.update({k:np.asarray(v)[None] for k,v in dataset.meta.items()})
            np.savez(self.prepared/(split+'.npz'),**arrays,mask=np.ones((1,11),bool),
                     label=np.full((1,11,6000),np.nan,np.float32))
        files={}
        for task,d in self.val.items():
            folder=self.original/task;folder.mkdir()
            p=[];y=[]
            for i in range(len(d)):
                item=d[i]
                p.append(np.repeat(item['anchor'],12,axis=0)*np.arange(1,13,dtype=np.float32)[:,None]*self.scale[:,None])
                y.append(item['target']*self.scale[:,None])
            np.save(folder/'prediction_uV.npy',np.stack(p));np.save(folder/'target_uV.npy',np.stack(y))
            csv_write(folder/'window_metadata.csv',d.rows)
            # Deliberately misleading old score: prepare must re-score frozen arrays.
            csv_write(folder/'overall_metrics.csv',[{'r_missing11':-100.}])
            for name in ('prediction_uV.npy','target_uV.npy','window_metadata.csv','overall_metrics.csv'):
                files[task+'/'+name]=sha(folder/name)
        write(self.prepared/'prepared.json',{'protocol':'B5-slow-probe-v1','completed':True,'checkpoint_sha256':sha(self.fm),
            'files':{n:sha(self.prepared/n) for n in ('records.json','train.npz','validation.npz')},
            'evaluation_dir':str(self.original),'evaluation_files':files,'scale_uV':self.scale.tolist(),
            'source_checks':{'source':{'path':str(source),'sha256':sha(source)}}})
        self.slow=self.root/'best_score.pt'
        torch.save({'protocol':'B5-slow-probe-v1','model':SlowCNN().state_dict(),
            'config':{'context_seconds':50,'group':'C','model_channels':48,'prepared_sha256':sha(self.prepared/'prepared.json')}},self.slow)

    def prepare(self):
        with patch('baselines.B5_s50.prepare.source_datasets',return_value=(None,self.train,self.val)):
            prepare(SimpleNamespace(fm_checkpoint=str(self.fm),slow_checkpoint=str(self.slow),prepared=str(self.prepared),output=str(self.shared),device='cpu'))
        return load_shared(self.shared)


class IntegrationTests(unittest.TestCase):
    def test_zero_injection_velocity_ode_and_first_gradient(self):
        torch.manual_seed(42)
        original=B5UNet(CONFIG)
        a=make_model(CONFIG,'control',original.state_dict()).eval()
        b=make_model(CONFIG,'slow',original.state_dict()).eval()
        data=inputs();slow=torch.randn(2,11,64);state=torch.randn_like(slow);time=torch.tensor([.2,.8])
        with torch.no_grad():
            av,ai=BoundModel(a)(state,time,*[data[k] for k in INPUTS])
            bv,bi=BoundModel(b,slow)(state,time,*[data[k] for k in INPUTS])
            self.assertTrue(torch.equal(av,bv));self.assertTrue(torch.equal(ai,bi))
            ap=sample(BoundModel(a),data,['a','b'],steps=2,samples=2)
            bp=sample(BoundModel(b,slow),data,['a','b'],steps=2,samples=2)
            self.assertTrue(torch.equal(ap,bp))
        before=state_hash(b.state_dict());b.train()
        batch={**data,'target':torch.randn(2,12,64),'quality_mask':torch.ones(2,12,dtype=torch.bool)}
        flow_loss(BoundModel(b,slow),batch,LOSS,torch.ones(12))[0].backward()
        self.assertGreater(b.slow_adapter[-1].weight.grad.norm().item(),0)
        self.assertEqual(before,state_hash(b.state_dict()))
        with self.assertRaisesRegex(ValueError,'requires'): BoundModel(b)

    def test_metadata_dropout_rng_matches_at_zero_initialization(self):
        a=B5UNet(CONFIG).train();b=make_model(CONFIG,'slow',a.state_dict()).train()
        data=inputs();s=torch.randn(2,11,64);t=torch.tensor([.3,.7]);rng=pack_rng()
        av=BoundModel(a)(s,t,*[data[k] for k in INPUTS])[0];after=torch.rand(5)
        restore_rng(rng)
        bv=BoundModel(b,s)(s,t,*[data[k] for k in INPUTS])[0];other=torch.rand(5)
        self.assertTrue(torch.equal(av,bv));self.assertTrue(torch.equal(after,other))

    def test_frozen_s50_matches_manual_windows_and_ignores_labels(self):
        class Echo(torch.nn.Module):
            def forward(self,anchor,**metadata): return anchor.repeat(1,11,1)+metadata['numeric'][:,0,None,None]
        network=Echo().eval()
        data=inputs(6000,2);data['target']=torch.full((2,11,6000),float('nan'))
        result=predict_native(network,data)
        expected=data['anchor'].repeat(1,11,1)+data['numeric'][:,0,None,None]
        self.assertTrue(torch.equal(result,expected))
        data['target']=torch.randn(1)
        self.assertTrue(torch.equal(result,predict_native(network,data)))
        slow=SlowCNN().eval()
        self.assertEqual(sum(p.numel() for p in slow.parameters()),213511)
        cp={'protocol':'B5-slow-probe-v1','config':{'context_seconds':50,'group':'C'},'model':slow.state_dict()}
        frozen=load_slow(cp,'cpu')
        self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
        actual=predict_native(frozen,{k:v[:1] for k,v in data.items() if k in INPUTS})
        x=data['anchor'][:1];pad=torch.cat((x[...,:1].expand(-1,-1,1000),x,x[...,-1:].expand(-1,-1,1000)),dim=-1)
        with torch.no_grad():
            manual=torch.cat([frozen(pad[...,i*500:i*500+2500],**{k:data[k][:1] for k in INPUTS[1:]})[...,1000:1500] for i in range(12)],dim=-1)
        torch.testing.assert_close(actual,manual,atol=2e-6,rtol=2e-5)
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))

    def test_full_record_interpolation_preserves_position_and_seams(self):
        native=np.arange(6000,dtype=np.float32)[None].repeat(11,axis=0)+123
        combined=np.concatenate([slow_window(native,i) for i in range(0,60000,5000)],axis=-1)
        expected=np.stack([np.interp(np.arange(60000),np.arange(6000)*10+4.5,row) for row in native]).astype(np.float32)
        np.testing.assert_array_equal(combined,expected)
        self.assertEqual(combined[0,0],123)
        self.assertAlmostEqual(float(combined[0,5000]-combined[0,4999]),.1,places=3)

    def test_ema_evaluation_restores_weights_mode_and_rng_even_on_error(self):
        model=B5UNet(CONFIG).train();before=state_hash(model.state_dict());rng=pack_rng()
        reference=torch.rand(5);restore_rng(rng)
        with self.assertRaisesRegex(RuntimeError,'test'):
            with ema_evaluation(model,{k:v+1 for k,v in model.state_dict().items()}):
                torch.rand(20);raise RuntimeError('test')
        self.assertEqual(before,state_hash(model.state_dict()));self.assertTrue(model.training)
        self.assertTrue(torch.equal(reference,torch.rand(5)))

    def test_selectors_keep_initial_candidate_and_update_independently(self):
        best={k:{'value':-float('inf'),'epoch':None} for k in CANDIDATES}
        def scores(total,r,bonus): return {'competition_total_score':total,'main_correlation':r,'task2_rmse_bonus_score':bonus}
        self.assertEqual(set(update_best(best,scores(35,.68,1),0)),set(CANDIDATES))
        self.assertEqual(update_best(best,scores(34,.7,.5),5),['r'])
        self.assertEqual(update_best(best,scores(34,.65,2),10),['bonus'])
        self.assertEqual(best['score']['epoch'],0)

    def test_prepare_preflight_init_checkpoints_resume_and_final_pipeline_without_updates(self):
        class StopBeforeTraining(RuntimeError): pass
        class NoTrainLoader:
            def __init__(self,*a,**k): pass
            def __iter__(self): raise StopBeforeTraining('Synthetic test intentionally stops before first batch')
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(tmp);shared,info=fixture.prepare()
            self.assertEqual(info['training']['epochs'],20);self.assertEqual(info['training']['learning_rate'],5e-5)
            self.assertEqual(info['sampling']['samples'],16)
            self.assertGreater(info['source_summaries']['task1']['r_missing11'],0)
            check(SimpleNamespace(shared=shared,report=Path(tmp)/'check.json'))
            self.assertTrue(read(Path(tmp)/'check.json')['ready'])
            with self.assertRaises(FileExistsError): fixture.prepare()
            runs=[]
            mapped=[]
            base_datasets=(None,fixture.train,fixture.val)
            real_loader=__import__('torch.utils.data',fromlist=['DataLoader']).DataLoader
            def loader_factory(dataset,*a,**kw):
                mapped.append(dataset.native)
                return NoTrainLoader() if kw.get('shuffle') else real_loader(dataset,*a,**kw)
            # Both update entry points are blocked: no local training occurs.
            with patch('baselines.B5_s50.runtime.source_datasets',return_value=base_datasets), \
                 patch('baselines.B5_s50.runtime.DataLoader',side_effect=loader_factory), \
                 patch('baselines.B5_s50.runtime.sample',side_effect=fake_sample), \
                 patch('torch.optim.AdamW.step',side_effect=AssertionError('No optimizer update allowed')):
                for group in ('control','slow'):
                    folder=Path(tmp)/group;out=folder/'checkpoints';runs.append(folder)
                    args=SimpleNamespace(shared=shared,group=group,output=out,device='cpu',resume=None)
                    with self.assertRaises(StopBeforeTraining): run(args)
                    self.assertEqual(read(out/'initial_evaluation.json')['score']['competition_score_protocol'],SCORE_PROTOCOL)
                    for name in (*CANDIDATES.values(),'last.pt'):
                        cp=load(out/name);self.assertEqual(cp['epoch'],0)
                        self.assertEqual(cp['slow_state_sha256'],info['slow_state_sha256'])
                    last=load(out/'last.pt');self.assertEqual(last['updates'],0);self.assertEqual(last['optimizer']['state'],{})
                    # Crash recovery repairs a corrupted best alias from last.pt.
                    (out/'best.pt').write_bytes(b'interrupted')
                    args.resume=out/'last.pt'
                    with self.assertRaises(StopBeforeTraining): run(args)
                    self.assertEqual(load(out/'best.pt')['epoch'],0)
                    args.group='slow' if group=='control' else 'control'
                    with self.assertRaisesRegex(ValueError,'Exact resume'): run(args)
                    for name in ('best','best_r','best_bonus','last'):
                        evaluate(SimpleNamespace(shared=shared,checkpoint=out/(name+'.pt'),output=folder/'evaluations'/name,
                                                 device='cpu',reuse_root=folder/'evaluations',plots=False))
                    self.assertIsNotNone(read(folder/'evaluations/best_r/evaluation.json')['reused_identical_ema_from'])
                    # Synthetic fixture marks completion for comparison plumbing;
                    # it does not represent an actual completed 20-epoch experiment.
                    write(out/'completed.json',{'completed':True,'additional_epochs':20})
                compare(SimpleNamespace(shared=shared,control=runs[0],slow=runs[1],output=Path(tmp)/'comparison'))
                self.assertTrue((Path(tmp)/'comparison/report.md').exists())
            cached=np.load(shared/'train_slow.npy',mmap_mode='r')
            wrapped=ConditionedDataset(fixture.train,cached,read(shared/'records.json')['train'])
            self.assertEqual(wrapped[0]['slow'].shape,(11,5000))
            (shared/'records.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Shared cache'): load_shared(shared)
            del wrapped
            cached._mmap.close()
            for array in mapped: array._mmap.close()

    def test_deploy_checkpoint_only_i_metadata_no_prepared_or_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(tmp);shared,info=fixture.prepare()
            source=torch.load(fixture.fm,weights_only=True);network=torch.load(fixture.slow,weights_only=True)
            raw=(fixture.train.full*info['scales']['ecg_machine_i'][0]).copy()
            for group in ('control','slow'):
                model=make_model(CONFIG,group,source['ema'])
                cp={'group':group,'model':model.state_dict(),'model_config':asdict(CONFIG),
                    'sampling':info['sampling'],'slow_state':network['model'],'scales':info['scales']}
                calls=[]
                def observed(bound,data,keys,**settings):
                    self.assertEqual(set(data),set(INPUTS));self.assertEqual(settings,info['sampling'])
                    if group=='slow': self.assertEqual(bound.slow.shape,(1,11,5000))
                    calls.extend(keys);return fake_sample(bound,data,keys,**settings)
                with patch('baselines.B5_s50.predict.sample',side_effect=observed):
                    y=predict_record(cp,raw,{'age':45},'new_record','cpu')
                self.assertEqual(y.shape,(12,60000));self.assertEqual(calls,[f'new_record:{i}' for i in range(0,60000,5000)])
                expected=fixture.train.full*np.arange(1,13,dtype=np.float32)[:,None]*fixture.scale[:,None]
                np.testing.assert_allclose(y,expected,atol=2e-3,rtol=2e-6)
                with self.assertRaisesRegex(ValueError,'Test input'): predict_record(cp,np.zeros((12,60000)),{},'x','cpu')


if __name__=='__main__': unittest.main()
