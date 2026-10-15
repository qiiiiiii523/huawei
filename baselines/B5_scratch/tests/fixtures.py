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
from baselines.B5_scratch import PROTOCOL,SCORE_PROTOCOL,FULL_SCORE_PROTOCOL,INPUTS,GROUPS
from baselines.B5_scratch.common import write,read,sha,state_hash,csv_write
from baselines.B5_scratch.model import make_model,BoundModel
from baselines.B5_scratch.slow import SlowCNN,predict_native,slow_window,load_slow,validate_slow_pair
from baselines.B5_scratch.data import ConditionedDataset,verify_inputs
from baselines.B5_scratch.prepare import prepare,load_shared
from baselines.B5_scratch.runtime import run,ema_evaluation,update_best,CANDIDATES
from baselines.B5_scratch.checkpoint import load
from baselines.B5_scratch.check import check
from baselines.B5_scratch.predict import predict_record
from baselines.B5_scratch.evaluate import evaluate
from baselines.B5_scratch.compare import compare
from baselines.B5_scratch.scoring import score_tasks
from ecg12gen.evaluate import evaluate_record_predictions

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
        mask=np.ones(12,bool)
        return {'anchor':anchor,**self.meta,'target':target,'quality_mask':mask,
            'target_uV':target*(np.arange(1,13,dtype=np.float32)*100)[:,None],
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
        public=copy.deepcopy(fm);public['stage']='public';public['epoch']=99
        public['ema'][next(iter(public['ema']))].add_(.01)
        self.public=self.root/'public.pt';torch.save(public,self.public)
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
        self.slow_f=self.root/'F_best_score.pt'
        slow_cfg={'context_seconds':50,'group':'C','model_channels':48,'prepared_sha256':sha(self.prepared/'prepared.json'),
            'init_checkpoint_sha256':'same_source_S30','initial_weights_sha256':'same_initial_weights','seed':42,'epochs':20,
            'batch_records':2,'validate_every':5,'learning_rate':1e-4,'weight_decay':.01,'huber_weight':1.,'pcc_weight':.1,
            'mean_weight':0.,'residual_weight':0.}
        slow_state=SlowCNN().state_dict()
        torch.save({'protocol':'B5-slow-probe-v1','model':slow_state,'config':slow_cfg},self.slow)
        f_state={k:v.clone() for k,v in slow_state.items()};f_state['output.bias']+=.25
        torch.save({'protocol':'B5-slow-probe-v1','model':f_state,
            'config':{**slow_cfg,'group':'F','residual_weight':2.5890171538829843}},self.slow_f)

    def prepare(self):
        preprocessor=SimpleNamespace(scale_uV_by_source={'d12':self.scale,'ecg_machine_i':self.scale[:1]})
        with patch('baselines.B5_scratch.prepare.source_datasets',return_value=(preprocessor,self.train,self.val)),patch(
                'baselines.B5.runtime.sample',side_effect=lambda model,data,keys,*args,**kwargs:fake_sample(model,data,keys)),patch(
                'baselines.B5.runtime.common_config',return_value=SimpleNamespace(path=lambda key:self.root/'source.csv')):
            prepare(SimpleNamespace(fm_checkpoint=str(self.fm),public_checkpoint=str(self.public),epochs=100,
                slow_checkpoint=str(self.slow),slow_f_checkpoint=str(self.slow_f),
                prepared=str(self.prepared),output=str(self.shared),device='cpu'))
        return load_shared(self.shared)
