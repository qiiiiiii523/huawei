"""Matched slow-FM rate/context/output experiments. Only generated state may cross leads."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from .native50 import Native50UNet,pack_i
from .config import ModelConfig
from .targets import GENERATED,missing11

def linear_expand(x,factor):
 if factor==1:return x
 pieces=[]
 for r in range(factor):
  delta=(r+.5)/factor-.5;neighbor=torch.cat((x[...,:1],x[...,:-1]),-1) if delta<0 else torch.cat((x[...,1:],x[...,-1:]),-1);pieces.append((1-abs(delta))*x+abs(delta)*neighbor)
 return torch.stack(pieces,-1).flatten(-2)

def restrict50(x,rate):
 if rate==50:return x
 factor=50//rate;n=20*factor+1;cut=.8/factor;p=np.arange(n)-(n-1)/2;h=cut*np.sinc(cut*p)*np.kaiser(n,5.);h/=h.sum();kernel=torch.as_tensor(h,dtype=x.dtype,device=x.device)[None,None].expand(x.shape[1],1,-1);pad=n//2;z=torch.cat((x[...,:1].expand(-1,-1,pad),x,x[...,-1:].expand(-1,-1,pad)),-1)
 return F.avg_pool1d(F.conv1d(z,kernel,groups=x.shape[1]),factor,factor)

class FMExperimentUNet(Native50UNet):
 def __init__(self,config=None,rate=5,input_seconds=120,output_seconds=120):
  if rate not in [50,10,5] or input_seconds not in [10,40,60,75,120] or output_seconds not in [10,30,60,75,120] or output_seconds>input_seconds:raise ValueError('Experiment contract')
  config=config or ModelConfig()
  minimum=config.min_bottleneck_points or 2**(len(config.channels)-1)
  if input_seconds*rate < minimum:raise ValueError(f"Input too short for configured U-Net levels; need >={minimum} native points")
  super().__init__(config);self.rate=rate;self.input_seconds=input_seconds;self.output_seconds=output_seconds
 def forward(self,state,time,anchor,metadata):
  ni=self.input_seconds*self.rate;no=self.output_seconds*self.rate;left=(ni-no)//2
  if state.shape!=(len(anchor),7,no) or anchor.shape[-1]!=self.input_seconds*500:raise ValueError('FM state/visible input coordinate mismatch')
  padded=F.pad(state,(left,ni-no-left));dense=linear_expand(padded,50//self.rate)
  if self.rate==50:
   allp=super().forward(dense,time,anchor,metadata);return allp[...,left:left+no]
  c=self.core;te=c.time_embedding(time);me=c.metadata_encoder(anchor,metadata)
  h=restrict50(c.stem(torch.cat((pack_i(anchor),dense),1)),self.rate)
  return c.forward_features(h,te,me)[...,left:left+no]


def build(config=None,rate=5,input_seconds=120,output_seconds=120):return FMExperimentUNet(config,rate,input_seconds,output_seconds)

@torch.no_grad()
def generate_native(model,raw_i,meta,scales,noise_std,kind,record_key,steps=16,samples=16,trajectory_batch=16):
 if kind!='FM' or samples%2 or 120%model.output_seconds or np.shape(raw_i)!=(1,60000):raise ValueError('Sampling contract')
 model.eval();device=next(model.parameters()).device;ni=model.input_seconds*model.rate;no=model.output_seconds*model.rate;left_seconds=(model.input_seconds-model.output_seconds)//2;pieces=[]
 for start in range(0,120,model.output_seconds):
  index=np.arange((start-left_seconds)*500,(start-left_seconds+model.input_seconds)*500).clip(0,59999);anchor=torch.as_tensor((np.asarray(raw_i,np.float64)[:,index]/scales[0]).copy(),dtype=torch.float32,device=device)[None];fields={k:torch.as_tensor(v,device=device)[None] for k,v in meta.items()};key=record_key if model.input_seconds==model.output_seconds==120 else record_key+':core'+str(start)
  seed=int.from_bytes(hashlib.sha256(('huawei-series1|42|'+key).encode()).digest()[:8],'little')%(2**63-1);gen=torch.Generator().manual_seed(seed);z=torch.randn(samples//2,7,no,generator=gen)*torch.as_tensor(noise_std,dtype=torch.float32)[None,:,None];starts=torch.stack((z,-z),1).flatten(0,1);total=torch.zeros(7,no,dtype=torch.float64,device=device)
  for offset in range(0,samples,trajectory_batch):
   state=starts[offset:offset+trajectory_batch].to(device);n=len(state);a=anchor.expand(n,-1,-1);m={k:v.expand(n,*v.shape[1:]) for k,v in fields.items()}
   for j in range(steps):
    dt=1/steps;t=anchor.new_full((n,),j/steps);v=model(state,t,a,m);w=model(state+dt*v,t+dt,a,m);state=state+.5*dt*(v+w)
   total+=state.double().sum(0)
  pieces.append((total/samples).cpu().numpy())
 return np.concatenate(pieces,-1)

@torch.no_grad()
def predict_slow(model,raw_i,meta,scales,noise_std,kind,record_key,**kw):
 p=generate_native(model,raw_i,meta,scales,noise_std,kind,record_key,**kw);dense=F.interpolate(torch.from_numpy(p)[None],size=60000,mode='linear',align_corners=False)[0].numpy()*np.asarray(scales)[list(GENERATED),None];return missing11(dense,raw_i)
