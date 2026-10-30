"""B5-final subject-disjoint gated/PCC500 slow-trend corrector experiment."""
from __future__ import annotations
import argparse, csv, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = (1, 6, 7, 8, 9, 10, 11)
IN = (0, 6, 7, 8, 9, 10, 11)

class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(width,width,3,padding=dilation,dilation=dilation),nn.GELU(),nn.Dropout(.1),nn.Conv1d(width,width,3,padding=dilation,dilation=dilation),nn.GELU(),nn.Dropout(.1))
    def forward(self,x): return x+self.net(x)

class Global(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(32); self.attn=nn.MultiheadAttention(32,4,dropout=.1,batch_first=True); self.proj=nn.Linear(32,32)
    def forward(self,x):
        n=x.shape[-1]; z=F.adaptive_avg_pool1d(x,120).transpose(1,2); z=self.norm(z); z,_=self.attn(z,z,z,need_weights=False); z=self.proj(z).transpose(1,2); return x+F.interpolate(z,size=n,mode='linear',align_corners=False)

class Candidate(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind=kind
        enc=lambda c: nn.Sequential(nn.Conv1d(c,32,5,padding=2),nn.GELU(),nn.Conv1d(32,32,5,padding=2),nn.GELU())
        self.ie=enc(2); self.pe=enc(8); self.gate=nn.Conv1d(64,32,1); nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias,-2.)
        self.local=nn.Sequential(*(Block(32,d) for d in (1,2,4,8)))
        self.wide=nn.Sequential(*(Block(32,d) for d in (1,4,12,24))) if kind=='gated' else None
        self.merge=nn.Conv1d(64,32,1) if self.wide else nn.Identity(); self.global_context=Global(); self.head=nn.Conv1d(32,7,1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,x):
        i=F.avg_pool1d(self.ie(x[:,[0,8]]),5,5); p=F.avg_pool1d(self.pe(x[:,1:]),5,5); g=torch.sigmoid(self.gate(torch.cat((p,i),1))); h=p+g*i; l=self.local(h); l=self.merge(torch.cat((l,self.wide(h)),1)) if self.wide else l; return self.head(self.global_context(l))

def compose(i, direct):
    ii=direct[:,0]; return torch.stack((i[:,0],ii,ii-i[:,0],-(i[:,0]+ii)/2,i[:,0]-ii/2,ii-i[:,0]/2,direct[:,1],direct[:,2],direct[:,3],direct[:,4],direct[:,5],direct[:,6]),1)

def corr(a,b):
    a=a-a.mean(-1,keepdim=True); b=b-b.mean(-1,keepdim=True); return (a*b).sum(-1)/(a.square().sum(-1).clamp_min(1e-6)*b.square().sum(-1).clamp_min(1e-6)).sqrt()

def load_data(cache, baseline, bundle):
    idx=json.loads((cache/'index.json').read_text())
    # scales are supplied separately by caller; cache itself is target-free.
    data=[]
    for row in idx['records']:
        p=np.load(cache/row['prediction_file']).astype('float32'); fast=np.load(cache/row['fast_file']).astype('float32'); slow=np.load(cache/row['slow_file']).astype('float32'); task=row['task_id']; inds=row['window_indices'];
        tar=np.load(baseline/task/'target_uV.npy',mmap_mode='r'); qual=np.load(baseline/task/'target_quality_mask.npy',mmap_mode='r'); y=np.concatenate([tar[i] for i in inds],-1).astype('float32'); m=np.all(qual[inds],0)
        data.append({'p':p,'fast':fast,'slow':slow,'y':y,'mask':m,'subject':row['subject_id'],'split':row['split'],'task':task})
    return data

def make_x(item, scales):
    branch=item['slow'].copy(); branch[0]=item['fast'][0]; x=np.median(branch.reshape(12,6000,10),-1)/scales[:,None]; x=np.concatenate([x[[0]],x[list(OUT)]],0); pos=np.linspace(-1,1,6000,dtype='float32')[None]; return torch.from_numpy(np.concatenate([x,pos],0).astype('float32'))

def train_one(model, train, val, scales, epochs, device, exact_pcc):
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=(-1,None); zero=None
    def score(items):
        model.eval(); vals=[]
        with torch.no_grad():
            for z in items:
                x=make_x(z,scales)[None].to(device); d=model(x); d=F.interpolate(d,size=60000,mode='linear',align_corners=False); d=d-d.mean(-1,keepdim=True); base=torch.from_numpy(z['p'][list(OUT)]).to(device)[None]+d*torch.from_numpy(scales[list(OUT)]).to(device)[None,:,None]; i=torch.from_numpy(z['p'][:1]).to(device)[None]; q=compose(i,base)[0,1:]; y=torch.from_numpy(z['y'][1:]).to(device); r=corr(q,y); vals.extend(r[torch.from_numpy(z['mask'][1:]).to(device)].cpu().tolist())
        return float(np.mean(vals))
    zero=score(val)
    for ep in range(epochs+1):
        if ep%10==0:
            s=score(val)
            if s>best[0]: best=(s,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if ep==epochs: break
        model.train(); random.shuffle(train)
        for st in range(0,len(train),16):
            batch=train[st:st+16]; x=torch.stack([make_x(z,scales) for z in batch]).to(device); d=model(x); base=torch.stack([torch.from_numpy(z['p'][list(OUT)]).reshape(7,6000,10).mean(-1) for z in batch]).to(device); target=torch.stack([torch.from_numpy(z['y'][list(OUT)]).reshape(7,6000,10).mean(-1) for z in batch]).to(device); m=torch.stack([torch.from_numpy(z['mask'][list(OUT)]) for z in batch]).to(device); scale7=torch.from_numpy(scales[list(OUT)]).to(device)[None,:,None]; d6000=F.interpolate(d,size=6000,mode='linear',align_corners=False); q=base+d6000*scale7; err=q-target; smooth=F.avg_pool1d(F.pad(err,(25,24),mode='replicate'),50,1); trend=(F.smooth_l1_loss(smooth,torch.zeros_like(smooth),reduction='none')*m[:,:,None]).mean(); qfull=base.repeat_interleave(10,-1)+F.interpolate(d*scale7,size=60000,mode='linear',align_corners=False); yfull=torch.stack([torch.from_numpy(z['y'][list(OUT)]) for z in batch]).to(device); pcc=(1-corr(qfull,yfull)).mean() if exact_pcc else (1-corr(q,target)).mean(); loss=trend+.1*pcc+.01*d.square().mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    return zero,best[0],best[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--baseline',type=Path,required=True); ap.add_argument('--scales',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--device',default='cuda'); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=False); scales=np.asarray(json.loads(args.scales.read_text())['scales_uV'],np.float32); data=load_data(args.cache,args.baseline,args.scales); subjects=sorted({z['subject'] for z in data}); folds=np.array_split(subjects,5); results={}
    for kind in ('gated','pcc500'):
        results[kind]=[]
        for fi,fold in enumerate(folds):
            valset=set(fold.tolist()); tr=[z for z in data if z['subject'] not in valset]; va=[z for z in data if z['subject'] in valset]; model=Candidate('gated').to(args.device); zero,best,state=train_one(model,tr,va,scales,args.epochs,args.device,kind=='pcc500'); torch.save({'model':state,'kind':kind,'fold':fi,'zero_r':zero,'best_r':best},args.output/f'{kind}_fold{fi}.pt'); results[kind].append({'fold':fi,'subjects':len(valset),'zero_r':zero,'best_r':best,'gain':best-zero}); print(kind,fi,results[kind][-1],flush=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2));
if __name__=='__main__': main()
