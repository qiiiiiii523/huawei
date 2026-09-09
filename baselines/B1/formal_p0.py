"""Protocol-compatible strict P0: machine-I -> full d12 with learned baseline."""
import argparse,json,os,random,time,platform
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'strict_data'
LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
def seed(s):
 os.environ['PYTHONHASHSEED']=str(s); random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def block(a,b): return nn.Sequential(nn.Conv1d(a,b,5,padding=2),nn.GroupNorm(max(1,min(8,b//4)),b),nn.SiLU(),nn.Conv1d(b,b,5,padding=2),nn.GroupNorm(max(1,min(8,b//4)),b),nn.SiLU())
class P0(nn.Module):
 def __init__(self,scale):
  super().__init__(); self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32)); w=(24,48,96,192); self.enc=nn.ModuleList([block(a,b) for a,b in zip((1,)+w[:-1],w)]); self.dec=nn.ModuleList([block(w[i+1]+w[i],w[i]) for i in (2,1,0)]); self.out=nn.Conv1d(w[0],12,1); self.base=nn.Sequential(nn.Linear(w[-1]+5,96),nn.SiLU(),nn.Dropout(.1),nn.Linear(96,12))
 def forward(self,x,stats):
  sk=[]
  for i,l in enumerate(self.enc): x=l(x if i==0 else F.max_pool1d(x,2)); sk.append(x)
  z=x
  for l,s in zip(self.dec,reversed(sk[:-1])): z=l(torch.cat((F.interpolate(z,size=s.shape[-1],mode='linear',align_corners=False),s),1))
  morph=self.out(z); pooled=F.adaptive_avg_pool1d(sk[-1],1).flatten(1); base=self.base(torch.cat((pooled,stats),1)); return morph,base
def scale_fit(a): return np.maximum(np.median(np.percentile(a,95,2)-np.percentile(a,5,2),0),25).astype('float32')
def trans(a,s):
 b=np.median(a,2,keepdims=True).astype('float32'); return np.clip((a-b)/s[None,:,None],-12,12).astype('float32')
def stats(a,s):
 q5=np.percentile(a[:,0],5,1); q95=np.percentile(a[:,0],95,1); return np.stack([np.median(a[:,0],1),a[:,0].mean(1),a[:,0].std(1),q5,q95],1).astype('float32')/float(s[0])
def pear(x,y):
 x=x.astype('float64').ravel(); y=y.astype('float64').ravel(); x-=x.mean(); y-=y.mean(); d=np.sqrt((x*x).sum()*(y*y).sum()); return float((x*y).sum()/d) if d else 0.0
def evalm(p,t):
 r=[pear(p[:,i],t[:,i]) for i in range(12)]; rm=[float(np.sqrt(np.mean((p[:,i]-t[:,i])**2))) for i in range(12)]; return {'r_raw_12':float(np.mean(r)),'r_submit_12':float(np.mean([1.0]+r[1:])),'r_missing11':float(np.mean(r[1:])),'rmse_raw_12':float(np.mean(rm)),'rmse_missing11':float(np.mean(rm[1:])),'per_lead_r':dict(zip(LEADS,r))}
def protocol_loss(pred,tgt,anchor,scale):
 hub=F.huber_loss(pred,tgt)
 p=pred-pred.mean(2,keepdim=True); q=tgt-tgt.mean(2,keepdim=True); corr=(p*q).sum(2)/(torch.sqrt((p.square().sum(2)*q.square().sum(2)).clamp_min(1e-8))); pcc=1-corr.mean()
 u=pred*scale.view(1,12,1); i,ii,iii,avr,avl,avf=[u[:,k] for k in range(6)]; res=torch.stack((iii-(ii-i),avr+(i+ii)/2,avl-(i-ii/2),avf-(ii-i/2)),1); res=res-res.median(2,keepdim=True).values; rs=scale[torch.tensor([2,3,4,5],device=pred.device)].view(1,4,1); phys=(res/rs).square().mean(); obs=F.huber_loss(pred[:,:1],anchor)
 return hub+0.1*pcc+0.05*phys+0.02*obs
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--out',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--data-dir',default=None,help='external strict_data directory'); a=ap.parse_args(); seed(a.seed); dev=torch.device(a.device); data=Path(a.data_dir) if a.data_dir else DATA; tr=np.load(data/'strict_train_target.npy',mmap_mode='r').astype('float32'); va=np.load(data/'strict_validation_target.npy',mmap_mode='r').astype('float32'); sc=scale_fit(tr); x=trans(tr[:,:1],sc[:1]); y=trans(tr,sc); xv=trans(va[:,:1],sc[:1]); st=stats(tr[:,:1],sc[:1]); sv=stats(va[:,:1],sc[:1]); base=np.median(tr,2).astype('float32')/sc[None,:]; model=P0(sc).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=5e-4); dl=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(base)),batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed)); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); best=-1e9; hist=[]; t0=time.time()
 for ep in range(1,a.epochs+1):
  model.train(); ls=[]
  for xb,yb,bb in dl:
   xb,yb,bb=xb.to(dev),yb.to(dev),bb.to(dev); opt.zero_grad(set_to_none=True); m,b=model(xb,bb[:,:5]); loss=protocol_loss(m,yb,xb,torch.as_tensor(sc,device=dev)); loss=loss+.1*F.smooth_l1_loss(b,bb); loss.backward(); opt.step(); ls.append(float(loss.detach().cpu()))
  model.eval(); pred=[]
  with torch.no_grad():
   for j,(xb,) in enumerate(DataLoader(TensorDataset(torch.from_numpy(xv)),batch_size=a.batch_size)):
    m,b=model(xb.to(dev),torch.from_numpy(sv[j*a.batch_size:j*a.batch_size+len(xb)]).to(dev)); pred.append((m.cpu().numpy()*sc[None,:,None]+b.cpu().numpy()[:,:,None]*sc[None,:,None]))
  p=np.concatenate(pred); met=evalm(p,va); row={'epoch':ep,'loss':float(np.mean(ls)),**met}; hist.append(row); print(json.dumps(row),flush=True)
  if met['r_submit_12']>best: best=met['r_submit_12']; torch.save({'model':model.state_dict(),'scale_uV':sc,'epoch':ep,'metric':best,'architecture_id':'B1_resdilated_p0_full12_baseline'},out/'checkpoint_best.pt')
 best_epoch=int(np.argmax([h['r_submit_12'] for h in hist])+1); state=torch.load(out/'checkpoint_best.pt',map_location=dev,weights_only=False); model.load_state_dict(state['model']); model.eval(); pred=[]
 with torch.no_grad():
  for j,(xb,) in enumerate(DataLoader(TensorDataset(torch.from_numpy(xv)),batch_size=a.batch_size)):
   m,b=model(xb.to(dev),torch.from_numpy(sv[j*a.batch_size:j*a.batch_size+len(xb)]).to(dev)); pred.append(m.cpu().numpy()*sc[None,:,None]+b.cpu().numpy()[:,:,None]*sc[None,:,None])
 p=np.concatenate(pred); np.save(out/'prediction_E1_raw_uV.npy',p); e2=p.copy(); e2[:,0]=va[:,0]; np.save(out/'prediction_E2_copy_at_eval_uV.npy',e2); summary={'status':'completed','protocol_stage':'P0_anchor_only','architecture_id':'B1_resdilated_p0_full12_baseline','best_epoch':best_epoch,'best_validation_r_submit_12':best,'n_train':len(tr),'n_validation':len(va),'seed':a.seed,'history':hist,'E1':evalm(p,va),'E2':evalm(e2,va),'elapsed_seconds':time.time()-t0,'environment':{'python':platform.python_version(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(dev)}}; (out/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__': main()
