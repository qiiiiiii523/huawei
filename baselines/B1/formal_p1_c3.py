"""C3 adaptation runner built on formal_p0.P0; context is conditioning only."""
import argparse,json,os,random,time,csv
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from formal_p0 import P0,scale_fit,trans,stats,evalm,protocol_loss

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
class C3(nn.Module):
 def __init__(self,p0,ctx_ch,ctx_scale):
  super().__init__(); self.p0=p0; self.register_buffer('ctx_scale',torch.tensor(ctx_scale,dtype=torch.float32)); self.ce=nn.Sequential(nn.Conv1d(ctx_ch,32,9,padding=4),nn.SiLU(),nn.Conv1d(32,64,9,padding=4),nn.SiLU(),nn.AdaptiveAvgPool1d(1)); self.film=nn.Linear(64,24); self.gate=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,12)); self.delta=nn.Sequential(nn.Linear(64+12,96),nn.SiLU(),nn.Linear(96,12)); nn.init.zeros_(self.gate[-1].weight); nn.init.constant_(self.gate[-1].bias,-2.944439); nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)
 def forward(self,x,st,c):
  m,b=self.p0(x,st); z=self.ce(c).flatten(1); gb=self.film(z); gam,beta=gb[:,:12],gb[:,12:]; gate=torch.sigmoid(self.gate(z)); d=self.delta(torch.cat((z,m.mean(2)),1)); return m*(1+0.05*torch.tanh(gam)[:,:,None])+0.05*beta[:,:,None]+gate[:,:,None]*d[:,:,None],b
def load_arr(root,task,split,body=False,body_raw=False):
 p=Path(root)
 if not body and task=='task1': return np.load(p/f'{task}_{split}_input.npy',mmap_mode='r').astype('float32'),np.load(p/f'{task}_{split}_target.npy',mmap_mode='r').astype('float32')
 if not body and task=='task2':
  c=np.load(p/f'{task}_{split}_input.npy',mmap_mode='r').astype('float32'); full=np.load(p/f'{task}_{split}_target.npy',mmap_mode='r').astype('float32'); kind='body_scale_d6' if body_raw else 'ecg_machine_d6'; rows=[r for r in csv.DictReader(open(p/'task2_window_metadata.csv',encoding='utf-8-sig')) if r['split']==split and r['input_type']==kind]; rows.sort(key=lambda r:int(r['array_index'])); idx=[int(r['array_index']) for r in rows]; return c[idx],full[idx]
 c=np.load(p/f'body_scale_{split}_input_B_raw_detrended_0p2Hz.npy',mmap_mode='r').astype('float32'); full=np.load(p/f'{task}_{split}_target.npy',mmap_mode='r').astype('float32'); rows=[r for r in csv.DictReader(open(p/'window_metadata_B_raw.csv',encoding='utf-8-sig')) if r['split']==split]; rows.sort(key=lambda r:int(r['local_array_index'])); idx=[int(r['canonical_array_index']) for r in rows]; return c,full[idx]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--task',choices=['task1','task2'],required=True); ap.add_argument('--context-root',required=True); ap.add_argument('--p0',required=True); ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--device',default='cuda'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--body-raw',action='store_true'); a=ap.parse_args(); seed(a.seed); dev=torch.device(a.device); cr=Path(a.context_root); is_body=a.task=='task2' and ('body' in a.out.lower()); trc,try_=load_arr(cr,a.task,'train',body=is_body and not a.body_raw,body_raw=is_body and a.body_raw); vac,vay=load_arr(cr,a.task,'validation',body=is_body and not a.body_raw,body_raw=is_body and a.body_raw); sc=scale_fit(try_); x=trans(try_[:,:1],sc[:1]); y=trans(try_,sc); xv=trans(vay[:,:1],sc[:1]); csc=scale_fit(trc); c=trans(trc,csc); cv=trans(vac,csc); st=stats(try_[:,:1],sc[:1]); sv=stats(vay[:,:1],sc[:1]); bb=np.median(try_,2).astype('float32')/sc[None,:]; model=C3(P0(sc),trc.shape[1],csc).to(dev); state=torch.load(a.p0,map_location=dev,weights_only=False); model.p0.load_state_dict(state['model']); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-4); dl=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(bb),torch.from_numpy(c)),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(a.seed)); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); best=-1e9; hist=[]
 for ep in range(1,a.epochs+1):
  model.train(); ls=[]
  for xb,yb,base,cb in dl:
   xb,yb,base,cb=xb.to(dev),yb.to(dev),base.to(dev),cb.to(dev); opt.zero_grad(set_to_none=True); m,b=model(xb,base[:,:5],cb); loss=protocol_loss(m,yb,xb,torch.as_tensor(sc,device=dev))+.1*F.smooth_l1_loss(b,base); loss.backward(); opt.step(); ls.append(float(loss.detach().cpu()))
  model.eval(); outp=[]
  with torch.no_grad():
   for j in range(0,len(xv),16):
    xb=torch.from_numpy(xv[j:j+16]).to(dev); m,b=model(xb,torch.from_numpy(sv[j:j+len(xb)]).to(dev),torch.from_numpy(cv[j:j+len(xb)]).to(dev)); outp.append(m.cpu().numpy()*sc[None,:,None]+b.cpu().numpy()[:,:,None]*sc[None,:,None])
  pred=np.concatenate(outp); met=evalm(pred,vay); row={'epoch':ep,'loss':float(np.mean(ls)),**met}; hist.append(row); print(json.dumps(row),flush=True)
  if met['r_submit_12']>best: best=met['r_submit_12']; torch.save({'model':model.state_dict(),'epoch':ep,'metric':best,'p0_checkpoint':a.p0,'protocol_stage':'P1-C3'},out/'checkpoint_best.pt')
 model.eval(); state=torch.load(out/'checkpoint_best.pt',map_location=dev,weights_only=False); model.load_state_dict(state['model']); outp=[]
 with torch.no_grad():
  for j in range(0,len(xv),16):
   xb=torch.from_numpy(xv[j:j+16]).to(dev); m,b=model(xb,torch.from_numpy(sv[j:j+len(xb)]).to(dev),torch.from_numpy(cv[j:j+len(xb)]).to(dev)); outp.append(m.cpu().numpy()*sc[None,:,None]+b.cpu().numpy()[:,:,None]*sc[None,:,None])
 pred=np.concatenate(outp); np.save(out/'prediction_E1_raw_uV.npy',pred); e2=pred.copy(); e2[:,0]=vay[:,0]; np.save(out/'prediction_E2_copy_at_eval_uV.npy',e2); (out/'run_summary.json').write_text(json.dumps({'status':'completed','protocol_stage':'P1-C3','task':a.task,'best_validation_r_submit_12':best,'best_epoch':state['epoch'],'E1':evalm(pred,vay),'E2':evalm(e2,vay),'history':hist},indent=2)+'\n')
if __name__=='__main__': main()
