import argparse, json, random
from pathlib import Path
import numpy as np, torch
from train_candidates import Candidate, load_data, train_one

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--baseline',type=Path,required=True); p.add_argument('--scales',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=50); p.add_argument('--device',default='cuda'); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    scales=np.asarray(json.loads(a.scales.read_text())['scales_uV'],np.float32); data=load_data(a.cache,a.baseline,a.scales); tr=[z for z in data if z['split']=='train']; va=[z for z in data if z['split']=='validation']; random.seed(42); np.random.seed(42); torch.manual_seed(42); model=Candidate('gated').to(a.device); zero,best,state=train_one(model,tr,va,scales,a.epochs,a.device,True); torch.save({'model':state,'kind':'pcc500','zero_r':zero,'best_r':best,'train_records':len(tr),'validation_records':len(va),'train_subjects':len({z['subject'] for z in tr}),'validation_subjects':len({z['subject'] for z in va})},a.output/'best.pt'); (a.output/'result.json').write_text(json.dumps({'zero_r':zero,'best_r':best,'gain':best-zero,'train_records':len(tr),'validation_records':len(va),'train_subjects':len({z['subject'] for z in tr}),'validation_subjects':len({z['subject'] for z in va})},indent=2)); print(json.dumps({'zero_r':zero,'best_r':best,'gain':best-zero},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
