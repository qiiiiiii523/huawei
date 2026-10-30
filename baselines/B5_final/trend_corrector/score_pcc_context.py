import argparse,csv,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,'/home/qht/train_scheduler/B5-residual-networks-20261030')
from train_candidates import Candidate,make_x,compose,corr,OUT

def mmap_map(path):
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    return {(r['pair_id'],r['start_sample_500hz']):i for i,r in enumerate(rows)},rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); root=a.root; cache=root/'cache_v2'; targetroot=cache/'targets'; study=Path('/home/qht/B5-final-tcn7-20261030/final_study'); base=Path('/home/qht/B5-final-tcn7-20261030/base_inference'); scales=np.asarray(json.load(open('/home/qht/B5-residual-networks-20261030/scales.json'))['scales_uV'],np.float32); model=Candidate('gated'); model.load_state_dict(torch.load(root/'pcc500_formal_v2/best.pt',map_location='cpu',weights_only=False)['model']); model.eval(); idx=json.load(open(cache/'index.json')); scores={k:[] for k in ('base','pcc','context')}; checked=[]
    for task in ('task1','task2'):
        sm,srows=mmap_map(study/'both'/task/'window_metadata.csv'); bm,brows=mmap_map(base/task/'window_metadata.csv'); sp=np.load(study/'both'/task/'prediction_uV.npy',mmap_mode='r'); bp=np.load(base/task/'prediction_uV.npy',mmap_mode='r'); tar=np.load(targetroot/task/'target_uV.npy',mmap_mode='r'); qual=np.load(targetroot/task/'target_quality_mask.npy',mmap_mode='r')
        for row in idx['records']:
            if row['task_id']!=task or row['split']!='validation': continue
            p0=np.load(cache/row['prediction_file']); fast=np.load(cache/row['fast_file']); slow=np.load(cache/row['slow_file']); starts=[str(i*5000) for i in range(12)]; keys=[(row['pair_id'],s) for s in starts]
            if any(k not in sm or k not in bm for k in keys): raise ValueError(f'context/base metadata mismatch: {row["pair_id"]}')
            si=[sm[k] for k in keys]; bi=[bm[k] for k in keys];
            if [srows[i]['start_sample_500hz'] for i in si] != [brows[i]['start_sample_500hz'] for i in bi]: raise ValueError('start alignment failed')
            target=np.concatenate([tar[i] for i in row['window_indices']],-1); mask=np.all(qual[row['window_indices']],0)
            with torch.no_grad(): d=model(make_x({'fast':fast,'slow':slow},scales)[None]); d=torch.nn.functional.interpolate(d,size=60000,mode='linear',align_corners=False); d=d-d.mean(-1,keepdim=True); pcc=p0.copy(); pcc[list(OUT)]+=d[0].numpy()*scales[list(OUT),None]
            ctx_delta=np.concatenate([sp[i]-bp[i] for i in si],-1); outputs={'base':p0,'pcc':pcc,'context':pcc+ctx_delta}
            for name,z in outputs.items():
                q=compose(torch.from_numpy(z[:1])[None],torch.from_numpy(z[list(OUT)])[None])[0].numpy()
                for lead in range(1,12):
                    if mask[lead]: scores[name].append(float(corr(torch.from_numpy(q[lead:lead+1]),torch.from_numpy(target[lead:lead+1]))[0]))
            checked.append({'task':task,'pair_id':row['pair_id'],'windows':12})
    result={'checked_records':len(checked),'checked_windows':len(checked)*12,**{k:float(np.mean(v)) for k,v in scores.items()},'context_gain_vs_pcc':float(np.mean(scores['context'])-np.mean(scores['pcc'])),'alignment':'pair_id+start_sample_500hz'}; a.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
