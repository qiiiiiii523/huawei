"""QC v3 selection plus full-population v2 on the same predictions."""
from __future__ import annotations

import numpy as np
from baselines.B5_best.scoring import correlation_score
from baselines.B5.selection import scores_from_summaries
from . import FULL_SCORE_PROTOCOL
from .common import csv_write,write,finite_json

LEADS=('II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')


def score_tasks(summaries,qc=True):
    if qc: return scores_from_summaries(summaries)
    a,b=summaries['task1'],summaries['task2']
    if a.get('quality_mask_applied') or b.get('quality_mask_applied'): raise ValueError('Cannot label QC summaries as full-population')
    score=correlation_score(a['r_missing11'],b['r_missing11'])
    score['score_protocol']=FULL_SCORE_PROTOCOL
    return score


def write_score(output,score):
    csv_write(output/'correlation_score.csv',[score])
    write(output/'correlation_score.json',score)



def slow(x):
    pad=np.pad(np.asarray(x,float),((0,0),(250,250)),mode='edge')
    total=np.concatenate((np.zeros((len(x),1)),np.cumsum(pad,axis=1)),axis=1)
    return (total[:,501:]-total[:,:-501])/501


def correlations(p,y):
    a,b=p-p.mean(-1,keepdims=True),y-y.mean(-1,keepdims=True)
    den=np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    return np.divide((a*b).sum(-1),den,out=np.full(den.shape,np.nan),where=den>0)


def diagnostics(prediction,target,metadata,masks,output,plots=False):
    pairs={}
    for pair in metadata['pairs']['task2']:
        if pair['target_record_id'] not in pairs or pair['input_type']=='ecg_machine_d6': pairs[pair['target_record_id']]=pair
    rows=[]
    for i,meta in enumerate(metadata['validation']):
        indices=pairs[meta['target_record_id']]['indices']
        p=np.concatenate([prediction[j] for j in indices],axis=1)[1:].astype(float)
        y=np.concatenate([target[j] for j in indices],axis=1)[1:].astype(float)
        ps,ys=slow(p),slow(y)
        pf,yf=p-ps,y-ys
        raw_r,slow_r,fast_r=correlations(p,y),correlations(ps,ys),correlations(pf,yf)
        seams=np.arange(5000,60000,5000)
        e=p-y
        for j,lead in enumerate(LEADS):
            peaks=np.abs(yf[j])>=np.quantile(np.abs(yf[j]),.99)
            rows.append({**meta,'lead':lead,'quality':'good' if masks[i,j] else 'flagged',
                'raw_r':raw_r[j],'slow_r':slow_r[j],'fast_r':fast_r[j],
                'raw_rmse_uV':np.sqrt(np.mean(e[j]**2)),'slow_rmse_uV':np.sqrt(np.mean((ps[j]-ys[j])**2)),
                'fast_rmse_uV':np.sqrt(np.mean((pf[j]-yf[j])**2)),
                'absolute_record_mean_error_uV':abs(e[j].mean()),
                'fast_top1pct_target_amplitude_mae_uV':np.abs(pf[j,peaks]-yf[j,peaks]).mean(),
                'seam_error_step_mae_uV':np.abs(e[j,seams]-e[j,seams-1]).mean()})
        if plots and i in (2,15,27):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            folder=output/'plots';folder.mkdir(exist_ok=True)
            for lead,j in (('V2',6),('V3',7),('V4',8)):
                fig,axes=plt.subplots(3,1,figsize=(13,8))
                t=np.arange(60000)/500
                axes[0].plot(t,ys[j],label='Target diagnostic slow',lw=.7)
                axes[0].plot(t,ps[j],label='FM predicted slow',lw=.7)
                sl=slice(18750,21250)
                axes[1].plot(t[sl],y[j,sl],label='Target raw',lw=.8)
                axes[1].plot(t[sl],p[j,sl],label='FM raw',lw=.8)
                axes[2].plot(t[sl],yf[j,sl],label='Target fast remainder',lw=.8)
                axes[2].plot(t[sl],pf[j,sl],label='FM fast remainder',lw=.8)
                for ax in axes:
                    ax.legend();ax.grid(alpha=.2);ax.set_ylabel('uV');ax.set_xlabel('Time (s)')
                axes[1].axvline(40,color='gray',ls=':');axes[2].axvline(40,color='gray',ls=':')
                fig.suptitle(meta['target_record_id']+' | '+lead+' | target used for evaluation only')
                fig.tight_layout();fig.savefig(folder/f'record_{i:02d}_{lead}.png',dpi=120);plt.close(fig)
    csv_write(output/'wave_record_lead.csv',rows)
    summaries=[]
    fields=('raw_r','slow_r','fast_r','raw_rmse_uV','slow_rmse_uV','fast_rmse_uV','absolute_record_mean_error_uV',
            'fast_top1pct_target_amplitude_mae_uV','seam_error_step_mae_uV')
    for cohort in ('all_unique','shared','exclusive'):
        for quality in ('all','good'):
            selected=[r for r in rows if (quality=='all' or r['quality']=='good') and
                (cohort=='all_unique' or r['shared_with_task1']==(cohort=='shared'))]
            if not selected: continue
            row={'cohort':cohort,'quality':quality}
            for field in fields:
                means=[np.mean([r[field] for r in selected if r['lead']==lead and np.isfinite(r[field])])
                       for lead in LEADS if any(r['lead']==lead and np.isfinite(r[field]) for r in selected)]
                row[field]=float(np.mean(means)) if means else float('nan')
            summaries.append(row)
    csv_write(output/'wave_summary.csv',summaries)
    return finite_json(summaries)
