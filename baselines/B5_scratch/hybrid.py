"""Evaluate each new FM fast remainder + the same frozen S50-F predictor output."""
from pathlib import Path
import numpy as np

from ecg12gen.evaluate import evaluate_record_predictions, write_report
from baselines.B5_best.combine import combine, upsample_native
from baselines.B5_best.scoring import diagnostic_summary
from .common import read,csv_read,csv_write
from .scoring import score_tasks,write_score


def evaluate_hybrid(output,shared,info):
    output,shared=Path(output),Path(shared)
    folder=output/'hybrid_slow_f'
    if (folder/'correlation_score.json').exists():
        return read(folder/'correlation_score.json')
    metadata=read(shared/'records.json')
    lookup={r['target_record_id']:i for i,r in enumerate(metadata['validation'])}
    native=np.load(shared/'validation_slow_f.npy',mmap_mode='r',allow_pickle=False)
    scale=np.asarray(info['scales']['d12'],float)
    summaries={}
    for task in ('task1','task2'):
        source=output/task
        pred=np.load(source/'prediction_uV.npy',mmap_mode='r',allow_pickle=False)
        target=np.load(Path(info['evaluation_dir'])/task/'target_uV.npy',mmap_mode='r',allow_pickle=False)
        quality=np.load(shared/(task+'_quality.npy'),allow_pickle=False)
        rows=csv_read(source/'window_metadata.csv')
        destination=folder/task;destination.mkdir(parents=True,exist_ok=True)
        result=np.lib.format.open_memmap(destination/'prediction_uV.npy',mode='w+',dtype=np.float32,shape=pred.shape)
        groups={}
        for i,row in enumerate(rows):groups.setdefault(row['pair_id'],[]).append(i)
        for indices in groups.values():
            indices.sort(key=lambda i:int(rows[i]['start_sample_500hz']))
            if [int(rows[i]['start_sample_500hz']) for i in indices]!=list(range(0,60000,5000)):
                raise ValueError('Hybrid needs complete real 120s records')
            key=rows[indices[0]]['target_record_id']
            fm=np.concatenate([pred[i] for i in indices],axis=-1)
            learned=upsample_native(native[lookup[key]].astype(float)*scale[1:,None])
            full=combine(fm,learned)
            for n,index in enumerate(indices):result[index]=full[:,n*5000:(n+1)*5000]
        result.flush()
        summary,details=evaluate_record_predictions(result,target,task,rows,target_quality_mask=quality)
        summaries[task]=diagnostic_summary(summary)
        write_report(destination,summaries[task],details,title='New FM fast remainder + frozen S50-F, QC r only')
        csv_write(destination/'window_metadata.csv',rows)
        del result,pred,target
    score=score_tasks(summaries)
    write_score(folder,score)
    return score
