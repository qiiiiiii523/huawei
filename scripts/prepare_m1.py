"""Read-only M1 data/scale audit; optional train-only fitting, never training."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/common.yaml'))
    p.add_argument('--task-id',choices=('task1','task2'),default='task1')
    p.add_argument('--stage',choices=('P0_anchor_only','P1_joint_anchor'),default='P0_anchor_only')
    p.add_argument('--context-source-type',choices=('watch_ecg','body_scale_d6','ecg_machine_d6'))
    p.add_argument('--scales',help='Optional existing B5/main-v3 NPZ scales')
    p.add_argument('--save-scales',help='Optional NEW NPZ destination')
    p.add_argument('--report',required=True)
    a=p.parse_args()
    if a.stage=='P0_anchor_only' and a.context_source_type: p.error('P0 cannot receive context')
    if a.stage=='P1_joint_anchor' and not a.context_source_type: p.error('P1 needs context source')
    if a.save_scales and Path(a.save_scales).exists(): p.error('Refusing to overwrite frozen scales')
    from ecg12gen.m1_data import fit_m1_preprocessor, build_m1_datasets
    report={'training_started':False,'errors':[]}
    try:
        pre=fit_m1_preprocessor(a.config,a.task_id,a.stage,a.context_source_type,scales_path=a.scales)
        tasks=('task1','task2') if a.stage=='P0_anchor_only' else (a.task_id,)
        validation={}
        for task in tasks:
            train,val=build_m1_datasets(a.config,task,a.stage,a.context_source_type,pre)
            train[0]; val[0]
            train_subjects={x.meta['subject_id'] for x in train.samples}
            val_subjects={x.meta['subject_id'] for x in val.samples}
            if train_subjects & val_subjects: raise ValueError('M1 train/validation subject overlap')
            validation[task]={'windows':len(val),'pairs':len({x.meta['pair_id'] for x in val.samples}),
                              'subjects':len(val_subjects)}
        report.update(train_windows=len(train),train_subjects=len(train_subjects),validation=validation,
                      scales={k:v.tolist() for k,v in pre.scale_uV_by_source.items()})
        if a.save_scales:
            dest=Path(a.save_scales); dest.parent.mkdir(parents=True,exist_ok=True); pre.save(dest)
    except (OSError,ValueError) as error: report['errors'].append(str(error))
    report['ready']=not report['errors']
    dest=Path(a.report); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['ready']: raise SystemExit(2)

if __name__=='__main__': main()
