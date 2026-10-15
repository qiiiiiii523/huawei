"""Compare control, frozen S50 and frozen S50-F on QC and full scores."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from . import GROUPS,SCORE_PROTOCOL,FULL_SCORE_PROTOCOL
from .common import read,write,csv_write,sha
from .prepare import load_shared
from .runtime import run_config,CANDIDATES
from .scoring import score_tasks
from .continuation import evaluation_directory


def compare(args):
    shared,info=load_shared(args.shared)
    output=Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use empty comparison output')
    original={'target_lead_qc':score_tasks(info['source_summaries']),
              'full_population':score_tasks(info['source_full_summaries'],qc=False)}
    rows=[{'group':'original_B5','checkpoint':'source_best','epoch':'source','score_view':view,**score}
          for view,score in original.items()]
    initials=[]
    budget=int(getattr(args,'total_epochs',20))
    folders={'control':Path(args.control),'slow':Path(args.slow)}
    if getattr(args,'slow_f',None): folders['slow_f']=Path(args.slow_f)
    for group,folder in folders.items():
        train=folder/'checkpoints'
        if read(train/'resolved_config.json')!=run_config(group,shared,info): raise ValueError('Unmatched three-arm run: '+group)
        completed=read(train/'completed.json')
        if not completed['completed'] or int(completed.get('total_epochs',completed['additional_epochs']))!=budget:
            raise ValueError('Finish all selected groups with the same requested total epoch budget first')
        initial=read(train/'initial_evaluation.json')
        initials.append({'target_lead_qc':initial['score'],'full_population':initial['full_score']})
        for name in (*CANDIDATES.values(),'last.pt'):
            evaluation=read(evaluation_directory(folder,budget,info['training']['epochs'])/Path(name).stem/'evaluation.json')
            fingerprint=evaluation['fingerprint']
            if (not evaluation['completed'] or fingerprint['group']!=group or
                    fingerprint['score_protocol']!=SCORE_PROTOCOL or fingerprint['shared_sha256']!=sha(shared/'shared.json')):
                raise ValueError('Final evaluation belongs to another group/cache/score protocol')
            if sha(train/name)!=evaluation['checkpoint_sha256']: raise ValueError('Checkpoint changed after evaluation')
            if name=='last.pt' and evaluation['epoch']!=budget: raise ValueError('Last evaluation has not reached requested budget')
            for view,score in (('target_lead_qc',evaluation['score']),('full_population',evaluation['full_score'])):
                expected=SCORE_PROTOCOL if view=='target_lead_qc' else FULL_SCORE_PROTOCOL
                if score['competition_score_protocol']!=expected: raise ValueError('Cannot mix QC and full scores')
                rows.append({'group':group,'checkpoint':Path(name).stem,'epoch':evaluation['epoch'],'score_view':view,**score})
    for initial in initials:
        for view,score in original.items():
            for key,value in score.items():
                if isinstance(value,(float,int)) and not np.isclose(initial[view][key],value,atol=2e-6,rtol=2e-6):
                    raise ValueError('Initial score differs from common B5 reference: '+view+'/'+key)
    for variant,reference in info['fixed_hybrid_references'].items():
        for view,score in (('target_lead_qc',reference['score']),('full_population',reference['full_score'])):
            rows.append({'group':variant+'_fixed_output_hybrid_reference','checkpoint':'slow_source',
                         'epoch':reference['epoch'],'score_view':view,**score})
    csv_write(output/'scores.csv',rows)
    write(output/'fixed_hybrid_references.json',info['fixed_hybrid_references'])
    fields=('task1_r_missing11','task2_r_missing11','ecg_machine_d6_task2_missing_lead_mean_rmse_uV',
            'body_scale_d6_task2_missing_lead_mean_rmse_uV','competition_total_score')
    lines=['# B5-S50慢条件接入结果','',f'同一原B5 EMA初始化，各累计新增{budget}轮，Heun16/K16，拼接120秒。',
           'QC v3为选模与主要比较口径；完整集合v2用同一预测另行算分，不据此重新挑模型。',
           '','|组别|权重|新增轮次|评分范围|Task1 r|Task2 r|机器RMSE/μV|体脂秤RMSE/μV|总分/60|',
           '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('|'+ '|'.join([str(row[k]) for k in ('group','checkpoint','epoch','score_view')]+[f'{row[k]:.6f}' for k in fields])+'|')
    best={group:next(r for r in rows if r['group']==group and r['checkpoint']=='best' and r['score_view']=='target_lead_qc') for group in folders}
    deltas={'S50_minus_control':best['slow']['competition_total_score']-best['control']['competition_total_score']}
    if 'slow_f' in best:
        deltas.update(S50F_minus_control=best['slow_f']['competition_total_score']-best['control']['competition_total_score'],
                      S50F_minus_S50=best['slow_f']['competition_total_score']-best['slow']['competition_total_score'])
    write(output/'deltas.json',{'score_protocol':SCORE_PROTOCOL,'total_epochs':budget,'QC_best_deltas':deltas})
    lines+=['']+[f'{key}：{value:+.6f} 分。' for key,value in deltas.items()]
    lines+=['','slow是原损失S50；slow_f使用已经接受F监督的冻结慢CNN。FM损失三组相同，没有在FM中增加F损失。',
        '检查各组当前预算评估目录中best/wave_summary.csv、wave_record_lead.csv和plots，关注独有患者、慢/快r、RMSE、波峰及接缝。',
        'best若仍为第0轮，说明该组继续训练未超过共同初始化。一次seed有收益后仍需复核。',
        '固定输出组合参考只用于判断是否值得接入，不是新的FM训练成绩。']
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'comparison':str(output),'QC_best_deltas':deltas},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('shared','control','slow','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--slow-f',help='Optional third arm; it must finish the same budget')
    p.add_argument('--total-epochs',type=int,default=20,help='Cumulative added epochs, e.g. 40 for the 20+20 continuation')
    compare(p.parse_args())


if __name__=='__main__': main()
