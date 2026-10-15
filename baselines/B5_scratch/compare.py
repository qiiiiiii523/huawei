"""Compare completed early-public runs at equal budgets, by mean r only."""
import argparse
from pathlib import Path

from . import SCORE_PROTOCOL,FULL_SCORE_PROTOCOL
from .prepare import load_shared
from .runtime import run_config
from .common import read,write,csv_write,sha
from .scoring import score_tasks
from .continuation import evaluation_directory


def compare(args):
    shared,info=load_shared(args.shared)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use empty comparison directory')
    budget=args.total_epochs or info['training']['epochs']
    rows=[]
    for label,qc,full in [('public_initializer',info['source_summaries'],info['source_full_summaries']),
                          ('old_Huawei_B5_reference',info['huawei_reference_summaries'],info['huawei_reference_full_summaries'])]:
        for view,score in [('target_lead_qc',score_tasks(qc)),('full_population',score_tasks(full,qc=False))]:
            rows.append({'group':label,'checkpoint':'reference','epoch':'reference','view':view,**score})
    folders={'control':Path(args.control),'slow':Path(args.slow)}
    if args.slow_f:folders['slow_f']=Path(args.slow_f)
    for group,folder in folders.items():
        train=folder/'checkpoints'
        if read(train/'resolved_config.json')!=run_config(group,shared,info):raise ValueError('Run/cache settings differ')
        done=read(train/'completed.json')
        if not done.get('completed') or done.get('total_epochs')!=budget:raise ValueError('Complete the same total budget in each selected group')
        for name in ('best_r','last'):
            e=read(evaluation_directory(folder,budget,info['training']['epochs'])/name/'evaluation.json')
            fp=e['fingerprint']
            if (not e.get('completed') or fp['group']!=group or fp['shared_sha256']!=sha(shared/'shared.json') or
                    fp['score_protocol']!=SCORE_PROTOCOL or sha(train/(name+'.pt'))!=e['checkpoint_sha256']):
                raise ValueError('Missing/stale/mixed-group final evaluation')
            if name=='last' and e['epoch']!=budget:raise ValueError('Last checkpoint is short of budget')
            for view,score,protocol in [('target_lead_qc',e['score'],SCORE_PROTOCOL),('full_population',e['full_score'],FULL_SCORE_PROTOCOL),
                                        ('hybrid_slow_f',e['hybrid_slow_f_score'],SCORE_PROTOCOL)]:
                if score['score_protocol']!=protocol:raise ValueError('Mixed r protocols')
                rows.append({'group':group,'checkpoint':name,'epoch':e['epoch'],'view':view,**score})
    for variant,ref in info['fixed_hybrid_references'].items():
        rows.append({'group':variant+'_old_fixed_output_reference','checkpoint':'reference','epoch':ref['epoch'],
                     'view':'target_lead_qc',**ref['score']})
    csv_write(out/'scores.csv',rows)
    best={g:next(r for r in rows if r['group']==g and r['checkpoint']=='best_r' and r['view']=='target_lead_qc') for g in folders}
    delta={g+'_minus_control':best[g]['main_correlation']-best['control']['main_correlation'] for g in folders if g!='control'}
    write(out/'deltas.json',{'score_protocol':SCORE_PROTOCOL,'total_epochs':budget,'mean_r_deltas':delta})
    lines=['# B5-scratch：公开初始化，华为第1轮加入慢条件','','按平均r比较；RMSE不算分。',
           'best_r按FM完整输出选；hybrid_slow_f是同一候选的输出组合，不声称遍历了所有组合最优epoch。','',
           '|组别|候选|轮次|输出|Task1 r|Task2 r|平均r|','|---|---|---:|---|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['group']}|{r['checkpoint']}|{r['epoch']}|{r['view']}|{r['task1_r_missing11']:.6f}|{r['task2_r_missing11']:.6f}|{r['main_correlation']:.6f}|")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(delta,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('shared','control','slow','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--slow-f');p.add_argument('--total-epochs',type=int)
    compare(p.parse_args())


if __name__=='__main__':main()
