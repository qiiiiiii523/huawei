"""Compare matched continued-FM control and S50-conditioned FM; no training."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import read,write,csv_write,sha
from .prepare import load_shared
from .runtime import run_config,CANDIDATES
from .scoring import score_tasks


def compare(args):
    shared,info=load_shared(args.shared)
    output=Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use empty comparison output')
    original=score_tasks(info['source_summaries'])
    rows=[{'group':'original_B5','checkpoint':'source_best','epoch':'source',**original}]
    initials=[]
    for group,folder in (('control',Path(args.control)),('slow',Path(args.slow))):
        train=folder/'checkpoints'
        if read(train/'resolved_config.json')!=run_config(group,shared,info): raise ValueError('Unmatched paired run')
        completed=read(train/'completed.json')
        if not completed['completed'] or completed['additional_epochs']!=20: raise ValueError('Finish the full paired budget first')
        initial=read(train/'initial_evaluation.json')['score']
        initials.append(initial)
        for name in (*CANDIDATES.values(),'last.pt'):
            marker=folder/'evaluations'/Path(name).stem/'evaluation.json'
            evaluation=read(marker)
            if not evaluation['completed'] or evaluation['fingerprint']['group']!=group or evaluation['fingerprint']['shared_sha256']!=sha(shared/'shared.json'):
                raise ValueError('Final evaluation belongs to a different run')
            if sha(train/name)!=evaluation['checkpoint_sha256']: raise ValueError('Checkpoint changed after evaluation')
            rows.append({'group':group,'checkpoint':Path(name).stem,'epoch':evaluation['epoch'],**evaluation['score']})
    for key in original:
        if isinstance(original[key],(float,int)) and abs(initials[0][key]-initials[1][key])>1e-6:
            raise ValueError('A/B initialization scores differ')
    reference=info.get('fixed_hybrid_reference')
    if reference:
        write(output/'fixed_hybrid_reference.json',reference)
        for score in reference['report']['scores']:
            if score.get('method')=='learned_slow_hybrid' and score.get('competition_score_protocol')==info['score_protocol']:
                rows.append({'group':'fixed_output_hybrid_reference','checkpoint':'S50_best_score','epoch':reference['report']['epoch'],
                             **{k:v for k,v in score.items() if k!='method'}})
    csv_write(output/'scores.csv',rows)
    fields=('task1_r_missing11','task2_r_missing11','ecg_machine_d6_task2_missing_lead_mean_rmse_uV',
            'body_scale_d6_task2_missing_lead_mean_rmse_uV','competition_total_score')
    lines=['# B5-S50 配对结果','','同一原 B5 EMA 初始化，新增20轮；完整集合v2、Heun16/K16、120秒评分。',
           '','|组别|权重|轮次|Task1 r|Task2 r|机器RMSE/μV|体脂秤RMSE/μV|总分/60|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('|'+ '|'.join([str(row[k]) for k in ('group','checkpoint','epoch')]+[f'{row[k]:.6f}' for k in fields])+'|')
    a=next(r for r in rows if r['group']=='control' and r['checkpoint']=='best')
    b=next(r for r in rows if r['group']=='slow' and r['checkpoint']=='best')
    delta=b['competition_total_score']-a['competition_total_score']
    lines+=['',f'慢条件组 best 相对匹配对照 best：{delta:+.6f} 分。',
        '','同时检查 evaluations/best/wave_summary.csv、wave_record_lead.csv 和 plots：整体/独有患者、正常导联慢r、快r、RMSE、波峰误差和窗口接缝。',
        '若 best 仍是第0轮，说明继续训练没有超过共同初始化，不能把它当作接入已取得训练收益。',
        '这是同一划分上的一次配对实验。即使有收益，也需换训练seed复核；无收益不能直接证明慢曲线无信息。',
        '固定B5快分量＋S50慢分量的历史组合分是输出替换参考，不是本次FM接入结果。']
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'comparison':str(output),'slow_minus_control_score':delta},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('shared','control','slow','output'): p.add_argument('--'+key,required=True)
    compare(p.parse_args())


if __name__=='__main__': main()
