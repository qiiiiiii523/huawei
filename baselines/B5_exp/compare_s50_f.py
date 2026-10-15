"""Compare matching S50 original-loss and S50-F; reuse the completed S50 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import sha, csv_read, csv_write, json_write
from .context_audit import read, CANDIDATES
from .check_loss import implementation_hashes
from .s50_f_audit import (PAIR_PROTOCOL, canonical, verify_control, verify_unchanged_code, assert_initial_equal)


def compare(control, experiment, output):
    paths = [Path(control).resolve(),Path(experiment).resolve()]
    configs = [read(path/'resolved_config.json') for path in paths]
    if [c['group'] for c in configs] != ['C','F'] or canonical(configs[0]) != canonical(configs[1]):
        raise ValueError('Expected matched S50 original-loss/F runs differing only in F activation')
    config = configs[1]
    if config.get('s50_f_protocol') != PAIR_PROTOCOL or config['implementation_sha256'] != implementation_hashes():
        raise ValueError('Use the matching S50-F implementation and preflight')
    verify_unchanged_code(config['implementation_sha256'])
    snapshot = paths[1]/'pair_preflight.json'
    proof = read(snapshot)['s50_f_pair']
    if sha(snapshot) != config['calibration_sha256'] or not proof.get('ready'):
        raise ValueError('Invalid S50-F preflight snapshot')
    if Path(proof['control_reference']['path']).resolve() != paths[0] or proof['control_reference'] != config['s50_f_control_reference']:
        raise ValueError('Experiment references another S50 control')
    verify_control(proof['control_reference'])
    assert_initial_equal(read(paths[1]/'initial_evaluation.json'),read(paths[0]/'initial_evaluation.json'))
    score_rows,wave_rows,history_rows,audits = [],[],[],[]
    for path,cfg,tag in zip(paths,configs,('S50','S50-F')):
        completed = read(path/'completed.json')
        if not completed.get('completed') or completed['additional_epochs'] != cfg['epochs']:
            raise ValueError('Training is incomplete: '+tag)
        initial = read(path/'initial_evaluation.json')
        score_rows.append({'experiment':tag,'candidate':'initializer','epoch':0,**initial['hybrid_score']})
        for candidate in CANDIDATES:
            folder = path/('evaluation_'+candidate)
            evaluation = read(folder/'evaluation.json')
            if not evaluation.get('completed') or evaluation['context_seconds'] != 50 or evaluation['group'] != cfg['group']:
                raise ValueError('Wrong group/context candidate evaluation')
            if evaluation['probe_checkpoint_sha256'] != sha(path/(candidate+'.pt')) or evaluation['prepared_sha256'] != cfg['prepared_sha256']:
                raise ValueError('Stale S50/F candidate evaluation')
            if evaluation['B5_checkpoint_sha256'] != proof['control_reference']['B5_checkpoint_sha256']:
                raise ValueError('Different fixed B5 predictions')
            if candidate == 'last' and evaluation['epoch'] != cfg['epochs']:
                raise ValueError('Last did not finish the matching budget')
            score = next(row for row in evaluation['scores'] if row['method']=='learned_slow_hybrid')
            score_rows.append({'experiment':tag,'candidate':candidate,'epoch':evaluation['epoch'],**{k:v for k,v in score.items() if k!='method'}})
            wave_rows.extend({'experiment':tag,'candidate':candidate,**row} for row in csv_read(folder/'wave_summary.csv'))
            audits.append({'experiment':tag,'candidate':candidate,'evaluation_sha256':sha(folder/'evaluation.json')})
        history = [json.loads(line) for line in (path/'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if [row['epoch'] for row in history] != list(range(cfg['epochs']+1)):
            raise ValueError('Missing or duplicated epoch history')
        for row in history:
            v=row.get('validation',{})
            history_rows.append({'experiment':tag,'epoch':row['epoch'],'loss':row.get('train',{}).get('loss'),
                'slow_r':v.get('slow_r_unique_quality_good_50Hz'),'total_score':v.get('hybrid_score',{}).get('competition_total_score')})
    for row in score_rows:
        baseline = next(r for r in score_rows if r['experiment']=='S50' and r['candidate']==row['candidate'])
        row['delta_total_vs_S50_same_candidate']=row['competition_total_score']-baseline['competition_total_score']
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use empty comparison output')
    output.mkdir(parents=True,exist_ok=True)
    csv_write(output/'raw_scores.csv',score_rows)
    csv_write(output/'wave_diagnostics.csv',wave_rows)
    csv_write(output/'history.csv',history_rows)
    json_write(output/'audit.json',{'completed':True,'protocol':PAIR_PROTOCOL,'same_initial_weights':config['initial_weights_sha256'],
        'control_reference':proof['control_reference'],'evaluations':audits,'only_active_change':'fixed_F_residual_loss',
        'epoch_zero_scores_equal':True,'target_used_at_inference':False,'FM_model_modified':False})
    lines=['# S50与S50-F配对慢CNN实验','','同一原S30-best50初始化、50秒I输入、原CNN与新增20轮，只改变F辅助损失。复用已有S50原损失对照。','',
        '| 实验 | 候选 | 新增轮次 | Task1 r | Task2 r | 机器RMSE | 体脂秤RMSE | 总分/60 | 相对S50 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in score_rows:
        lines.append(f"| {row['experiment']} | {row['candidate']} | {row['epoch']} | {row['task1_r_missing11']:.6f} | "
            f"{row['task2_r_missing11']:.6f} | {row['ecg_machine_d6_task2_missing_lead_mean_rmse_uV']:.2f} | "
            f"{row['body_scale_d6_task2_missing_lead_mean_rmse_uV']:.2f} | {row['competition_total_score']:.3f} | {row['delta_total_vs_S50_same_candidate']:+.3f} |")
    lines+=['','先比较同等20轮last，再比较同一选模规则的best。看正常导联/独有目标慢r和RMSE，也检查快细节、V2/V3/V4波峰和边界。',
        '成绩仍是固定B5快分量＋慢CNN预测的输出组合；尚未把慢CNN接入FM。一次探索不能保证小收益稳定。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(output/'report.md'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',required=True)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    compare(args.control,args.experiment,args.output)


if __name__=='__main__':
    main()
