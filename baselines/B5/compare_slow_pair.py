"""Compare completed paired runs by real ODE validation, without running a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnostic_signals import fresh_directory, read_rows, write_rows
from .objective import assert_pair_configs


def run_summary(directory: Path) -> dict:
    config = json.loads((directory/'resolved_config.json').read_text(encoding='utf-8'))
    provenance = json.loads((directory/'initialization.json').read_text(encoding='utf-8'))
    initial = json.loads((directory/'initial_validation.json').read_text(encoding='utf-8'))
    history = [json.loads(line) for line in (directory/'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    candidates = [initial]+[r for r in history if 'validation' in r]
    selection = config['validation']['selection_task']
    best = max(candidates,key=lambda r:r['validation'][selection]['r_missing11'])
    v = best['validation']
    metrics = {'run':str(directory.resolve()),'completed_epochs':history[-1]['epoch'] if history else 0,
               'best_epoch':best['epoch'],'initial_task1_r':initial['validation']['task1']['r_missing11'],
               'best_task1_r':v['task1']['r_missing11'],'best_task2_r':v['task2']['r_missing11'],
               'best_task2_chest_rmse_uV':v['task2']['task2_missing_lead_mean_rmse_uV'],
               'task1_gain_over_initializer':v['task1']['r_missing11']-initial['validation']['task1']['r_missing11']}
    return {'config':config,'provenance':provenance,'initial':initial,'metrics':metrics}


def compare_runs(control: Path, slow: Path) -> tuple[dict,dict]:
    a,b = run_summary(control),run_summary(slow)
    assert_pair_configs(a['config'],b['config'])
    pa,pb = a['provenance'],b['provenance']
    da = pa['initialization'].get('source_checkpoint_sha256')
    db = pb['initialization'].get('source_checkpoint_sha256')
    if not da or da!=db:
        raise ValueError('Paired initializers are not verified identical checkpoint contents')
    for key in ('scales_sha256','manifests','selection_task','architecture_hash'):
        if pa[key]!=pb[key]:
            raise ValueError('Paired provenance differs: '+key)
    for task in ('task1','task2'):
        for key in ('r_missing11','missing11_mean_rmse_uV'):
            if abs(a['initial']['validation'][task][key]-b['initial']['validation'][task][key])>1e-7:
                raise ValueError('Initial ODE validations differ; check sampling/data/software before attributing gains')
    return a,b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-run',type=Path,default=Path('results/B5/E3_slow_control'))
    parser.add_argument('--slow-run',type=Path,default=Path('results/B5/E3_slow_trend'))
    parser.add_argument('--diagnostics-dir',type=Path,help='Output of diagnose_evaluation with labels Control and Slow')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    a,b=compare_runs(args.control_run,args.slow_run)
    output=fresh_directory(args.output_dir)
    rows=[{'group':name,**data['metrics']} for name,data in (('Control',a),('Slow',b))]
    if args.diagnostics_dir:
        manifest=json.loads((args.diagnostics_dir/'manifest.json').read_text(encoding='utf-8'))
        if set(manifest['evaluations']) != {'Control','Slow'}:
            raise ValueError('Diagnostics must use exactly Control and Slow labels')
        for label,run in (('Control',args.control_run),('Slow',args.slow_run)):
            if Path(manifest['evaluations'][label]['directory']).resolve() != (run/'best_evaluation').resolve():
                raise ValueError('Diagnostics path differs from the paired best_evaluation output')
        diagnostics=read_rows(args.diagnostics_dir/'task_summary.csv')
        for row in rows:
            for task in ('task1','task2'):
                d=next(r for r in diagnostics if r['model']==row['group'] and r['task']==task)
                expected=row['best_'+task+'_r']
                if abs(float(d['raw_r_missing11'])-expected)>1e-7:
                    raise ValueError('Diagnostic predictions do not match selected best checkpoint validation')
                row[task+'_fast_r_diagnostic']=float(d['fast_r_missing11_diagnostic'])
                row[task+'_slow_r_diagnostic']=float(d['slow_r_missing11_diagnostic'])
    write_rows(output/'comparison.csv',rows)
    conclusion={'training_started':False,'model_inference_started':False,
                'initializer_sha256':a['provenance']['initialization']['source_checkpoint_sha256'],
                'slow_minus_control_task1_r':b['metrics']['best_task1_r']-a['metrics']['best_task1_r'],
                'slow_minus_control_task2_r':b['metrics']['best_task2_r']-a['metrics']['best_task2_r'],
                'slow_minus_control_task2_chest_rmse_uV':b['metrics']['best_task2_chest_rmse_uV']-a['metrics']['best_task2_chest_rmse_uV'],
                'runs_finished_20_epochs':all(row['completed_epochs']==20 for row in rows),
                'note':'Best selection includes the unchanged initializer at epoch0; compare raw r first, fast diagnostics and RMSE as safeguards.'}
    (output/'comparison.json').write_text(json.dumps(conclusion,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# B5 慢走势配对实验比较','','相同初始化、训练配置、数据、采样已核对。epoch0是初始化模型，没有优化器更新。','',
           '| 组 | 完成轮数 | 最佳轮数 | Task1 r | Task2 r | Task2胸导联RMSE μV |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['group']} | {r['completed_epochs']} | {r['best_epoch']} | {r['best_task1_r']:.6f} | {r['best_task2_r']:.6f} | {r['best_task2_chest_rmse_uV']:.2f} |")
    lines+=['',f"实验−对照 Task1 r：{conclusion['slow_minus_control_task1_r']:+.6f}",
            f"实验−对照 Task2 r：{conclusion['slow_minus_control_task2_r']:+.6f}",
            '辅助loss下降不是成功证据；需要原始r改善且快变化不明显退步。单seed实验不证明稳定收益。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(conclusion,indent=2))


if __name__=='__main__':
    main()
