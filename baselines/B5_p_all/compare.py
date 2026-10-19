"""Summarize paired best and last, always including the P-best initialization."""
import argparse
from pathlib import Path
from baselines.B5_best.common import read,write,csv_write
from . import PROTOCOL,GROUPS


def compare(run):
    run=Path(run).resolve();s=read(run/'shared/study.json')
    if s.get('protocol')!=PROTOCOL:raise ValueError('Wrong paired experiment')
    keys=('task1_r_missing11','task2_r_missing11','main_correlation')
    rows=[{'group':'P-initial','checkpoint':'source_best','new_updates':0,**{k:s['reference_score'][k] for k in keys}}]
    for g in GROUPS:
        if read(run/g/'completed.json').get('completed') is not True:raise ValueError('Group did not finish: '+g)
        for name in ('best','last'):
            pointer=run/g/f'latest_{name}_evaluation.txt'
            e=read(Path(pointer.read_text().strip())/'evaluation.json')
            if e.get('group')!=g or e.get('completed') is not True or e['score']['score_protocol']!=s['score_protocol']:
                raise ValueError('Missing matching independent evaluation')
            rows.append({'group':g,'checkpoint':name,'new_updates':e['step'],**{k:e['score'][k] for k in keys}})
    for r in rows:r['delta_vs_P_initial']=r['main_correlation']-rows[0]['main_correlation']
    best={r['group']:r for r in rows if r['checkpoint']=='best'}
    csv_write(run/'summary.csv',rows)
    write(run/'summary.json',{'completed':True,'score_protocol':s['score_protocol'],'rows':rows,
                            'best_ALL_minus_control':best['all']['main_correlation']-best['control']['main_correlation']})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args()
    for r in compare(a.run):print(r)


if __name__=='__main__':main()
