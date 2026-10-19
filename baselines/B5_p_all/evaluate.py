"""Unchanged P full120s cached-FM scorer, with explicit paired-study metadata."""
import argparse
from pathlib import Path
from baselines.B5.runtime import device_from_name
from baselines.B5_best.common import read, write, sha
from baselines.B5_public_slow.evaluate import evaluate as original_evaluate
from .common import load_study, load_checkpoint


def evaluate(model, study, bundle, device, output, group, step, initial=False, save_arrays=False):
    score=original_evaluate(model,study,bundle,device,output,save_arrays)
    matched=None
    if initial:
        matched=all(abs(score[k]-study['reference_score'][k])<=bundle['reference_tolerance']
                    for k in ('task1_r_missing11','task2_r_missing11','main_correlation'))
    path=Path(output)/'evaluation.json'
    report=read(path)
    report.update(group=group,step=step,reference_matched=matched,
                  source_P_step=study['source_step'],FM_optimizer_updates=0,
                  multiscale_weight=study['multiscale']['weight'] if group=='all' else 0.,
                  selection_metric='main_correlation',FM_prediction_handling='fixed_historical_control100_cache')
    write(path,report)
    if initial and not matched:
        raise ValueError('P-best initial score did not reproduce; inspect reports before training')
    print({'group':group,'step':step,**score},flush=True)
    return score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--save-arrays',action='store_true')
    a=p.parse_args();device=device_from_name(a.device)
    cp,study,bundle,model=load_checkpoint(a.checkpoint,a.study,device)
    load_study(a.study,deep=True)
    evaluate(model,study,bundle,device,a.output,cp['group'],cp['step'],save_arrays=a.save_arrays)
    path=Path(a.output)/'evaluation.json';report=read(path)
    report.update(checkpoint_sha256=sha(a.checkpoint),study_sha256=sha(a.study))
    write(path,report)


if __name__=='__main__':main()
