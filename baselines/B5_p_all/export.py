"""Export an evaluated checkpoint as a complete unchanged FM + updated P slow bundle."""
import argparse
import copy
from pathlib import Path
import torch
from baselines.B5.checkpoint import atomic_save
from baselines.B5_best.bundle import validate_bundle
from baselines.B5_best.common import write,sha,state_sha
from . import PROTOCOL
from .common import load_checkpoint


def export(study_path,checkpoint,output):
    output=Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError('Choose a new portable bundle path')
    cp,study,original,model=load_checkpoint(checkpoint,study_path,torch.device('cpu'))
    score=cp.get('validation_score')
    if score is None:raise ValueError('Export a scored best.pt or evaluated final last.pt')
    bundle=copy.deepcopy(original)
    bundle['slow']['model']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    bundle['slow_state_sha256']=state_sha(bundle['slow']['model'])
    bundle['reference_qc_r']={k:score[k] for k in ('task1_r_missing11','task2_r_missing11')}
    bundle['reference_score_protocol']=study['score_protocol']
    bundle['provenance'].update(experiment=PROTOCOL,experiment_group=cp['group'],source_updated_slow_step=cp['step'],
          slow_sha256=sha(checkpoint),study_sha256=sha(study_path),FM_frozen=True,stable_reference=False,
          P_best_source_bundle_sha256=study['bundle_sha256'],training_seed=study['training']['seed'],
          multiscale_weight=cp['multiscale']['weight'],enabled_bands=cp['multiscale']['enabled_bands'])
    validate_bundle(bundle)
    if state_sha(bundle['fm']['ema'])!=original['fm_state_sha256']:raise ValueError('FM export changed')
    atomic_save(output,bundle)
    write(output.with_suffix('.json'),{'bundle_sha256':sha(output),'checkpoint_sha256':sha(checkpoint),
          'group':cp['group'],'step':cp['step'],'score':score,'FM_unchanged':True,
          'sampling':study['sampling'],'beta':study['beta'],'target_required_at_inference':False,
          'cache_required_at_inference':False})
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('study','checkpoint','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();export(a.study,a.checkpoint,a.output)


if __name__=='__main__':main()
