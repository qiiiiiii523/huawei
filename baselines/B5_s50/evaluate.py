"""Final Heun16/K16 evaluation of a self-contained paired checkpoint."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from baselines.B5.runtime import device_from_name
from .common import read,write,sha
from .checkpoint import load,model_from
from .prepare import load_shared
from .runtime import run_config,datasets,validate


def fingerprint(checkpoint):
    return {key:checkpoint[key] for key in ('group','model_ema_sha256','slow_state_sha256','sampling')} | {
        'shared_sha256':checkpoint['config']['shared_sha256']}


def evaluate(args):
    shared,info=load_shared(args.shared)
    checkpoint=load(args.checkpoint)
    if checkpoint['config']!=run_config(checkpoint['group'],shared,info):
        raise ValueError('Checkpoint and shared experiment settings differ')
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use empty final evaluation output')
    identity=fingerprint(checkpoint)
    reused=None
    if args.reuse_root:
        for marker in sorted(Path(args.reuse_root).glob('*/evaluation.json')):
            previous=read(marker)
            if previous.get('completed') and previous.get('fingerprint')==identity and (not args.plots or previous.get('plots')):
                reused=marker.parent
                break
    if reused:
        output.mkdir(parents=True,exist_ok=True)
        def link_or_copy(source,destination):
            try: os.link(source,destination)
            except OSError: shutil.copy2(source,destination)
        for source in reused.iterdir():
            if source.name=='evaluation.json': continue
            if source.is_dir(): shutil.copytree(source,output/source.name,copy_function=link_or_copy)
            else: link_or_copy(source,output/source.name)
        result=read(output/'metrics.json')
    else:
        device=device_from_name(args.device)
        _,validation,_=datasets(shared,info)
        result=validate(model_from(checkpoint,device),validation,shared,info,device,output,args.plots)
    write(output/'evaluation.json',{'completed':True,'fingerprint':identity,'epoch':checkpoint['epoch'],
        'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_sha256':sha(args.checkpoint),
        'plots':bool(args.plots) or bool(reused and read(reused/'evaluation.json').get('plots')),
        'reused_identical_ema_from':str(reused) if reused else None,'target_used_at_inference':False,
        'score':result['score']})
    print(result['score'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--reuse-root',help='Reuse only an identical group/EMA/cache/sampler already evaluated here')
    parser.add_argument('--plots',action='store_true')
    evaluate(parser.parse_args())


if __name__=='__main__': main()
