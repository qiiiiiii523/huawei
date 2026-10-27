"""C5 = C0 Huawei10 adaptation + absolute record position; reuse C0 public P."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import copy
import json
import torch

from .checkpoint import file_sha256, file_hashes, load_checkpoint, model_from_checkpoint
from .config import read_json, SamplingConfig
from .data import read_manifest
from .evaluate import evaluate
from .position import validate_position_views
from .round1 import lock_run, write
from .train import train, code_sha

ROOT = Path(__file__).resolve().parents[2]


def check_control(control, manifest_path):
    specification = read_json(control/'specification.json')
    if specification['variant'] != 'C0':
        raise ValueError('C5 must compare against C0, not a changed-loss variant')
    cfg = read_json(ROOT/'configs/round1/C5/A.json')
    comparable = copy.deepcopy(cfg)
    if comparable['model'].pop('record_position') is not True or comparable != specification['configs']['A']:
        raise ValueError('C5 must preserve the actual C0 A recipe apart from position')
    if file_sha256(manifest_path) != specification['manifest_sha256'][1]:
        raise ValueError('C5 Huawei10 index differs from C0')
    manifest = read_manifest(manifest_path)
    validate_position_views(manifest, cfg['stage'])
    if file_sha256(manifest['view_source_manifest']) != manifest['view_source_sha256']:
        raise ValueError('Original full-record Huawei index changed')
    hashes = file_hashes(r['resolved_path'] for r in manifest['records'])
    if any(specification['raw_sha256'].get(p) != h for p,h in hashes.items()):
        raise ValueError('C5 cache differs from the actual C0 cache')
    checkpoint = control/'P/best_r.pt'
    p = load_checkpoint(checkpoint)
    done = read_json(control/'P/completed.json')
    if (not done['completed'] or p['stage'] != 'public10' or p['update'] != done['best_update']
            or p['best_r'] != done['best_r'] or p['identity']['config'] != specification['configs']['P']
            or p['scales_uV'] != manifest['scales_uV']):
        raise ValueError('C5 requires the verified C0 public best EMA and frozen scales')
    return cfg, manifest, checkpoint, hashes


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with lock_run(output/'.c5.lock'):
        cfg, manifest, checkpoint, hashes = check_control(Path(args.control_root).resolve(), args.manifest)
        full_path = Path(args.full_manifest).resolve()
        full = read_manifest(full_path)
        if full['scales_uV'] != manifest['scales_uV']:
            raise ValueError('Full120 evaluation scales differ from C0/C5')
        original = read_manifest(manifest['view_source_manifest'])
        expected = {r['record_id']:r for r in original['records'] if r['split']=='validation'}
        if {r['record_id'] for r in full['records']} != set(expected):
            raise ValueError('Full120 evaluation must contain exactly the original validation records')
        for row in full['records']:
            parent = expected[row['record_id']]
            if (row['split'] != 'validation' or row['subject_id'] != parent['subject_id']
                    or row['group'] != parent['group'] or row['target_record_id'] != parent['target_record_id']
                    or row['resolved_path'] != parent['resolved_path']
                    or row['input_reliable'] != parent['input_reliable']
                    or 'window_start' in row):
                raise ValueError('Full120 manifest changed original validation/QC/identity')
        identity = dict(variant='C5',config=cfg,code_sha256=code_sha(),
                        control_root=str(Path(args.control_root).resolve()),
                        checkpoint_sha256=file_sha256(checkpoint),manifest_sha256=file_sha256(args.manifest),
                        full_manifest_sha256=file_sha256(full_path),record_hashes=hashes)
        spec = output/'specification.json'
        if spec.exists() and read_json(spec)!=identity:
            raise ValueError('C5 code/config/data changed; use a new output')
        write(spec, identity)
        if args.check_only:
            print('[C5 CHECKED] C0 recipe/data/P match; no training',flush=True)
            return
        print('[C5] C0 P -> Huawei10 + absolute position; max4000; no P/Q/B training',flush=True)
        a = output/'A'
        if not (a/'completed.json').exists():
            train(Namespace(config=str(ROOT/'configs/round1/C5/A.json'),manifest=args.manifest,
                            output=str(a),device=args.device,resume=(a/'last.pt').exists(),
                            init_checkpoint=None if (a/'last.pt').exists() else str(checkpoint)))
        done=read_json(a/'completed.json')
        if not done['completed']:
            raise ValueError('C5 adaptation did not complete')
        # Same precision policy and complete120 scorer as C0-C4 full-record audit.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        reports={}
        previous=None
        for label,filename in [('best_segment_selected','best_r.pt'),('final','last.pt')]:
            target=output/f'full120_{label}.json'
            ck=load_checkpoint(a/filename)
            if ck['identity']['code_sha256']!=code_sha() or ck['identity']['config']!=cfg:
                raise ValueError('C5 checkpoint changed')
            if not target.exists():
                if previous is not None and previous[0]['update']==ck['update'] and all(
                        torch.equal(v,previous[0]['ema'][k]) for k,v in ck['ema'].items()):
                    metrics,rows=previous[1],previous[2]
                else:
                    model=model_from_checkpoint(ck,args.device)
                    metrics,rows=evaluate(model,full,SamplingConfig(**ck['sampling']),'huawei40')
                    del model
                write(target,dict(variant='C5',update=ck['update'],selection=label,
                                  checkpoint_sha256=file_sha256(a/filename),scored_seconds=120,
                                  metrics=metrics,records=rows))
                previous=(ck,metrics,rows)
            reports[label]=read_json(target)['metrics']
        write(output/'completed.json',dict(completed=True,variant='C5',A=done,full120=reports,
              control='C0',note='Segment and full120 scores are distinct; no full120-based checkpoint selection'))


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--control-root',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--full-manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--check-only',action='store_true')
    run(p.parse_args())


if __name__=='__main__':
    main()
