"""Migrate an interrupted P to a new run, changing only validation frequency.

Deploy this script alone. Do not switch the live model package or reset AdamW.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def assert_stopped(source):
    """Linux server: detect old train/pipeline even if the output arg is relative."""
    if not Path('/proc').is_dir():
        return
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            parts = (entry/'cmdline').read_bytes().decode().split('\0')
            if not any(x in parts for x in ('baselines.B7.train','baselines.B7.pipeline')):
                continue
            position = parts.index('--output')
            value = Path(parts[position+1])
            if not value.is_absolute():
                value = (entry/'cwd').resolve()/value
            value = value.resolve()
            if value == source or value.parent == source:
                raise RuntimeError(f'Old B7 process {entry.name} is still active; stop its tmux controller first')
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue


def migrate(source, output, manifest, config_path, every):
    from baselines.B7.checkpoint import load_checkpoint, save_checkpoint, file_sha256, model_from_checkpoint
    from baselines.B7.data import read_manifest
    from baselines.B7.train import code_sha, validate_settings
    from baselines.B7.pipeline import acquire_lock
    source, output, manifest, config_path = [Path(p).resolve() for p in (source,output,manifest,config_path)]
    if every <= 0 or output.exists() or not source.is_dir():
        raise ValueError('Need positive interval, existing source and NEW output directory')
    if output == source or source in output.parents:
        raise ValueError('New output must not be inside the original run')
    assert_stopped(source)
    lock = acquire_lock(source/'.pipeline.lock')
    try:
        if any((source/name).exists() for name in ('A','Q','B')) or (source/'P/completed.json').exists():
            raise ValueError('This migration is only for unfinished P before A/Q/B have started')
        last = load_checkpoint(source/'P/last.pt')
        best = load_checkpoint(source/'P/best_r.pt')
        identity = last['identity']
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        if config != identity['config'] or json.loads((source/'P/config.json').read_text()) != config:
            raise ValueError('Current config differs from saved P; do not edit it manually before migration')
        if config['stage'] != 'public10' or last['stage'] != 'public10':
            raise ValueError('Only P/public10 may be rescheduled here')
        if identity['code_sha256'] != code_sha():
            raise ValueError('Model package changed; deploy only this script on the original code version')
        if identity['manifest_sha256'] != file_sha256(manifest):
            raise ValueError('Manifest changed')
        data = read_manifest(manifest)
        hashes = {r['record_id']:file_sha256(r['resolved_path']) for r in data['records']}
        if hashes != identity['record_sha256'] or list(last['scales_uV']) != list(data['scales_uV']):
            raise ValueError('Prepared data/scales changed')
        if best['identity'] != identity or best['best_r'] != last['best_r'] or best['update'] > last['update']:
            raise ValueError('Best and last lineage/score do not agree')
        for key in ('optimizer','rng','model','ema','initialization'):
            if key not in last:
                raise ValueError(f'Cannot resume: missing {key}')
        if not 0 <= last['update'] < config['training']['updates']:
            raise ValueError('Checkpoint must be unfinished and within budget')
        model_from_checkpoint(best)
        new_config = copy.deepcopy(config)
        new_config['training']['validate_every'] = every
        validate_settings(new_config)
        new_identity = copy.deepcopy(identity)
        new_identity['config'] = new_config
        output.mkdir(parents=True)
        target = output/'P'
        target.mkdir()
        # Originals remain byte-for-byte untouched; only identity/config changes
        # in NEW checkpoint copies. No tensor, optimizer, RNG or step changes.
        record = {'source':str(source),'manifest':str(manifest),'source_last_sha256':file_sha256(source/'P/last.pt'),
                  'source_best_sha256':file_sha256(source/'P/best_r.pt'), 'source_code_sha256':code_sha(),
                  'old_validate_every':config['training']['validate_every'],'new_validate_every':every,
                  'resume_update':last['update'],'best_r':last['best_r'],'complete':False}
        (output/'migration.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        (output/'original_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        for name, value in (('last.pt',last),('best_r.pt',best),('best.pt',best)):
            value = dict(value, identity=new_identity)
            value['validation_reschedule'] = {k:v for k,v in record.items() if k!='complete'}
            save_checkpoint(target/name,value)
        (target/'config.json').write_text(json.dumps(new_config,indent=2),encoding='utf-8')
        if (source/'P/history.jsonl').is_file():
            shutil.copy2(source/'P/history.jsonl',target/'history.jsonl')
        # Configuration changes only after the new checkpoint copies exist.
        temporary = config_path.with_suffix('.json.reschedule.tmp')
        temporary.write_text(json.dumps(new_config,indent=2)+'\n',encoding='utf-8')
        temporary.replace(config_path)
        record['complete']=True
        (output/'migration.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps({'status':'MIGRATED','resume_update':last['update'],'best_r':last['best_r'],
                          'validate_every':every,'output':str(output),'original_checkpoints_untouched':True}),flush=True)
        return record
    finally:
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/b7_public10.json'))
    parser.add_argument('--validate-every',type=int,default=2000)
    args=parser.parse_args()
    os.chdir(ROOT)
    migrate(args.source,args.output,args.manifest,args.config,args.validate_every)


if __name__=='__main__':
    main()
