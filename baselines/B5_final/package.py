"""Stage pinned weights and/or build a Windows onedir EXE without training/downloads."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from baselines.B5_best.common import read,sha,write
from .profile import resolve_assets,BASE_SHA,BODY_SHA


def _build_exe(root,destination,packages,config_sources):
    """Build a Windows onedir executable into an already staged release folder."""
    with tempfile.TemporaryDirectory(prefix='b5_windows_build_') as temp:
        temp=Path(temp)
        command=[sys.executable,'-m','PyInstaller','--clean','--onedir','--console','--name','B5Submit',
            '--distpath',str(temp/'dist'),'--workpath',str(temp/'work'),'--specpath',str(temp),
            '--paths',str(root)]
        for package in packages:
            command+=['--collect-submodules',package]
        for source,destination_name in config_sources:
            # PyInstaller 6 parses --add-data as a single option value.
            # On Windows the source may itself contain a drive colon, so using
            # the explicit "--add-data=..." form avoids it being treated as a
            # positional argument.
            command+=[f'--add-data={source}{os.pathsep}{destination_name}']
        # The entry point is the final positional argument, not bundled data.
        command+=[str(root/'competition_main.py')]
        subprocess.run(command,cwd=root,check=True)
        shutil.copytree(temp/'dist/B5Submit',destination,dirs_exist_ok=True)


def _finish(destination,manifest,*,archive):
    write(destination/'package_manifest.json',manifest)
    (destination/'RUN.txt').write_text(manifest.pop('run_instructions'),encoding='utf-8')
    archive_path=(shutil.make_archive(str(destination)+'_package','zip',destination.parent,destination.name)
                  if archive else None)
    return archive_path


def stage(destination,base_checkpoint,body_checkpoint,*,build_exe=False,archive=True):
    root=Path(__file__).resolve().parents[2];destination=Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('Use a new empty release directory')
    if archive and Path(str(destination)+'_package.zip').exists(): raise FileExistsError('Release ZIP already exists')
    if root==destination or destination in root.parents: raise ValueError('Cannot package into a source ancestor')
    if any(root/folder in destination.parents for folder in ('baselines','ecg12gen','competition_interface')):
        raise ValueError('Cannot create release inside a source package')
    assets=resolve_assets(root,checkpoint=base_checkpoint,body_checkpoint=body_checkpoint)
    if build_exe:
        if os.name!='nt': raise RuntimeError('Windows EXE must be built on Windows, not Linux')
        for module in ('PyInstaller','torch','scipy','yaml','numpy'):
            if importlib.util.find_spec(module) is None: raise RuntimeError('Install build dependency in the active Windows environment: '+module)
    destination.mkdir(parents=True,exist_ok=True)
    models=destination/'models';models.mkdir()
    for source,name,digest in ((assets.base,'base.pt',BASE_SHA),(assets.body,'body.pt',BODY_SHA)):
        shutil.copyfile(source,models/name)
        if sha(models/name)!=digest: raise ValueError('Copied model checksum mismatch')
    shutil.copyfile(root/'configs/b5_submission.json',destination/'submission.json')
    if build_exe:
        _build_exe(root,destination,
                   ('baselines.B5','baselines.B5_best','baselines.B5_best_context','baselines.B5_final','ecg12gen','competition_interface'),
                   tuple((root/'configs'/name,'configs') for name in ('b5_final.json','b5_best_context.json','b5_submission.json')))
    else:
        # A Python staging package can be tested on the Linux training server.
        # These are code files only; do not copy results, caches, metadata or targets.
        for folder in ('baselines','ecg12gen','competition_interface'):
            shutil.copytree(root/folder,destination/folder,ignore=shutil.ignore_patterns('__pycache__','tests','test_data','*.pyc'))
        (destination/'configs').mkdir()
        for name in ('b5_final.json','b5_best_context.json','b5_submission.json'):
            shutil.copyfile(root/'configs'/name,destination/'configs'/name)
        shutil.copyfile(root/'competition_main.py',destination/'competition_main.py')
    manifest={'windows_exe_built':build_exe,'training_started':False,'base_sha256':BASE_SHA,'body_sha256':BODY_SHA,
        'model_weights_changed':False,'test_data_included':False,'runtime_entry':'B5Submit.exe' if build_exe else 'python competition_main.py',
        'checks':{'weights_SHA256':True,'real_inference_checked':False,'clean_PC_checked':False},
        'files':{p.relative_to(destination).as_posix():sha(p) for p in destination.rglob('*') if p.is_file()},
        'run_instructions':'Place test_data beside B5Submit.exe (or competition_main.py).\n'
        'Run B5Submit.exe without arguments. Results: results/<userid>.csv.\n'
        'Four-point organizer samples support --inspect-input only.\n'
        'Keep _internal, models and submission.json with the EXE. Do not submit only the EXE.\n'
        'Clear/archive old results before another run. Missing sampling metadata requires an explicit verified config fallback.\n'}
    archive_path=_finish(destination,manifest,archive=archive)
    print({'package':str(destination),'zip':archive_path,'windows_exe_built':build_exe,
           'note':'Run on real 120s inputs and a clean Windows PC before submission; weight-copy/build success is not a performance test.'},flush=True)
    return manifest


def stage_g120_c3(destination,asset_root,*,build_exe=False,archive=False):
    """Stage the four selected weights plus G120 study and, optionally, the EXE."""
    root=Path(__file__).resolve().parents[2];destination=Path(destination).resolve();asset_root=Path(asset_root).resolve()
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('Use a new empty release directory')
    if archive and Path(str(destination)+'_package.zip').exists(): raise FileExistsError('Release ZIP already exists')
    if root==destination or destination in root.parents: raise ValueError('Cannot package into a source ancestor')
    if any(root/folder in destination.parents for folder in ('baselines','ecg12gen','competition_interface')):
        raise ValueError('Cannot create release inside a source package')
    sources={
        'base.pt':asset_root/'models/base.pt',
        'original_body.pt':asset_root/'models/original_body.pt',
        'g120.pt':asset_root/'models/g120_best.pt',
        'body_c3.pt':asset_root/'models/body_c3_seed42_best.pt',
        'slow120_study.json':asset_root/'config/slow120_study.json',
    }
    if any(not path.is_file() for path in sources.values()):
        missing=[str(path) for path in sources.values() if not path.is_file()]
        raise FileNotFoundError('Missing G120+C3 asset: '+', '.join(missing))
    from .g120_submission import resolve_assets
    setting=read(root/'configs/b5_submission_g120_c3.json')
    # Verify source bytes before staging, then verify the copied files once more.
    source_paths={**setting,
        'base_checkpoint':str(sources['base.pt']), 'original_body_checkpoint':str(sources['original_body.pt']),
        'g120_checkpoint':str(sources['g120.pt']), 'c3_checkpoint':str(sources['body_c3.pt']),
        'slow120_study':str(sources['slow120_study.json'])}
    resolve_assets(root,source_paths)
    destination.mkdir(parents=True);models=destination/'models';models.mkdir();config=destination/'config';config.mkdir()
    for name in ('base.pt','original_body.pt','g120.pt','body_c3.pt'):
        shutil.copyfile(sources[name],models/name)
    shutil.copyfile(sources['slow120_study.json'],config/'slow120_study.json')
    # Frozen Python modules are stored inside PyInstaller's archive, so retain
    # the exact small Slow120 source set beside the EXE for the study's runtime
    # fingerprint check.  These files are code provenance, never training data.
    runtime_source=destination/'runtime_source'/'B5_slow120';runtime_source.mkdir(parents=True)
    for source in (root/'baselines'/'B5_slow120').glob('*.py'):
        shutil.copyfile(source,runtime_source/source.name)
    shutil.copyfile(root/'configs/b5_submission_g120_c3.json',destination/'submission.json')
    staged=read(destination/'submission.json')
    resolve_assets(destination,staged)
    if build_exe:
        if os.name!='nt': raise RuntimeError('Windows EXE must be built on Windows, not Linux')
        for module in ('PyInstaller','torch','scipy','yaml','numpy'):
            if importlib.util.find_spec(module) is None: raise RuntimeError('Install build dependency in the active Windows environment: '+module)
        _build_exe(root,destination,
                   ('baselines.B5','baselines.B5_best','baselines.B5_best_context','baselines.B5_final','baselines.B5_slow120','ecg12gen','competition_interface'),
                   ((root/'configs/b5_final.json','configs'),(root/'configs/b5_best_context.json','configs'),
                    (root/'configs/b5_submission_g120_c3.json','configs/b5_submission.json')))
    else:
        for folder in ('baselines','ecg12gen','competition_interface'):
            shutil.copytree(root/folder,destination/folder,ignore=shutil.ignore_patterns('__pycache__','tests','test_data','*.pyc'))
        (destination/'configs').mkdir()
        for name in ('b5_final.json','b5_best_context.json','b5_submission_g120_c3.json'):
            shutil.copyfile(root/'configs'/name,destination/'configs'/name)
        shutil.copyfile(root/'competition_main.py',destination/'competition_main.py')
    manifest={'windows_exe_built':build_exe,'training_started':False,'model_variant':'G120+C3',
        'assets':resolve_assets(destination,staged).report(),'test_data_included':False,
        'runtime_entry':'B5Submit.exe' if build_exe else 'python competition_main.py',
        'checks':{'weights_SHA256':True,'model_forward_checked':False,'real_inference_checked':False,'clean_PC_checked':False},
        'files':{p.relative_to(destination).as_posix():sha(p) for p in destination.rglob('*') if p.is_file()},
        'run_instructions':'Place organizer test_data beside B5Submit.exe. Run B5Submit.exe without arguments.\n'
        'This G120+C3 package needs models/base.pt, models/original_body.pt, models/g120.pt, models/body_c3.pt and config/slow120_study.json.\n'
        'Keep _internal, models, config, runtime_source and submission.json together. Results are written to results/<userid>.csv.\n'
        'Clear/archive old results before another run. Do not include test_data, results, training caches or gold files in the submission ZIP.\n'}
    archive_path=_finish(destination,manifest,archive=archive)
    print({'package':str(destination),'zip':archive_path,'windows_exe_built':build_exe,
           'model_variant':'G120+C3','note':'No training or target data were used while staging/building.'},flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-checkpoint');p.add_argument('--body-checkpoint')
    p.add_argument('--asset-root',help='Asset root containing models/base.pt, original_body.pt, g120_best.pt, body_c3_seed42_best.pt and config/slow120_study.json')
    p.add_argument('--destination',required=True)
    p.add_argument('--build-exe',action='store_true',help='Windows only; omit for portable Python staging/test package')
    p.add_argument('--no-zip',action='store_true',help='Leave the requested release folder uncompressed')
    args=p.parse_args()
    if args.asset_root:
        if args.base_checkpoint or args.body_checkpoint: p.error('Use either --asset-root for G120+C3 or old --base-checkpoint/--body-checkpoint')
        stage_g120_c3(args.destination,args.asset_root,build_exe=args.build_exe,archive=not args.no_zip)
    else:
        if not args.base_checkpoint or not args.body_checkpoint: p.error('Old package mode requires --base-checkpoint and --body-checkpoint')
        stage(args.destination,args.base_checkpoint,args.body_checkpoint,build_exe=args.build_exe,archive=not args.no_zip)


if __name__=='__main__': main()
