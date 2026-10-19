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

from baselines.B5_best.common import sha,write
from .profile import resolve_assets,BASE_SHA,BODY_SHA


def stage(destination,base_checkpoint,body_checkpoint,*,build_exe=False):
    root=Path(__file__).resolve().parents[2];destination=Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('Use a new empty release directory')
    if Path(str(destination)+'_package.zip').exists(): raise FileExistsError('Release ZIP already exists')
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
        with tempfile.TemporaryDirectory(prefix='b5_windows_build_') as temp:
            temp=Path(temp)
            command=[sys.executable,'-m','PyInstaller','--clean','--onedir','--console','--name','B5Submit',
                '--distpath',str(temp/'dist'),'--workpath',str(temp/'work'),'--specpath',str(temp),
                '--paths',str(root)]
            for package in ('baselines.B5','baselines.B5_best','baselines.B5_best_context','baselines.B5_final','ecg12gen','competition_interface'):
                command+=['--collect-submodules',package]
            for name in ('b5_final.json','b5_best_context.json'):
                command+=['--add-data',str(root/'configs'/name)+os.pathsep+'configs']
            command+=['--add-data',str(root/'configs/b5_submission.json')+os.pathsep+'configs',
                      str(root/'competition_main.py')]
            subprocess.run(command,cwd=root,check=True)
            shutil.copytree(temp/'dist/B5Submit',destination,dirs_exist_ok=True)
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
        'files':{p.relative_to(destination).as_posix():sha(p) for p in destination.rglob('*') if p.is_file()}}
    write(destination/'package_manifest.json',manifest)
    (destination/'RUN.txt').write_text('Place test_data beside B5Submit.exe (or competition_main.py).\n'
        'Run B5Submit.exe without arguments. Results: results/<userid>.csv.\n'
        'Four-point organizer samples support --inspect-input only.\n'
        'Keep _internal, models and submission.json with the EXE. Do not submit only the EXE.\n'
        'Clear/archive old results before another run. Missing sampling metadata requires an explicit verified config fallback.\n',encoding='utf-8')
    archive=shutil.make_archive(str(destination)+'_package','zip',destination.parent,destination.name)
    print({'package':str(destination),'zip':archive,'windows_exe_built':build_exe,
           'note':'Run on real 120s inputs and a clean Windows PC before submission; weight-copy/build success is not a performance test.'},flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-checkpoint',required=True);p.add_argument('--body-checkpoint',required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--build-exe',action='store_true',help='Windows only; omit for portable Python staging/test package')
    args=p.parse_args();stage(args.destination,args.base_checkpoint,args.body_checkpoint,build_exe=args.build_exe)


if __name__=='__main__': main()
