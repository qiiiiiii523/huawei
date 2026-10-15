"""Read-only provenance and raw-array checks for post-hoc QC reassessment."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .objective import file_sha256

SAMPLING={'seed':42,'solver':'heun','steps':16,'samples':16}
IDENTITY=('pair_id','target_record_id','subject_id','start_sample_500hz','input_type')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def state_digest(state):
    digest=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        digest.update(key.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def payload_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def identity(row):
    if any(not str(row.get(k,'')) for k in IDENTITY): raise ValueError('Missing window identity fields')
    return tuple(str(row[k]) for k in IDENTITY)


def align_reference(rows,reference):
    lookup={identity(row):i for i,row in enumerate(reference['rows'])}
    keys=[identity(row) for row in rows]
    if len(lookup)!=len(reference['rows']) or len(set(keys))!=len(keys) or set(keys)!=set(lookup):
        raise ValueError('Cached/current validation population differs or has duplicate windows')
    indices=np.asarray([lookup[k] for k in keys])
    return indices,np.asarray(reference['quality'])[indices]


def checked_arrays(folder,reference,reference_folder=None):
    """Reuse old row order; compare every raw target sample against the current cache."""
    folder=Path(folder);source=Path(reference_folder) if reference_folder else folder
    rows=csv_read(source/'window_metadata.csv')
    indices,quality=align_reference(rows,reference)
    prediction=np.load(folder/'prediction_uV.npy',mmap_mode='r',allow_pickle=False)
    target=np.load(source/'target_uV.npy',mmap_mode='r',allow_pickle=False)
    if prediction.shape!=target.shape or target.shape!=(len(rows),12,5000):
        raise ValueError('Requires raw-uV [N,12,5000] matching prediction and target arrays')
    for start in range(0,len(rows),8):
        expected=reference['target'][indices[start:start+8]]
        if not np.array_equal(target[start:start+8],expected):
            raise ValueError('Cached targets differ from the current raw validation cache')
    return prediction,target,rows,quality


def verified_prepared(path,best_path):
    path=Path(path).resolve();info=read(path/'prepared.json')
    if info.get('protocol')!='B5-slow-probe-v1' or not info.get('completed'):
        raise ValueError('Requires a completed original slow-probe prepared directory')
    if info['checkpoint_sha256']!=file_sha256(best_path):
        raise ValueError('Prepared belongs to another B5 best.pt; do not substitute best_r.pt')
    for name,expected in info['files'].items():
        if file_sha256(path/name)!=expected: raise ValueError('Prepared file changed: '+name)
    evaluation=Path(info['evaluation_dir'])
    for name,expected in info['evaluation_files'].items():
        if file_sha256(evaluation/name)!=expected: raise ValueError('Frozen source evaluation changed: '+name)
    provenance=read(evaluation/'inference_sampling.json')
    if provenance.get('sampling')!=SAMPLING or provenance.get('context_used') is not False:
        raise ValueError('Original prepared evaluation must be Heun16/K16/seed42 without context')
    return info


def trusted_cached_evaluation(folder,checkpoint_path,ema_digest,prepared_info,prepared_ema):
    """Folder names and a checkpoint pathname alone cannot prove weight identity."""
    folder=Path(folder)
    try:
        meta=read(folder/'inference_sampling.json')
        if meta.get('sampling')!=SAMPLING or meta.get('context_used') is not False: return False
        if prepared_info and folder.resolve()==Path(prepared_info['evaluation_dir']).resolve():
            return ema_digest==prepared_ema
        if meta.get('checkpoint_sha256')!=file_sha256(checkpoint_path): return False
        if meta.get('model_ema_sha256')!=ema_digest: return False
        expected={task+'/'+name for task in ('task1','task2') for name in
                  ('prediction_uV.npy','target_uV.npy','window_metadata.csv')}
        files=meta.get('files',{})
        if not expected.issubset(files): return False
        return all(file_sha256(folder/name)==files[name] for name in expected)
    except (OSError,ValueError,KeyError,TypeError): return False


def evaluation_provenance(folder,checkpoint_path,checkpoint):
    folder=Path(folder)
    files={task+'/'+name:file_sha256(folder/task/name) for task in ('task1','task2')
           for name in ('prediction_uV.npy','target_uV.npy','window_metadata.csv')}
    return {'checkpoint':str(Path(checkpoint_path).resolve()),'checkpoint_sha256':file_sha256(checkpoint_path),
        'model_ema_sha256':state_digest(checkpoint['ema']),'sampling':SAMPLING,'context_used':False,
        'files':files,'target_used_at_inference':False}
