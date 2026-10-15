"""Self-contained inference from full 120s I and demographics; no prepared/target files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from baselines.B5.flow import sample
from baselines.B5.metadata import encode_demographics
from baselines.B5.runtime import device_from_name
from .checkpoint import load,model_from
from .model import BoundModel
from .slow import SlowCNN,predict_native,slow_window
from .common import read,write,sha


@torch.no_grad()
def predict_record(checkpoint,anchor_uV,metadata,record_key,device):
    raw=np.asarray(anchor_uV,dtype=np.float32)
    if raw.ndim==1: raw=raw[None]
    if raw.shape!=(1,60000) or not np.isfinite(raw).all():
        raise ValueError('Test input must be finite synchronous I [1,60000] at 500Hz; no target leads')
    if not str(record_key): raise ValueError('A stable record key is required for keyed noise')
    scale=np.asarray(checkpoint['scales']['d12'],np.float32)
    meta=encode_demographics(metadata)
    model=model_from(checkpoint,device)
    native=None
    if checkpoint['group']=='slow':
        network=SlowCNN().to(device).requires_grad_(False).eval()
        network.load_state_dict(checkpoint['slow_state'],strict=True)
        # Mean raw I first, then normalize, exactly as the S50 prepared interface.
        inputs={'anchor':torch.from_numpy((raw.reshape(1,6000,10).mean(-1)/scale[0])[None]).to(device)}
        inputs.update({k:torch.from_numpy(np.asarray(v)[None]).to(device) for k,v in meta.items()})
        native=predict_native(network,inputs)[0].cpu().numpy()
    pieces=[]
    for start in range(0,60000,5000):
        inputs={'anchor':torch.from_numpy((raw[:,start:start+5000]/scale[0])[None]).to(device)}
        inputs.update({k:torch.from_numpy(np.asarray(v)[None]).to(device) for k,v in meta.items()})
        condition=None if native is None else torch.from_numpy(slow_window(native,start)[None]).to(device)
        generated=sample(BoundModel(model,condition),inputs,[str(record_key)+':'+str(start)],**checkpoint['sampling'])
        pieces.append(generated[0].float().cpu().numpy()*scale[:,None])
    return np.concatenate(pieces,axis=-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--anchor-npy',required=True,help='Raw uV synchronous I [1,60000] at 500Hz')
    parser.add_argument('--metadata-json',help='Raw age/sex(or gender)/height/weight; fields may be missing')
    parser.add_argument('--record-key',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use empty inference output')
    checkpoint=load(args.checkpoint)
    raw=np.load(args.anchor_npy,allow_pickle=False)
    metadata=read(args.metadata_json) if args.metadata_json else {}
    result=predict_record(checkpoint,raw,metadata,args.record_key,device_from_name(args.device))
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/'prediction_uV.npy',result)
    write(output/'inference.json',{'checkpoint_sha256':sha(args.checkpoint),'record_key':args.record_key,
        'shape':list(result.shape),'unit':'uV','sampling':checkpoint['sampling'],'group':checkpoint['group'],
        'target_used_at_inference':False,'prepared_data_required':False,'slow_checkpoint_embedded':True})


if __name__=='__main__': main()
