"""Original C5 EMA + learned1s mean corrector; targets never enter inference."""
import argparse,json,sys,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baselines.B7.checkpoint import load_checkpoint,model_from_checkpoint,file_sha256
from baselines.B7.config import SamplingConfig
from baselines.B7.inference import predict_records
from mean_model import MeanCorrector,apply_correction,GENERATED

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['base-checkpoint','corrector-checkpoint','anchor','record-key','output']:p.add_argument('--'+arg,required=True)
    p.add_argument('--metadata',help='JSON with numeric,sex,field_mask,age_topcoded')
    p.add_argument('--device',default='cpu');a=p.parse_args()
    out=Path(a.output);offsets=out.with_suffix('.offsets.npy');provenance=out.with_suffix('.json')
    if any(f.exists() for f in [out,offsets,provenance]):raise FileExistsError('Use fresh output paths')
    base=load_checkpoint(a.base_checkpoint)
    correction=torch.load(a.corrector_checkpoint,map_location='cpu',weights_only=True)
    spec=correction['specification']
    if file_sha256(a.base_checkpoint)!=spec['checkpoint_sha256']:
        raise ValueError('Corrector must use its exact frozen C5 base checkpoint')
    if list(base['scales_uV'])!=list(spec['scales_uV']) or base['sampling']!=spec['sampling']:
        raise ValueError('Frozen scale or sampling protocol changed')
    anchor=np.load(a.anchor,allow_pickle=False).astype('float32')
    if anchor.shape==(60000,):anchor=anchor[None]
    if anchor.shape!=(1,60000) or not np.isfinite(anchor).all():raise ValueError('Anchor must be rawuV500Hz full120s')
    md=json.loads(Path(a.metadata).read_text()) if a.metadata else None
    config=SamplingConfig(**base['sampling']);model=model_from_checkpoint(base,a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    predicted=predict_records(model,[anchor],base['scales_uV'],[a.record_key],[md],config)[0].cpu().numpy()
    corrector=MeanCorrector().to(a.device).eval();corrector.load_state_dict(correction['model'],strict=True)
    result,delta=apply_correction(corrector,anchor,predicted[list(GENERATED)],base['scales_uV'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:np.save(f,result,allow_pickle=False)
    with offsets.open('wb') as f:np.save(f,delta,allow_pickle=False)
    provenance.write_text(json.dumps(dict(base_sha256=spec['checkpoint_sha256'],corrector_sha256=file_sha256(a.corrector_checkpoint),
        corrector_epoch=correction['epoch'],sample_rate=500,seconds=120,base_weights='EMA',
        correction='II,V1-V6 1s offsets, repeat500, fixed I and limb formula',target_inputs=False),indent=2))
    print(str(out))

if __name__=='__main__':main()
