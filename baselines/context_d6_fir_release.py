"""Optional loader for the named V6/FIR25 inference snapshot (no B5 mutation)."""
from pathlib import Path
import torch
from baselines.B5.config import ModelConfig
from baselines.B5.checkpoint import checkpoint_preprocessor
from baselines.context_d6_v6 import ContextD6V6UNet
from ecg12gen.preprocessing import PreprocessingConfig

def load_release(checkpoint_path, *, device='cpu', preprocessing_config='configs/preprocessing.yaml'):
    """Return V6 in eval mode and its embedded frozen preprocessor.

    Requires this branch's inherited V2--V6 model modules; not original B5 .pt.
    Caller must use record-context collation/masks and existing flow sampler.
    """
    ck=torch.load(Path(checkpoint_path),map_location='cpu',weights_only=True)
    if ck.get('release_id')!='d6-context-v6-fir25-20261017':raise ValueError('Wrong release checkpoint')
    if ck['architecture_id']!='B5-U-context-d6-v6-reliability-gated':raise ValueError('Not V6')
    cc=ck['context_config']
    model=ContextD6V6UNet(ModelConfig.from_dict(ck['model_config']),cc['dimension'],
        initial_query_fraction=cc['initial_query_fraction'],injection_layers=tuple(cc['injection_layers']),
        cross_attention_heads=cc['cross_attention_heads'],cross_attention_dropout=cc['cross_attention_dropout'],
        reliability_hidden=cc['reliability_hidden'],reliability_logit_clamp=cc['reliability_logit_clamp'])
    model.load_state_dict(ck['ema'],strict=True)
    preprocessor=checkpoint_preprocessor(ck,PreprocessingConfig.from_yaml(preprocessing_config))
    return model.to(torch.device(device)).eval(),preprocessor
