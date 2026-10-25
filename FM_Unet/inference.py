from pathlib import Path
import numpy as np
import torch
from .fm_experiment import build, predict_slow
from .config import ModelConfig

class SlowPredictor:
    def __init__(self, checkpoint=None, device="cuda"):
        checkpoint=checkpoint or Path(__file__).with_name("best.pt")
        self.cp=torch.load(checkpoint,map_location="cpu",weights_only=True)
        c=self.cp
        self.model=build(ModelConfig.from_dict(c["model_config"]),c["native_hz"],c["input_seconds"],c["output_seconds"]).to(device).eval()
        self.model.load_state_dict(c["ema"],strict=True)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    def predict(self, raw_i_uV, metadata, record_key, trajectory_batch=4):
        """Return [11,60000] slow uV: II,III,aVR,aVL,aVF,V1..V6.
        Metadata must use original B5 encoding, without batch dimension.
        """
        raw=np.asarray(raw_i_uV,dtype=np.float64).reshape(1,-1)
        if raw.shape!=(1,60000) or not np.isfinite(raw).all():
            raise ValueError("Need finite 120s I at 500Hz in microvolts")
        if not metadata or set(metadata)!={"numeric","sex","field_mask","age_topcoded"}:
            raise ValueError("Supply original B5 encoded demographics")
        return predict_slow(self.model,raw,metadata,self.cp["scale_uV"],self.cp["noise_std7"],"FM",str(record_key),steps=16,samples=16,trajectory_batch=trajectory_batch)
