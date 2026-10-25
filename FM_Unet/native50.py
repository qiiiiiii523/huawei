"""50Hz state with every original500Hz I sample retained as10 raw feature channels."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import SlowUNet7
from .config import ModelConfig


def pack_i(anchor):
    if anchor.ndim!=3 or anchor.shape[1]!=1 or anchor.shape[-1]%10:
        raise ValueError('Need [B,1,10*T] current I at500Hz')
    return anchor[:,0].reshape(len(anchor),-1,10).transpose(1,2)


def to_native(target):
    return target.reshape(*target.shape[:-1],-1,10).mean(-1)


class Native50UNet(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config or ModelConfig()
        self.core=SlowUNet7(self.config)
        self.core.stem=nn.Conv1d(17,self.config.channels[0],7,padding=3)

    def forward(self,state,time,anchor,metadata):
        c=self.core
        if state.shape!=(len(anchor),7,anchor.shape[-1]//10):raise ValueError('State/anchor coordinates disagree')
        te=c.time_embedding(time);me=c.metadata_encoder(anchor,metadata)
        h=c.stem(torch.cat((pack_i(anchor),state),1))
        return c.forward_features(h,te,me)
