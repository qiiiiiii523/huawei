"""B1 lightweight residual-dilated U-Net with the frozen C3 interface."""
from __future__ import annotations
import hashlib, json
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import ContractError, WINDOW_SAMPLES

N12, N6, CCTX = 12, 6, 64

class ResBlock(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__(); g=max(1,min(8,cout//4))
        self.conv1=nn.Conv1d(cin,cout,5,padding=2*dilation,dilation=dilation); self.n1=nn.GroupNorm(g,cout)
        self.conv2=nn.Conv1d(cout,cout,5,padding=2*dilation,dilation=dilation); self.n2=nn.GroupNorm(g,cout)
        self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x):
        y=F.silu(self.n1(self.conv1(x))); y=self.n2(self.conv2(y)); return F.silu(y+self.skip(x))

class ContextEncoder(nn.Module):
    def __init__(self, channels, d6=False):
        super().__init__(); self.d6=d6; self.channels=channels
        self.net=nn.Sequential(nn.Conv1d(1,32,9,padding=4),nn.GroupNorm(4,32),nn.SiLU(),
                               nn.Conv1d(32,CCTX,9,padding=4),nn.GroupNorm(8,CCTX),nn.SiLU())
        self.proj=nn.Sequential(nn.LayerNorm(CCTX),nn.Linear(CCTX,CCTX),nn.SiLU(),nn.LayerNorm(CCTX))
        self.lead=nn.Parameter(torch.zeros(N6,CCTX)) if d6 else None
        nn.init.normal_(self.lead,std=.02) if self.lead is not None else None
    def forward(self,x,mask=None):
        if x.ndim!=3 or x.shape[-1]!=WINDOW_SAMPLES: raise ContractError("invalid context shape")
        z=self.net(x.reshape(-1,1,WINDOW_SAMPLES)); z=F.adaptive_avg_pool1d(z,1).reshape(x.shape[0],x.shape[1],CCTX)
        if self.d6:
            if mask is None or mask.shape!=(x.shape[0],N6): raise ContractError("d6 mask required")
            idx=torch.argsort(mask.to(torch.int64),dim=1)[:,-x.shape[1]:]; idx=torch.sort(idx,dim=1).values
            z=z+self.lead[idx]
        return self.proj(z.mean(1))

class B1Model(nn.Module):
    def __init__(self, fusion_mode="none", dropout=.1):
        super().__init__();
        if fusion_mode not in {"none","film_gated_residual"}: raise ContractError("unsupported fusion_mode")
        self.fusion_mode=fusion_mode; w=(24,48,96,192)
        self.enc=nn.ModuleList([ResBlock(a,b,1 if i<2 else 2) for i,(a,b) in enumerate(zip((1,)+w[:-1],w))])
        self.down=nn.ModuleList([nn.Conv1d(w[i],w[i],4,stride=2,padding=1) for i in range(3)])
        self.dec=nn.ModuleList([ResBlock(w[i+1]+w[i],w[i],1) for i in (2,1,0)])
        self.out=nn.Conv1d(w[0],N12,1); self.dropout=nn.Dropout(dropout)
        self.watch_context_encoder=ContextEncoder(1,False); self.machine_d6_context_encoder=ContextEncoder(6,True); self.body_d6_context_encoder=ContextEncoder(6,True)
        self.film=nn.Linear(CCTX,2*w[-1]); self.gate=nn.Linear(CCTX,N12)
        self.residual_adapter=nn.Sequential(nn.Conv1d(w[0]+CCTX,w[0],3,padding=1),nn.GroupNorm(6,w[0]),nn.SiLU(),nn.Conv1d(w[0],1,3,padding=1))
        self.baseline_head=nn.Sequential(nn.Linear(w[-1],64),nn.SiLU(),nn.Linear(64,N12)); self._init_adapters()
    def _init_adapters(self):
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias); nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias,-2.944439)
        nn.init.zeros_(self.residual_adapter[-1].weight); nn.init.zeros_(self.residual_adapter[-1].bias); nn.init.zeros_(self.baseline_head[-1].weight); nn.init.zeros_(self.baseline_head[-1].bias)
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
    @property
    def architecture_id(self): return "B1_residual_dilated_unet_feature_c3_v1"
    @property
    def architecture_config(self): return {"architecture_id":self.architecture_id,"widths":[24,48,96,192],"downsample":"stride2_x3","norm":"GroupNorm","c3":"feature_film_then_gated_residual","context_dim":CCTX}
    @property
    def architecture_config_hash(self): return hashlib.sha256(json.dumps(self.architecture_config,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    @property
    def anchor_parameter_names(self):
        p=("watch_context_encoder.","machine_d6_context_encoder.","body_d6_context_encoder.","film.","gate.","residual_adapter.")
        return tuple(n for n,_ in self.named_parameters() if not n.startswith(p))
    def _anchor(self,x):
        if x.ndim!=3 or x.shape[1:]!=(1,WINDOW_SAMPLES): raise ContractError("anchor_i must be [B,1,5000]")
        skips=[]; z=x
        for i,e in enumerate(self.enc):
            z=e(z); skips.append(z)
            if i<3: z=self.down[i](z)
        bottleneck=z
        for d,s in zip(self.dec,reversed(skips[:-1])):
            z=d(torch.cat((F.interpolate(z,size=s.shape[-1],mode='linear',align_corners=False),s),1))
        return self.dropout(z),bottleneck
    def _ctx(self,c,src,mask):
        srcs=[src]*c.shape[0] if isinstance(src,str) else list(src)
        if any(s not in {"watch_ecg","ecg_machine_d6","body_scale_d6"} for s in srcs): raise ContractError("invalid context source")
        if all(s=="watch_ecg" for s in srcs): return self.watch_context_encoder(c)
        if any(s=="watch_ecg" for s in srcs): raise ContractError("mixed context sources")
        out=[]
        for i,s in enumerate(srcs): out.append((self.machine_d6_context_encoder if s=="ecg_machine_d6" else self.body_d6_context_encoder)(c[i:i+1],mask[i:i+1])[0])
        return torch.stack(out)
    def forward(self,anchor_i,*,context_ecg=None,context_source_type=None,context_lead_mask=None):
        high,bottleneck=self._anchor(anchor_i)
        if self.fusion_mode=="none": return self.out(high)
        if context_ecg is None or context_source_type is None: raise ContractError("context required")
        c=self._ctx(context_ecg,context_source_type,context_lead_mask); gam,bet=self.film(c).chunk(2,1); bottleneck=bottleneck*(1+gam[:,:,None])+bet[:,:,None]
        # C3 FiLM is applied before the decoder; decode once more from conditioned bottleneck.
        skips=[]; z=anchor_i
        for i,e in enumerate(self.enc):
            z=e(z); skips.append(z)
            if i<3: z=self.down[i](z)
        z=bottleneck
        for d,s in zip(self.dec,reversed(skips[:-1])): z=d(torch.cat((F.interpolate(z,size=s.shape[-1],mode='linear',align_corners=False),s),1))
        pred=self.out(z); bsz=anchor_i.shape[0]; cmap=c[:,None,:,None].expand(bsz,N12,CCTX,WINDOW_SAMPLES); h=z[:,None].expand(bsz,N12,z.shape[1],WINDOW_SAMPLES); din=torch.cat((h,cmap),2).reshape(bsz*N12,z.shape[1]+CCTX,WINDOW_SAMPLES); delta=self.residual_adapter(din).reshape(bsz,N12,WINDOW_SAMPLES); return pred+torch.sigmoid(self.gate(c))[:,:,None]*delta
    @torch.no_grad()
    def predict_baseline(self,anchor_i): return self.baseline_head(self._anchor(anchor_i)[1].mean(-1))
