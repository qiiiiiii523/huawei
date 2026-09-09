from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from .contracts import ContractError, D12_LEADS, WINDOW_SAMPLES
ARCHITECTURE_VERSION='M1-axial-lead-time-v1'
ARCHITECTURE_ID='M1-P0-multiscale-cnn-axial-lead-time'
DEFAULT_CONFIG={'architecture_version':ARCHITECTURE_VERSION,'architecture_id':ARCHITECTURE_ID,'d_model':128,'num_blocks':4,'num_heads':4,'ffn_dim':256,'dropout':.1,'cnn_channels':[64,128,256],'decoder_channels':[128,64],'time_tokens':250,'lead_order':list(D12_LEADS)}
def architecture_config_hash(config=None):
    value=dict(DEFAULT_CONFIG if config is None else config); return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def merged_config(config=None):
    value=dict(DEFAULT_CONFIG); value.update(config or {})
    if value['architecture_version']!=ARCHITECTURE_VERSION: raise ValueError('M1 requires Axial architecture_version')
    if tuple(value['lead_order'])!=D12_LEADS: raise ContractError('M1 lead order must match canonical d12 order')
    if not 4<=int(value['num_blocks'])<=8: raise ValueError('num_blocks must be [4,8]')
    if int(value['d_model'])%int(value['num_heads']): raise ValueError('d_model must divide heads')
    if list(value['cnn_channels'])!=[64,128,256] or int(value['time_tokens'])!=250: raise ValueError('M1 scale/token defaults are fixed')
    return value
class ConvNormAct(nn.Module):
    def __init__(self,cin,cout,kernel=5,stride=1,dilation=1):
        super().__init__(); pad=(kernel-1)*dilation//2; self.net=nn.Sequential(nn.Conv1d(cin,cout,kernel,stride,pad,dilation=dilation),nn.GroupNorm(min(8,cout),cout),nn.GELU())
    def forward(self,x): return self.net(x)
class LocalResidualBlock(nn.Module):
    def __init__(self,c,dilation,p):
        super().__init__(); self.a=ConvNormAct(c,c,5,dilation=dilation); self.b=nn.Sequential(nn.Conv1d(c,c,3,padding=dilation,dilation=dilation),nn.GroupNorm(min(8,c),c),nn.Dropout(p))
    def forward(self,x): return F.gelu(x+self.b(self.a(x)))
class MultiScaleCNNEncoder(nn.Module):
    def __init__(self,p=.1):
        super().__init__(); self.stem=ConvNormAct(1,64,7); self.f0=nn.Sequential(LocalResidualBlock(64,1,p),LocalResidualBlock(64,2,p)); self.to_f1=ConvNormAct(64,128,7,stride=4); self.f1=nn.Sequential(LocalResidualBlock(128,1,p),LocalResidualBlock(128,2,p)); self.to_f2=ConvNormAct(128,256,5,stride=5); self.f2=nn.Sequential(LocalResidualBlock(256,1,p),LocalResidualBlock(256,2,p))
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(1,5000): raise ContractError('M1 anchor must be [B,1,5000]')
        f0=self.f0(self.stem(x)); f1=self.f1(self.to_f1(f0)); f2=self.f2(self.to_f2(f1))
        if f0.shape[1:]!=(64,5000) or f1.shape[1:]!=(128,1250) or f2.shape[1:]!=(256,250): raise RuntimeError('M1 CNN scale contract failed')
        return f0,f1,f2
class FeedForward(nn.Module):
    def __init__(self,d,h,p): super().__init__(); self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Dropout(p),nn.Linear(h,d),nn.Dropout(p))
    def forward(self,x): return self.net(x)
class AxialLeadTimeBlock(nn.Module):
    def __init__(self,d=128,heads=4,ffn=256,p=.1):
        super().__init__(); self.time_norm=nn.LayerNorm(d); self.time_attention=nn.MultiheadAttention(d,heads,dropout=p,batch_first=True); self.time_ffn_norm=nn.LayerNorm(d); self.time_ffn=FeedForward(d,ffn,p); self.lead_norm=nn.LayerNorm(d); self.lead_attention=nn.MultiheadAttention(d,heads,dropout=p,batch_first=True); self.lead_ffn_norm=nn.LayerNorm(d); self.lead_ffn=FeedForward(d,ffn,p); self.dropout=nn.Dropout(p); self.time_attention_calls=0; self.lead_attention_calls=0; self.last_time_is_causal=False
    def forward(self,z):
        if z.ndim!=4 or z.shape[1:3]!=(12,250): raise ContractError('Axial block expects [B,12,250,d]')
        b,_,t,d=z.shape; x=z.reshape(b*12,t,d); q=self.time_norm(x)
        y,_=self.time_attention(q,q,q,need_weights=False,is_causal=False); self.time_attention_calls+=1; x=x+self.dropout(y); x=x+self.time_ffn(self.time_ffn_norm(x)); z=x.reshape(b,12,t,d)
        x=z.permute(0,2,1,3).reshape(b*t,12,d); q=self.lead_norm(x); y,_=self.lead_attention(q,q,q,need_weights=False,is_causal=False); self.lead_attention_calls+=1; x=x+self.dropout(y); x=x+self.lead_ffn(self.lead_ffn_norm(x)); return x.reshape(b,t,12,d).permute(0,2,1,3)
class LeadConditionedSkip(nn.Module):
    def __init__(self,cin,cout,d): super().__init__(); self.signal=nn.Conv1d(cin,cout,1); self.lead=nn.Linear(d,cout)
    def forward(self,x,e): return self.signal(x)[:,None]+self.lead(e)[None,:,:,None]
class PerLeadConv(nn.Module):
    def forward(self,x,layer): b,l,c,t=x.shape; return layer(x.reshape(b*l,c,t)).reshape(b,l,-1,t)
class LeadTimeDecoder(nn.Module):
    def __init__(self,d,p):
        super().__init__(); self.coarse=nn.Linear(d,128); self.r1250=nn.Sequential(nn.Conv1d(128,128,5,padding=2),nn.GroupNorm(8,128),nn.GELU(),nn.Dropout(p)); self.skip1=LeadConditionedSkip(128,128,d); self.fuse1=nn.Sequential(nn.Conv1d(256,128,3,padding=1),nn.GroupNorm(8,128),nn.GELU()); self.r5000=nn.Sequential(nn.Conv1d(128,64,5,padding=2),nn.GroupNorm(8,64),nn.GELU(),nn.Dropout(p)); self.skip0=LeadConditionedSkip(64,64,d); self.fuse0=nn.Sequential(nn.Conv1d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU()); self.head=nn.Conv1d(64,1,3,padding=1); self.lead_bias=nn.Linear(d,1); self.each=PerLeadConv()
    def forward(self,z,f0,f1,e):
        b=z.shape[0]; x=self.coarse(z).permute(0,1,3,2); x=F.interpolate(x.reshape(b*12,128,250),size=1250,mode='linear',align_corners=False); x=self.each(x.reshape(b,12,128,1250),self.r1250); x=self.each(torch.cat((x,self.skip1(f1,e)),2),self.fuse1); x=F.interpolate(x.reshape(b*12,128,1250),size=5000,mode='linear',align_corners=False); x=self.each(x.reshape(b,12,128,5000),self.r5000); x=self.each(torch.cat((x,self.skip0(f0,e)),2),self.fuse0); return self.each(x,self.head).squeeze(2)+self.lead_bias(e).view(1,12,1)
class WatchContextEncoder(nn.Module):
    def __init__(self,d): super().__init__(); self.stem=nn.Sequential(ConvNormAct(1,64,9),LocalResidualBlock(64,2,.1)); self.proj=nn.Linear(64,d); self.lead_embedding=nn.Parameter(torch.randn(1,d)*.02); self.source_embedding=nn.Parameter(torch.randn(1,d)*.02)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(1,5000): raise ContractError('watch context must be [B,1,5000]')
        return self.proj(self.stem(x).mean(-1))+self.lead_embedding+self.source_embedding
class _D6ContextEncoder(nn.Module):
    source_type=''
    def __init__(self,d): super().__init__(); self.lead_stem=nn.Sequential(ConvNormAct(1,16,9),LocalResidualBlock(16,2,.1)); self.proj=nn.Linear(16,d); self.lead_embedding=nn.Parameter(torch.randn(6,d)*.02); self.source_embedding=nn.Parameter(torch.randn(1,d)*.02)
    def forward(self,x,mask=None):
        if x.ndim!=3 or x.shape[1:]!=(6,5000): raise ContractError('task2 context must be canonical d6 [B,6,5000]')
        if mask is not None and (mask.shape!=(x.shape[0],6) or not bool(mask.all())): raise ContractError('M1 task2 requires complete canonical d6')
        b=x.shape[0]; per=self.lead_stem(x.reshape(b*6,1,5000)).mean(-1).reshape(b,6,16); return (self.proj(per)+self.lead_embedding[None]).mean(1)+self.source_embedding
class BodyD6ContextEncoder(_D6ContextEncoder): source_type='body_scale_d6'
class MachineD6ContextEncoder(_D6ContextEncoder): source_type='ecg_machine_d6'
@dataclass(frozen=True)
class M1ForwardTrace:
    z_shape:tuple[int,...]; time_attention_calls:int; lead_attention_calls:int; time_attention_is_causal:bool
class M1AxialLeadTimeModel(nn.Module):
    def __init__(self,*,fusion_mode='none',task_id='task1',config=None,context_dropout=0.,source_dropout=0.):
        super().__init__();
        if fusion_mode not in {'none','film','gated_residual','film_gated_residual'}: raise ValueError('unsupported fusion_mode')
        if task_id not in {'task1','task2'}: raise ValueError('task_id must be task1 or task2')
        self.config=merged_config(config); self.fusion_mode=fusion_mode; self.task_id=task_id; self.context_dropout=float(context_dropout); self.source_dropout=float(source_dropout); d=int(self.config['d_model']); self.d_model=d
        self.cnn_encoder=MultiScaleCNNEncoder(float(self.config['dropout'])); self.anchor_projection=nn.Linear(256,d); self.time_position=nn.Parameter(torch.randn(1,250,d)*.02); self.lead_embedding=nn.Parameter(torch.randn(12,d)*.02); self.lead_state_embedding=nn.Parameter(torch.randn(2,d)*.02); self.axial_blocks=nn.ModuleList([AxialLeadTimeBlock(d,int(self.config['num_heads']),int(self.config['ffn_dim']),float(self.config['dropout'])) for _ in range(int(self.config['num_blocks']))]); self.final_norm=nn.LayerNorm(d); self.decoder=LeadTimeDecoder(d,float(self.config['dropout']))
        self.watch_context_encoder=None; self.body_d6_context_encoder=None; self.machine_d6_context_encoder=None; self.film=None; self.gate=None; self.residual=None
        if fusion_mode!='none':
            if task_id=='task1': self.watch_context_encoder=WatchContextEncoder(d)
            else: self.body_d6_context_encoder=BodyD6ContextEncoder(d); self.machine_d6_context_encoder=MachineD6ContextEncoder(d)
            if fusion_mode in {'film','film_gated_residual'}: self.film=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2*d)); nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
            if fusion_mode in {'gated_residual','film_gated_residual'}: self.gate=nn.Linear(d,1); nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias,-2.944439); self.residual=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,1)); nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
        self.last_trace=M1ForwardTrace((0,),0,0,False)
    @property
    def architecture_metadata(self): return {'architecture_version':ARCHITECTURE_VERSION,'architecture_id':ARCHITECTURE_ID,'architecture_config':dict(self.config),'architecture_config_hash':architecture_config_hash(self.config),'lead_order':list(D12_LEADS),'d_model':self.d_model,'fusion_mode':self.fusion_mode,'task_id':self.task_id}
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
    def _context(self,x,source,mask):
        if self.task_id=='task1':
            if source!='watch_ecg' or self.watch_context_encoder is None: raise ContractError('task1 requires exactly watch_ecg context')
            z=self.watch_context_encoder(x)
        elif source=='body_scale_d6' and self.body_d6_context_encoder is not None: z=self.body_d6_context_encoder(x,mask)
        elif source=='ecg_machine_d6' and self.machine_d6_context_encoder is not None: z=self.machine_d6_context_encoder(x,mask)
        else: raise ContractError('task2 requires exactly one d6 context source')
        p=max(self.context_dropout,self.source_dropout)
        if self.training and p>0 and torch.rand((),device=z.device)<p:z=torch.zeros_like(z)
        return z
    def _grid(self,anchor,mask):
        f0,f1,f2=self.cnn_encoder(anchor); h=self.anchor_projection(f2.transpose(1,2))+self.time_position
        if h.shape[1:]!=(250,self.d_model): raise RuntimeError('M1 H must be [B,250,d]')
        observed=torch.zeros((anchor.shape[0],12),dtype=torch.bool,device=anchor.device); observed[:,0]=True
        if mask is not None and (mask.shape!=observed.shape or not torch.equal(mask.to(torch.bool),observed)): raise ContractError('target-time lead mask must mark only I')
        return h[:,None]+self.lead_embedding[None,:,None]+self.lead_state_embedding[(~observed).long()][:,:,None,:],f0,f1
    def forward(self,anchor_i,*,context=None,context_source_type=None,context_lead_mask=None,lead_mask=None,return_trace=False):
        if anchor_i.ndim!=3 or anchor_i.shape[1:]!=(1,5000): raise ContractError('M1 input must be [B,1,5000]')
        if self.fusion_mode=='none':
            if context is not None or context_source_type is not None or context_lead_mask is not None: raise ContractError('none path must not read context')
        elif context is None or context_source_type is None: raise ContractError('P1 requires context and explicit source type')
        z,f0,f1=self._grid(anchor_i,lead_mask)
        if self.fusion_mode!='none':
            z_context=self._context(context,context_source_type,context_lead_mask)
            if self.fusion_mode in {'film','film_gated_residual'}:
                gamma,beta=self.film(z_context).chunk(2,-1); z=z*(1+gamma[:,None,None])+beta[:,None,None]
        for block in self.axial_blocks:
            block.time_attention_calls=0
            block.lead_attention_calls=0
            z=block(z)
        z=self.final_norm(z); base=self.decoder(z,f0,f1,self.lead_embedding); prediction=base
        if self.fusion_mode in {'gated_residual','film_gated_residual'}:
            delta=self.residual(z+z_context[:,None,None]).squeeze(-1); delta=F.interpolate(delta.reshape(-1,1,250),size=5000,mode='linear',align_corners=False).reshape(-1,12,5000); prediction=base+torch.sigmoid(self.gate(z_context)).view(-1,1,1)*delta
        trace=M1ForwardTrace(tuple(z.shape),sum(x.time_attention_calls for x in self.axial_blocks),sum(x.lead_attention_calls for x in self.axial_blocks),False); self.last_trace=trace
        return (prediction,trace) if return_trace else prediction
