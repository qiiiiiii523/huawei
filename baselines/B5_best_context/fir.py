"""Optional raw-uV postprocessor. No target input, no default-pipeline mutation.

Fixed61-tap/alpha.01 historical I->II FIR, 25% mixture after limb projection.
Invalid history falls back to projected V6 (NOT the legacy public FIR fallback).
"""
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

B=np.array([1,1,-.5,-.5,1.])[:,None]
C=np.array([0,-1,-.5,1,-.5])[:,None]

def _design(signal,mask,length,start=0,stop=None,stride=5):
    stop=len(signal[0]) if stop is None else stop
    x=sliding_window_view(np.asarray(signal[0,start:stop],dtype=float)/1000,length)[::stride].copy()
    m=length//2
    y=np.asarray(signal[1,start+m:stop-m:stride],dtype=float)/1000
    good=sliding_window_view(mask[0,start:stop],length)[::stride].all(axis=1)&mask[1,start+m:stop-m:stride]
    return x[good],y[good]

def _sufficient(x,y):
    if len(y)<500:raise ValueError('fewer than500 valid training positions')
    mx=x.mean(axis=0);my=y.mean();xc=x-mx;yc=y-my
    return xc.T@xc/len(y),xc.T@yc/len(y),mx,float(my)

def _fit(stats,alpha):
    gram,cross,mx,my=stats
    coef=np.linalg.solve(gram+np.eye(len(cross))*max(float(np.trace(gram)/len(cross)),1e-8)*alpha,cross)
    return coef,float(my-mx@coef)

def _pearson(x,y):
    x=x.astype(np.float64,copy=False);y=y.astype(np.float64,copy=False)
    x=x-x.mean();y=y-y.mean();denominator=np.sqrt(np.sum(x*x)*np.sum(y*y))
    return float(np.sum(x*y)/denominator) if denominator>0 else float('nan')

def _predict(i,model):
    coef,bias=model;m=len(coef)//2
    return (np.convolve(np.pad(np.asarray(i,dtype=float)/1000,(m,m),mode='edge'),coef[::-1],mode='valid')+bias)*1000

@dataclass(frozen=True)
class HistoricalFIR:
    coef: np.ndarray | None
    bias: float
    available: bool
    reason: str
    holdout_r: float | None = None

def historical_fir(history_uV, valid_mask) -> HistoricalFIR:
    """Receives history only. Shape/missing/insufficient input safely disables FIR."""
    if history_uV is None or valid_mask is None:return HistoricalFIR(None,0.,False,'missing_history')
    h=np.asarray(history_uV,dtype=float);mask=np.asarray(valid_mask,dtype=bool)
    if h.ndim!=2 or h.shape[0]<2 or mask.shape!=h.shape:
        return HistoricalFIR(None,0.,False,'invalid_history_shape')
    h=h[:2];mask=mask[:2]&np.isfinite(h);h=np.where(mask,h,0.)
    if h.shape[1]<1000:return HistoricalFIR(None,0.,False,'short_history')
    if any(mask[j].sum()<500 or np.std(h[j,mask[j]])<1e-8 for j in range(2)):
        return HistoricalFIR(None,0.,False,'flat_or_insufficient_history')
    try:
        n=h.shape[1];stats=_sufficient(*_design(h,mask,61));model=_fit(stats,.01)
        internal=_sufficient(*_design(h,mask,61,0,int(n*.4)))
        x,y=_design(h,mask,61,int(n*.6),n)
        if len(y)<500:raise ValueError('insufficient holdout')
        coef,bias=_fit(internal,.01);r=_pearson(x@coef+bias,y)
        if not np.isfinite(r):raise ValueError('undefined holdout')
        if not np.isfinite(model[0]).all() or not np.isfinite(model[1]):raise ValueError('nonfinite model')
        return HistoricalFIR(model[0],model[1],True,'ok',r)
    except (ValueError,np.linalg.LinAlgError):
        return HistoricalFIR(None,0.,False,'insufficient_or_degenerate_history')

def apply_postprocess(prediction_uV, observed_i_uV, model: HistoricalFIR, *, strength=.25):
    """Raw physical units mandatory; missing current I/invalid V6 output is an error.

    Return [12,T] float64 copy and metadata. Does not change observed/predicted I,
    chest leads, caller arrays, model weights, or target labels.
    """
    p=np.asarray(prediction_uV);i=np.asarray(observed_i_uV,dtype=float)
    if p.ndim!=2 or p.shape[0]!=12 or i.shape not in [(p.shape[1],),(1,p.shape[1])]:
        raise ValueError('Expected prediction[12,T], observed I[T] or[1,T] in uV')
    if not np.isfinite(p).all() or not np.isfinite(i).all():raise ValueError('Nonfinite current input/prediction')
    if strength not in [.1,.25]:raise ValueError('Only approved fixed strengths .1/.25')
    i=i.reshape(1,-1);out=p.astype(float,copy=True)
    q=(B*(out[1:6]-C*i)).sum(axis=0)/float((B*B).sum())
    used=bool(model.available and model.coef is not None and np.shape(model.coef)==(61,)
              and np.isfinite(model.coef).all() and np.isfinite(model.bias))
    if used:q=(1-strength)*q+strength*_predict(i[0],(model.coef,model.bias))
    out[1:6]=B*q+C*i
    return out,{'personal_fir_used':used,'fallback':'none' if used else 'projected_v6',
                'history_reason':model.reason,'strength':strength}
