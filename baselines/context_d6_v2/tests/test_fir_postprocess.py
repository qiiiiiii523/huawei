import numpy as np
import pytest
from baselines.context_d6_fir_postprocess import historical_fir,apply_postprocess,HistoricalFIR
from baselines.context_d6_fir_postprocess import B,C

@pytest.mark.parametrize('kind',['missing','short','bad','nan','flat'])
def test_fallback(kind):
    h=np.ones((6,20000));m=np.ones_like(h,dtype=bool)
    if kind=='missing':h=m=None
    if kind=='short':h=h[:,:200];m=m[:,:200]
    if kind=='bad':m[:]=False
    if kind=='nan':h[:]=np.nan
    model=historical_fir(h,m);assert not model.available
    p=np.random.default_rng(1).normal(size=(12,500));i=p[:1].copy()
    out,status=apply_postprocess(p,i,model)
    q=(B*(p[1:6]-C*i)).sum(axis=0)/(B*B).sum()
    assert np.allclose(out[1:6],B*q+C*i);assert not status['personal_fir_used']
    assert np.array_equal(out[6:],p[6:])

def test_no_mutation_and_contract():
    rng=np.random.default_rng(3);p=rng.normal(size=(12,500));old=p.copy();i=rng.normal(size=(1,500))
    model=HistoricalFIR(np.ones(61)/61,0.,True,'ok')
    out,status=apply_postprocess(p,i,model)
    assert status['personal_fir_used'];assert np.array_equal(old,p);assert np.array_equal(out[6:],p[6:])
    assert np.array_equal(out[0],p[0]);assert np.allclose(out[2],out[1]-i[0])
    with pytest.raises(ValueError):apply_postprocess(p[:6],i,model)
    with pytest.raises(ValueError):apply_postprocess(p,i,model,strength=.3)
    i[:]=np.nan
    with pytest.raises(ValueError):apply_postprocess(p,i,model)

def test_known_fir_and_masked_history():
    from baselines.context_d6_fir_postprocess import _predict
    rng=np.random.default_rng(42);i=rng.normal(size=20000)*200
    coef=np.zeros(61);coef[30]=.7;coef[29]=.2
    ii=_predict(i,(coef,.02));history=np.stack([i,ii]);mask=np.ones_like(history,dtype=bool)
    model=historical_fir(history,mask)
    assert model.available and model.holdout_r>.99
    p=rng.normal(size=(12,len(i)));a,status=apply_postprocess(p,i,model)
    q=(B*(p[1:6]-C*i)).sum(axis=0)/(B*B).sum()
    expected=.75*q+.25*_predict(i,(model.coef,model.bias))
    assert np.allclose(a[1:6],B*expected+C*i)
    assert status['personal_fir_used']
    masked=history.copy();masked[:,1000:2000]=np.nan;mask[:,1000:2000]=False
    assert historical_fir(masked,mask).available
