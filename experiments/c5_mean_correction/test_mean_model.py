"""Behavioral checks: exact scoring, no target input leakage, identity/learnability."""
import unittest
import numpy as np
import torch
from mean_model import (MeanCorrector, make_example, make_features, apply_correction, compose11, offsets11,
                        statistics_metrics, learning_loss)


class MeanCorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.rng=np.random.default_rng(42)
        self.a=self.rng.normal(size=(1,60000))*100
        self.p=self.rng.normal(size=(7,60000))*150
        self.y=self.rng.normal(size=(12,60000))*200
        self.mask=np.ones(12,dtype=bool)
        self.scales=np.arange(12)*10+100.

    def test_moments_match_direct_full_record_scoring(self):
        ex=make_example(self.a,self.p,self.y,self.mask,self.scales)
        delta=self.rng.normal(size=(7,120))*30
        d11=offsets11(torch.from_numpy(delta)[None])
        r,rmse=statistics_metrics(*(torch.from_numpy(ex[k])[None] for k in ('pm','ym','pss','yss','cross')),d11)
        direct=compose11(self.a,self.p+np.repeat(delta,500,axis=-1))
        for j in range(11):
            self.assertAlmostEqual(float(r[0,j]),np.corrcoef(direct[j],self.y[j+1])[0,1],places=12)
            self.assertAlmostEqual(float(rmse[0,j]),np.sqrt(np.mean((direct[j]-self.y[j+1])**2)),places=10)

    def test_target_free_prediction_matches_training_features_and_known_offsets(self):
        ex=make_example(self.a,self.p,self.y,self.mask,self.scales)
        visible=np.concatenate((self.a,self.p)).astype(np.float64)
        expected=visible.reshape(8,6000,10).mean(-1)/self.scales[[0,1,6,7,8,9,10,11],None]
        pos=(np.arange(6000)*10+4.5)/30000-1
        expected=np.concatenate((expected,pos[None])).astype(np.float32)
        np.testing.assert_array_equal(make_features(self.a,self.p,self.scales),expected)
        np.testing.assert_array_equal(ex['features'],expected)
        m=MeanCorrector().eval()
        self.assertEqual(sum(p.numel() for p in m.parameters()),31687)
        with torch.no_grad():m.head.bias.fill_(.1)
        actual,delta=apply_correction(m,self.a,self.p,self.scales)
        delta_expected=np.repeat((.1*self.scales[[1,6,7,8,9,10,11]])[:,None],120,axis=1)
        np.testing.assert_allclose(delta,delta_expected,rtol=1e-6)
        expected_output=np.concatenate((self.a,compose11(self.a,self.p+np.repeat(delta_expected,500,axis=-1)))).astype(np.float32)
        np.testing.assert_allclose(actual,expected_output,rtol=1e-6,atol=1e-4)

    def test_targets_cannot_change_features_and_masked_nan_is_ignored(self):
        first=make_example(self.a,self.p,self.y,self.mask,self.scales)
        y=self.y.copy()*100
        mask=self.mask.copy();mask[6]=False;y[6]=np.nan
        second=make_example(self.a,self.p,y,mask,self.scales)
        np.testing.assert_array_equal(first['features'],second['features'])
        self.assertTrue(np.isfinite(second['ym']).all())

    def test_zero_initialization_is_identity_and_preserves_limb_algebra(self):
        model=MeanCorrector().eval()
        inp=torch.randn(2,9,6000)
        with torch.no_grad():
            prediction=model(inp)
        self.assertEqual(tuple(prediction.shape),(2,7,120))
        self.assertTrue(torch.equal(prediction,torch.zeros_like(prediction)))
        delta=self.rng.normal(size=(7,120))
        actual=compose11(self.a,self.p+np.repeat(delta,500,axis=-1))-compose11(self.a,self.p)
        expected=np.repeat(offsets11(torch.from_numpy(delta)[None])[0].numpy(),500,axis=-1)
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_objective_has_finite_gradient_and_known_offsets_are_learnable(self):
        self.mask[6]=False
        self.y[6]=np.nan
        ex=make_example(self.a,self.p,self.y,self.mask,self.scales)
        batch={k:torch.from_numpy(v)[None] for k,v in ex.items()}
        batch['weights']=torch.ones(1,11,dtype=torch.float64)
        model=MeanCorrector()
        pred=model(batch['features'])
        loss=learning_loss(batch,pred,torch.from_numpy(self.scales))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        # Independent functional probe: a visible, learnable local offset rule.
        model=MeanCorrector()
        x=torch.randn(4,9,500)
        target=.25*x[:,:7].reshape(4,7,10,50).mean(-1)+.2
        optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        initial=float(target.square().mean())
        for _ in range(120):
            model.train();optimizer.zero_grad()
            objective=(model(x)-target).square().mean()
            objective.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            final=float((model(x)-target).square().mean())
        self.assertLess(final,initial*.2,(initial,final))


if __name__=='__main__':
    unittest.main(verbosity=2)
