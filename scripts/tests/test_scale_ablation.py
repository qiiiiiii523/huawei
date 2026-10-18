from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.B5_control_pcc import GROUPS, BANDS_BY_GROUP
from baselines.B5_control_pcc.prepare import paired_training_config
from baselines.B5_control_pcc.common import multiscale_for_group
from baselines.B5_control_pcc.losses import multiscale_shape_loss, objective
from baselines.B5_control_pcc.core import generate
from baselines.B5_best.slow import predict_native
from baselines.B5_best import SAMPLING
import diagnose_b5_record_edges as edge
from test_control100_multiscale_diagnosis import fixture, diag


class ScaleAblationTests(unittest.TestCase):
    def test_seed_only_changes_paired_training_config(self):
        source={'seed':42,'epochs':20,'validate_every':5,'learning_rate':.0001}
        new=paired_training_config(source,20,43)
        self.assertEqual(new,{**source,'seed':43})
        self.assertEqual(source['seed'],42)
        self.assertEqual(SAMPLING['seed'],42)
        low=paired_training_config(source,20,43,2e-5)
        self.assertEqual(low,{**new,'learning_rate':2e-5})
        self.assertEqual(source['learning_rate'],.0001)
        for rate in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): paired_training_config(source,20,43,rate)
        for seed in (-1,2**32,True):
            with self.assertRaises(ValueError): paired_training_config(source,20,seed)

    def test_enabled_scales_equal_explicit_component_average_and_mask_gradients(self):
        torch.manual_seed(11)
        t=torch.arange(6000)/50
        y=(torch.sin(t*.2)+.4*torch.sin(t*1.6)+.1*torch.sin(t*8))[None,None].repeat(1,11,1)
        p=(y*.6+.15*torch.cos(t*.7)[None,None]).requires_grad_()
        mask=torch.ones(1,11,dtype=torch.bool); mask[:,8]=False
        cfg={'weight':.1,'shape_huber_weight':.1,'std_floor':[[1e-4]*11]*3}
        values={}
        for group in GROUPS:
            config=multiscale_for_group({'multiscale':cfg},group)
            self.assertEqual(config['enabled_bands'],list(BANDS_BY_GROUP[group]))
            self.assertEqual(config['weight'],.1 if BANDS_BY_GROUP[group] else 0.)
        for ids in ([0],[1],[2],[0,1],[0,1,2]):
            values[tuple(ids)]=multiscale_shape_loss(p,y,mask,{**cfg,'enabled_bands':ids})[0]
        torch.testing.assert_close(values[(0,1)],(values[(0,)]+values[(1,)])/2)
        torch.testing.assert_close(values[(0,1,2)],sum(values[(i,)] for i in range(3))/3)
        torch.testing.assert_close(values[(0,1,2)],multiscale_shape_loss(p,y,mask,cfg)[0],rtol=0,atol=0)
        values[(0,1,2)].backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(p.grad[:,8].abs().max()),0.)
        self.assertGreater(float(p.grad.abs().sum()),0.)
        with self.assertRaises(ValueError): multiscale_shape_loss(p,y,mask,{**cfg,'enabled_bands':[0,0]})
        # The B control really bypasses the extra term, while preserving base + combo objectives.
        batch={'mask':torch.cat((torch.ones(1,1,dtype=torch.bool),mask),1),'slow_target':y,
               'fm_uV':torch.zeros(1,12,60000),'target_uV':torch.zeros(1,12,60000)}
        base=objective(p.detach(),batch,torch.ones(12),2.589017,.1,beta=.75)[0]
        control=objective(p.detach(),batch,torch.ones(12),2.589017,.1,beta=.75,
                          multiscale=multiscale_for_group({'multiscale':cfg},GROUPS[0]))[0]
        torch.testing.assert_close(base,control,rtol=0,atol=0)

    def test_s50_cropping_padding_and_deployment_equivalence(self):
        class Spy(torch.nn.Module):
            def forward(self,anchor,**metadata):
                self.windows=anchor.detach().clone()
                return anchor.repeat(1,11,1)
        model=Spy().eval()
        anchor=torch.arange(6000,dtype=torch.float32)[None,None]
        inputs={'anchor':anchor,'numeric':torch.zeros(1,3),'sex':torch.zeros(1,dtype=torch.long),
                'field_mask':torch.ones(1,4),'age_topcoded':torch.zeros(1,1)}
        output=generate(model,inputs)
        torch.testing.assert_close(output,anchor.repeat(1,11,1),rtol=0,atol=0)
        self.assertTrue((model.windows[0,:,:1000]==0).all())
        self.assertTrue((model.windows[-1,:,1500:]==5999).all())
        torch.testing.assert_close(output,predict_native(model,inputs),rtol=0,atol=0)
        layout=edge.context_layout()
        self.assertEqual([r['left_repeat_s'] for r in layout[:3]],[20,10,0])
        self.assertEqual([r['right_repeat_s'] for r in layout[-3:]],[0,10,20])
        self.assertEqual(sum(r['real_output_samples'] for r in layout),6000)

    def test_regional_shape_error_ignores_only_constant_for_diagnosis(self):
        t=np.arange(60000)/500
        y=np.sin(t*.15)*100
        fast=np.sin(t*15)*50
        slow=edge.moving_average(y,501)
        result=edge.region_metrics(fast+slow+250,y,fast)
        for row in result:
            self.assertAlmostEqual(row['slow_rmse_uV'],250,places=8)
            self.assertLess(row['slow_centered_rmse_uV'],1e-9)
            self.assertAlmostEqual(row['slow_r'],1,places=7)

    def test_edge_end_to_end_source_immutability_and_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); args=fixture(root)
            view_path=root/'run/shared/cache_view.json'
            view=diag.read(view_path)
            for entry in view['validation']:
                path=root/'cache'/entry['file']
                with np.load(path) as archive:
                    values={k:archive[k].copy() for k in archive.files}
                values['anchor']=values['target_uV'][0].reshape(1,6000,10).mean(-1)
                np.savez(path,**values); entry['sha256']=diag.sha(path)
            diag.write(view_path,view)
            files=[p for p in root.rglob('*') if p.is_file()]
            before={str(p):diag.sha(p) for p in files}
            edge.run(SimpleNamespace(run=args.run,output=args.output))
            report=diag.read(root/'out/edge_audit.json')
            self.assertTrue(report['completed']); self.assertFalse(report['model_loaded'])
            rows=diag.csv_read(root/'out/target_lead_regions.csv')
            self.assertTrue(all(r['subject_id']!='badI' for r in rows))
            self.assertFalse(any(r['subject_id']=='good1' and r['lead']=='V4' for r in rows))
            self.assertTrue(any(r['n_pairs']=='2' for r in rows))
            self.assertEqual(before,{str(p):diag.sha(p) for p in files})


if __name__=='__main__': unittest.main()
