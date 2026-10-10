from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np
import torch

from baselines.B5.check_slow_pair import freeze_snapshot
from baselines.B5.compare_slow_pair import compare_runs
from baselines.B5.config import ModelConfig, load_config
from baselines.B5.diagnostic_signals import boundary_metrics
from baselines.B5.flow import linear_path, sample
from baselines.B5.losses import flow_loss
from baselines.B5.model import B5UNet
from baselines.B5.objective import assert_pair_configs, file_sha256, slow_settings, validate_resume_loss
from baselines.B5.runtime import initialize_validation_baseline
from baselines.B5.slow_trend import centered_slow_curve, masked_slow_trend_loss
from ecg12gen.losses import masked_huber_loss, masked_pcc_loss

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[3]


def legacy_weights():
    return {'huber':.1,'pcc':.1,'anchor':.02,'physiology':0.,'huber_delta':1.}


def tiny_inputs():
    return {'anchor':torch.randn(1,1,64),'numeric':torch.zeros(1,3),'sex':torch.zeros(1,dtype=torch.long),
            'field_mask':torch.ones(1,4,dtype=torch.bool),'age_topcoded':torch.zeros(1,1)}


class SlowLossTests(unittest.TestCase):
    def test_pool_matches_numpy_centered_definition_and_full_window(self):
        rng=np.random.default_rng(42)
        signal=rng.normal(size=(1,2,5000)).astype(np.float32)
        expected=[]
        for lead in signal[0].astype(np.float64):
            curve=np.convolve(np.pad(lead,(250,250),mode='edge'),np.ones(501)/501,mode='valid')
            expected.append(curve-curve.mean())
        actual=centered_slow_curve(torch.tensor(signal),501).numpy()[0]
        np.testing.assert_allclose(actual,np.stack(expected),atol=2e-6,rtol=1e-4)

    def test_constant_offset_invariance_and_no_input_mutation(self):
        x=torch.sin(torch.arange(128,dtype=torch.float32)/9).reshape(1,1,-1).repeat(1,11,1)
        y=x*.8
        before=x.clone(),y.clone()
        mask=torch.ones(1,11,dtype=torch.bool)
        first=masked_slow_trend_loss(x,y,mask,31)
        second=masked_slow_trend_loss(x+40,y-20,mask,31)
        self.assertAlmostEqual(first.item(),second.item(),places=5)
        self.assertTrue(torch.equal(x,before[0]) and torch.equal(y,before[1]))

    def test_bad_leads_nan_mask_and_gradient(self):
        p=torch.randn(2,11,128,requires_grad=True)
        y=torch.randn_like(p)
        mask=torch.ones(2,11,dtype=torch.bool); mask[:,7]=False
        y[:,7]=float('nan')
        loss=masked_slow_trend_loss(p,y,mask,31)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(p.grad[:,7].abs().sum().item(),0)
        self.assertEqual(p.grad[:,:5].abs().sum().item(),0)
        self.assertGreater(p.grad[:,5:7].abs().sum().item(),0)
        self.assertEqual(masked_slow_trend_loss(p,y,torch.zeros_like(mask),31).item(),0)

    def test_legacy_disabled_loss_and_rng_are_unchanged(self):
        torch.manual_seed(42)
        model=B5UNet(ModelConfig(channels=(8,16,32),time_dim=16,metadata_dim=16,metadata_dropout=0.)).eval()
        inputs=tiny_inputs(); target=torch.randn(1,12,64); mask=torch.ones(1,12,dtype=torch.bool)
        noise=torch.randn(1,11,64); time=torch.tensor([.3]); weights=legacy_weights()
        batch={**inputs,'target':target,'quality_mask':mask}
        state,velocity=linear_path(target[:,1:],mask[:,1:],noise,time)
        predicted,anchor=model(state,time,inputs['anchor'],inputs['numeric'],inputs['sex'],inputs['field_mask'],inputs['age_topcoded'])
        full=torch.cat((anchor,state+(1-time[:,None,None])*predicted),dim=1)
        old=((predicted-velocity)**2).mean()+weights['huber']*masked_huber_loss(full,target,mask,1.)+weights['pcc']*masked_pcc_loss(full,target,mask)+weights['anchor']*masked_huber_loss(anchor,inputs['anchor'],mask[:,:1],1.)
        rng=torch.get_rng_state().clone()
        actual,parts=flow_loss(model,batch,weights,torch.ones(12),noise,time)
        self.assertTrue(torch.allclose(actual,old,atol=1e-7,rtol=1e-6))
        self.assertNotIn('slow_trend',parts)  # Legacy logging contract is unchanged.
        disabled,explicit_parts=flow_loss(model,batch,{**weights,'slow_trend':0},torch.ones(12),noise,time)
        self.assertTrue(torch.equal(disabled,actual))
        self.assertEqual(explicit_parts['slow_trend'].item(),0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        active={**weights,'slow_trend':.1,'slow_trend_seconds':.04}
        enhanced,parts=flow_loss(model,batch,active,torch.ones(12),noise,time)
        self.assertTrue(torch.allclose(enhanced,actual+.1*parts['slow_trend']))
        enhanced.backward()  # Only derivative verification; no optimizer is created or updated.

    def test_target_free_sampling_unchanged_by_auxiliary_calculation(self):
        torch.manual_seed(3)
        model=B5UNet(ModelConfig(channels=(8,16,32),time_dim=16,metadata_dim=16,metadata_dropout=0.)).eval()
        inputs=tiny_inputs()
        before=sample(model,inputs,['r:0'],42,2,'heun',1)
        batch={**inputs,'target':torch.randn(1,12,64),'quality_mask':torch.ones(1,12,dtype=torch.bool)}
        flow_loss(model,batch,{**legacy_weights(),'slow_trend':.1,'slow_trend_seconds':.04},torch.ones(12))
        after=sample(model,inputs,['r:0'],42,2,'heun',1)
        self.assertTrue(torch.equal(before,after))


class PairTests(unittest.TestCase):
    def configs(self):
        return load_config(ROOT/'configs/experiments/b5_slow_control.yaml'),load_config(ROOT/'configs/experiments/b5_slow_trend.yaml')

    def test_pair_has_only_loss_weight_difference(self):
        control,slow=self.configs(); assert_pair_configs(control,slow)
        self.assertEqual(control['training']['epochs'],20)
        self.assertEqual(slow_settings(slow['loss'])['width'],501)
        changed=copy.deepcopy(slow); changed['training']['learning_rate']/=2
        with self.assertRaises(ValueError): assert_pair_configs(control,changed)

    def test_invalid_settings_and_resume_compatibility(self):
        for value in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): slow_settings({'slow_trend':value})
        with self.assertRaises(ValueError): slow_settings({'slow_trend_seconds':10})
        old={'config':{'loss':legacy_weights()}}
        validate_resume_loss(old,{**legacy_weights(),'slow_trend':0})
        with self.assertRaises(ValueError): validate_resume_loss(old,{**legacy_weights(),'slow_trend':.1})

    def test_snapshot_reuses_identical_and_rejects_different(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); source=p/'source.pt'; source.write_bytes(b'unchanged weights')
            destination=p/'snapshot.pt'; digest=file_sha256(source)
            freeze_snapshot(source,destination,digest); freeze_snapshot(source,destination,digest)
            source.write_bytes(b'different weights')
            with self.assertRaises(FileExistsError): freeze_snapshot(source,destination,file_sha256(source))

    def test_initializer_baseline_is_saved_without_updates(self):
        from baselines.B5 import runtime
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            values={'task1':{'r_missing11':.57},'task2':{'r_missing11':.56}}
            with patch.object(runtime,'validate',return_value=values):
                score=initialize_validation_baseline(None,None,None,None,None,output,'task1',
                    lambda score:{'epoch':-1,'updates':0,'best_score':score})
            checkpoint=torch.load(output/'best.pt',weights_only=True)
            self.assertEqual(checkpoint['epoch'],-1)
            self.assertEqual(checkpoint['updates'],0)
            self.assertEqual(score,.57)

    def test_compare_uses_epoch0_when_training_deteriorates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); configs=self.configs()
            for i,name in enumerate(('control','slow')):
                directory=root/name; directory.mkdir()
                (directory/'resolved_config.json').write_text(json.dumps(configs[i]))
                provenance={'initialization':{'source_checkpoint_sha256':'same'},'scales_sha256':'same',
                            'manifests':{'train':'same'},'selection_task':'task1','architecture_hash':'same'}
                (directory/'initialization.json').write_text(json.dumps(provenance))
                values={'task1':{'r_missing11':.57,'missing11_mean_rmse_uV':400},
                        'task2':{'r_missing11':.56,'missing11_mean_rmse_uV':400,'task2_missing_lead_mean_rmse_uV':700}}
                (directory/'initial_validation.json').write_text(json.dumps({'epoch':0,'validation':values}))
                history=copy.deepcopy(values); history['task1']['r_missing11']=.55 if i==0 else .58
                (directory/'history.jsonl').write_text(json.dumps({'epoch':20,'validation':history})+'\n')
            a,b=compare_runs(root/'control',root/'slow')
            self.assertEqual(a['metrics']['best_epoch'],0)
            self.assertEqual(b['metrics']['best_epoch'],20)
            provenance=json.loads((root/'slow/initialization.json').read_text()); provenance['initialization']['source_checkpoint_sha256']='different'
            (root/'slow/initialization.json').write_text(json.dumps(provenance))
            with self.assertRaises(ValueError): compare_runs(root/'control',root/'slow')

    def test_training_cli_still_requires_explicit_opt_in(self):
        result=subprocess.run([sys.executable,'-m','baselines.B5.train_huawei','--config',
            'configs/experiments/b5_slow_trend.yaml'],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Training was not started',result.stderr)

    def test_preflight_main_freezes_snapshot_without_model_inference(self):
        from baselines.B5 import check_slow_pair, checkpoint, data, runtime
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); control,slow=self.configs()
            for config in (control,slow):
                config['repository_root']=str(root)
                config['paths']['demographics']=str(root/'demographics.csv')
            for name in ('demographics.csv','subject_split_csv','device_interpretation_qc_csv'):
                (root/name).write_text('fixture\n')
            processor=SimpleNamespace(scale_uV_by_source={'d12':np.ones(12,dtype=np.float32)})
            dataset=MagicMock(); dataset.manifest_digest='dataset'; dataset.__len__.return_value=1
            manifests={'train':'dataset','task1':'dataset','task2':'dataset',
                       'demographics_csv':file_sha256(root/'demographics.csv'),
                       'subject_split_csv':file_sha256(root/'subject_split_csv'),
                       'device_interpretation_qc_csv':file_sha256(root/'device_interpretation_qc_csv')}
            payload={'architecture_hash':ModelConfig.from_dict(control['model']).fingerprint,
                     'scales_sha256':checkpoint.scales_digest(checkpoint.scales_payload(processor)),
                     'stage':'finetune','epoch':14,'manifests':manifests}
            source=root/'source.pt'; source.write_bytes(b'snapshot fixture')
            snapshot=root/'shared/init.pt'; report=root/'preflight.json'
            argv=['check','--init-checkpoint',str(source),'--snapshot',str(snapshot),'--report',str(report)]
            with patch.object(sys,'argv',argv), patch.object(check_slow_pair,'load_config',side_effect=[control,slow]), \
                 patch.object(checkpoint,'load_checkpoint',return_value=payload), \
                 patch.object(data,'load_preprocessor',return_value=processor), \
                 patch.object(data,'common_config',return_value=SimpleNamespace(path=lambda key:root/key)), \
                 patch.object(runtime,'build_datasets',return_value=(dataset,{'task1':dataset,'task2':dataset})), \
                 patch.object(runtime,'B5UNet',side_effect=AssertionError('Preflight must not run a model')):
                check_slow_pair.main()
            result=json.loads(report.read_text())
            self.assertTrue(result['ready'])
            self.assertFalse(result['training_started'])
            self.assertFalse(result['model_inference_started'])
            self.assertEqual(result['source_completed_epoch'],15)
            self.assertEqual(snapshot.read_bytes(),source.read_bytes())

    def test_comparison_cli_produces_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); configs=self.configs()
            for i,name in enumerate(('control','slow')):
                directory=root/name; directory.mkdir()
                (directory/'resolved_config.json').write_text(json.dumps(configs[i]))
                provenance={'initialization':{'source_checkpoint_sha256':'same'},'scales_sha256':'same',
                            'manifests':{'train':'same'},'selection_task':'task1','architecture_hash':'same'}
                (directory/'initialization.json').write_text(json.dumps(provenance))
                values={'task1':{'r_missing11':.57,'missing11_mean_rmse_uV':400},
                        'task2':{'r_missing11':.56,'missing11_mean_rmse_uV':400,'task2_missing_lead_mean_rmse_uV':700}}
                (directory/'initial_validation.json').write_text(json.dumps({'epoch':0,'validation':values}))
                (directory/'history.jsonl').write_text(json.dumps({'epoch':20,'validation':values})+'\n')
            result=subprocess.run([sys.executable,'-m','baselines.B5.compare_slow_pair','--control-run',str(root/'control'),
                '--slow-run',str(root/'slow'),'--output-dir',str(root/'comparison')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'comparison/comparison.json').read_text())
            self.assertTrue(report['runs_finished_20_epochs'])
            self.assertEqual(report['slow_minus_control_task1_r'],0)


class BoundaryTests(unittest.TestCase):
    def test_detects_window_offset_jump_not_natural_target_changes(self):
        y=np.sin(np.arange(10000)/20)*100
        p=y.copy(); p[5000:]+=300
        metrics=boundary_metrics(p,y)
        self.assertEqual(metrics['boundary_count'],1)
        self.assertAlmostEqual(metrics['point_jump_error_mae_uV'],300)
        self.assertAlmostEqual(metrics['window_mean_change_error_rmse_uV'],300)
        self.assertEqual(boundary_metrics(y+200,y)['point_jump_error_mae_uV'],0)
        self.assertIsNone(boundary_metrics(y[:5000],y[:5000])['point_jump_error_mae_uV'])


if __name__=='__main__':
    unittest.main()
