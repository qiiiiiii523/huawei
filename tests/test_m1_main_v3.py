"""Migration contract checks; no training loop or optimizer updates."""
from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from ecg12gen.m1_data import M1PreparedDataset, m1_collate
from ecg12gen.m1_axial import M1AxialLeadTimeModel
from ecg12gen.m1_axial_train import validate, validate_p0_checkpoint, _loss
from ecg12gen.m1_protocol import protocol_metadata, checkpoint_preprocessor
from ecg12gen.dataset import ECGDataConfig

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

class MigrationTest(unittest.TestCase):
    def pre(self):
        pc=PreprocessingConfig.from_yaml(ROOT/'configs/preprocessing.yaml')
        return ECGPreprocessor(pc,{'d12':np.full(12,100,dtype=np.float32),'ecg_machine_i':np.array([100],dtype=np.float32),'watch_ecg':np.array([50],dtype=np.float32)})

    def sample(self,start=0,complete=True):
        wave=np.sin(np.arange(5000,dtype=np.float32)/50)*100+400
        y=np.stack([wave+i*10 for i in range(12)])
        valid=np.ones((1,5000),dtype=bool)
        if not complete: valid[:,0]=False
        return SimpleNamespace(Y_12lead=y,context_ecg=(wave+1000)[None],context_record_baseline_uV=np.array([1300],dtype=np.float32),context_time_mask=valid,input_quality_mask=np.ones(1,dtype=bool),context_lead_mask=np.array([True]+[False]*11),target_quality_mask=np.ones(12,dtype=bool),context_source_type='watch_ecg',split='validation',supervision_mode='joint_anchor_adaptation',meta={'pair_id':'p','target_record_id':'r','start_sample_500hz':str(start),'expected_window_count':'2'},validate=lambda:None)

    def test_raw_target_and_record_context(self):
        s=self.sample(); ds=M1PreparedDataset([s],self.pre(),'P1_joint_anchor','watch_ecg'); item=ds[0]
        np.testing.assert_allclose(item.target_model.numpy()*100,s.Y_12lead,rtol=1e-6)
        np.testing.assert_allclose(item.context_model.numpy()*50,s.context_ecg-1300,rtol=1e-6)
        self.assertGreater(float(item.anchor_model.mean()),3)

    def test_missing_context_keeps_window(self):
        ds=M1PreparedDataset([self.sample(complete=False)],self.pre(),'P1_joint_anchor','watch_ecg')
        self.assertEqual(len(ds),1)
        item=ds[0]; self.assertFalse(item.context_available)
        self.assertEqual(float(item.context_model[0,0]),0)
        self.assertFalse(m1_collate([item])['context_available'][0])

    def test_full_record_validation_interface(self):
        ds=M1PreparedDataset([self.sample(0),self.sample(5000)],self.pre(),'P0_anchor_only')
        class Echo(torch.nn.Module):
            fusion_mode='none'; task_id='task1'
            def forward(self,anchor,**kwargs):
                return anchor.repeat(1,12,1)+torch.arange(12,device=anchor.device)[None,:,None]/10
        with tempfile.TemporaryDirectory() as tmp:
            score=validate(Echo(),ds,np.full(12,100,dtype=np.float32),torch.device('cpu'),Path(tmp))
            self.assertAlmostEqual(score['r_missing11'],1,places=6)
            self.assertEqual(score['n_records'],1)
            self.assertTrue((Path(tmp)/'window_metadata.csv').exists())
            self.assertFalse((Path(tmp)/'prediction_submit.npy').exists())
            with self.assertRaises(ValueError): validate(Echo(),ds,np.ones(12),torch.device('cpu'),Path(tmp),max_batches=1)

    def test_legacy_checkpoint_rejected_and_scale_guard(self):
        model=M1AxialLeadTimeModel(config={'d_model':16,'ffn_dim':32})
        with self.assertRaisesRegex(ValueError,'protocol'): validate_p0_checkpoint({},model,np.ones(12))
        ck={**model.architecture_metadata,**protocol_metadata(),'stage':'P0_anchor_only','target_d12_scale_uV':[100]*12}
        validate_p0_checkpoint(ck,model,np.full(12,100))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scales.npz'; self.pre().save(path)
            config=ECGDataConfig.from_yaml(ROOT/'configs/common.yaml')
            checkpoint_preprocessor(ck,config,path)
            ck['target_d12_scale_uV']=[101]*12
            with self.assertRaisesRegex(ValueError,'scales'): checkpoint_preprocessor(ck,config,path)
            ck['target_d12_scale_uV']=[100]*12
            ck.update(stage='P1_joint_anchor',context_source_type='watch_ecg',context_scale_uV=[50])
            checkpoint_preprocessor(ck,config,path)
            ck['context_scale_uV']=[51]
            with self.assertRaisesRegex(ValueError,'context scales'): checkpoint_preprocessor(ck,config,path)

    def test_disabled_context_cannot_change_prediction(self):
        torch.manual_seed(42)
        model=M1AxialLeadTimeModel(fusion_mode='film_gated_residual',config={'d_model':16,'ffn_dim':32}).eval()
        with torch.no_grad():
            model.film[-1].bias.fill_(0.2); model.residual[-1].bias.fill_(1)
            anchor=torch.randn(1,1,5000)
            y=model(anchor,context=torch.randn_like(anchor),context_source_type='watch_ecg',context_available=torch.tensor([False]))
            model.fusion_mode='none'; base=model(anchor)
        torch.testing.assert_close(y,base,rtol=0,atol=0)

    def test_cli_help_and_training_opt_in(self):
        for script in ('train_m1.py','train_m1_axial.py','validate_m1.py','prepare_m1.py','predict_m1_axial.py'):
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/train_m1_axial.py'),'--task-id','task1','--stage','P0_anchor_only','--output-dir','unused'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('Training was not started',result.stderr)

    def test_current_main_loss_backward_without_updates(self):
        model=M1AxialLeadTimeModel(config={'d_model':16,'ffn_dim':32})
        ds=M1PreparedDataset([self.sample()],self.pre(),'P0_anchor_only')
        batch=m1_collate([ds[0]])
        pred=model(batch['anchor_model'])
        loss=_loss(model,pred,batch,torch.full((12,),100.))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.anchor_projection.weight.grad)

    def test_target_free_batched_prediction_cli(self):
        model=M1AxialLeadTimeModel(config={'d_model':16,'ffn_dim':32}).eval()
        ck={**model.architecture_metadata,**protocol_metadata(),'stage':'P0_anchor_only','model':model.state_dict(),'target_d12_scale_uV':[100]*12}
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); torch.save(ck,run/'fixture.pt'); self.pre().save(run/'preprocessing_scales.npz')
            anchor=np.stack([self.sample().Y_12lead[:1]]*3); np.save(run/'anchor.npy',anchor)
            out=run/'prediction'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/predict_m1_axial.py'),'--checkpoint',str(run/'fixture.pt'),'--anchor-npy',str(run/'anchor.npy'),'--task-id','task2','--batch-size','2','--output-dir',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            pred=np.load(out/'prediction_raw.npy')
            self.assertEqual(pred.shape,(3,12,5000)); self.assertTrue(np.isfinite(pred).all())
            np.testing.assert_array_equal(pred,np.load(out/'prediction_submit.npy'))
            self.assertFalse(np.array_equal(pred[:,:1],anchor))

if __name__=='__main__': unittest.main()
