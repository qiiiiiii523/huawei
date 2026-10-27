from dataclasses import replace
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from baselines.B7.config import SamplingConfig
from baselines.B7.data import WindowDataset
from baselines.B7.flow import sample
from baselines.B7.inference import predict_records
from baselines.B7.model import B7UNet
from baselines.B7.position import inference_offsets, validate_position_views, view_metadata
from baselines.B7.huawei10 import VIEW_PROTOCOL
from baselines.B7.train import initialize_stage, validate_settings, validation_due
from baselines.B7.tests.test_b7 import small_config
from baselines.B7.tests.test_b7 import payload
from baselines.B7.checkpoint import save_checkpoint, load_checkpoint
from baselines.B7.data import DATA_PROTOCOL
from baselines.B7 import train as trainer


class C5Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_C5_only_changes_position_from_C0(self):
        root=Path(__file__).resolve().parents[3]
        c0=json.loads((root/'configs/round1/C0/A.json').read_text())
        c5=json.loads((root/'configs/round1/C5/A.json').read_text())
        mc,_,sc=validate_settings(c5)
        self.assertTrue(c5['model'].pop('record_position'))
        self.assertEqual(c0,c5)
        self.assertEqual(sc.window_seconds,10)
        self.assertEqual([i for i in range(1,4001) if validation_due(c5['training'],i)],
                         [1000,2000,3000,4000])
        self.assertEqual(sum(p.numel() for p in B7UNet(mc).parameters()),14503071)

    def test_strict_P_upgrade_zero_output_RNG_and_position_gradient(self):
        torch.manual_seed(42)
        old=B7UNet(small_config())
        rng=torch.get_rng_state()
        torch.manual_seed(42)
        new=B7UNet(replace(small_config(),record_position=True))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        source=dict(model_config=old.config.to_dict(),ema=old.state_dict(),
                    scales_uV=[100.]*12,stage='public10')
        self.assertTrue(initialize_stage(new,source,[100.]*12,'huawei10'))
        x,a,t=torch.randn(2,7,128),torch.randn(2,1,128),torch.rand(2)
        pos=torch.linspace(-1,1,128)[None,None].expand(2,1,128)
        y=new(x,t,a,dict(record_position=pos))
        torch.testing.assert_close(y,old(x,t,a),rtol=0,atol=0)
        y.square().mean().backward()
        self.assertGreater(new.position_projection.weight.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):
            initialize_stage(new,{**source,'stage':'huawei10'},[100.]*12,'huawei10')
        broken=dict(source['ema']);del broken['stem.bias']
        with self.assertRaises(ValueError):
            initialize_stage(new,{**source,'ema':broken},[100.]*12,'huawei10')

    def test_window_view_keeps_original_110_second_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'parent.npz'
            np.savez(path,anchor_uV=np.ones((1,60000),dtype='float32'),
                     target_uV=np.ones((12,60000),dtype='float32'),lead_mask=np.ones(12))
            row=dict(split='train',input_reliable=True,resolved_path=str(path),
                     window_start=55000,window_stop=60000,parent_record_id='p')
            m=dict(scales_uV=[100.]*12,records=[row],huawei_window_protocol=VIEW_PROTOCOL)
            validate_position_views(m,'huawei10')
            batch=WindowDataset(m,10,0)[0]
            p=batch['metadata']['record_position'][0]
            self.assertAlmostEqual(p[0].item(),55000/30000-1,places=6)
            self.assertAlmostEqual(p[-1].item(),59999/30000-1,places=6)
            self.assertEqual(batch['metadata']['record_offset'].item(),55000)
            with self.assertRaises(ValueError):
                validate_position_views({**m,'records':[{**row,'window_start':None}]},'huawei10')

    def test_inference_offset_bounds_and_missing_position(self):
        for m in [None,{},dict(record_offset=55001),dict(record_offset=-5000),
                  dict(record_offset=60000),dict(record_offset=0.)]:
            with self.assertRaises(ValueError):
                inference_offsets([m],5000,'cpu')
        self.assertEqual(inference_offsets([None],60000,'cpu').item(),0)

    def test_record_batch_uses_distinct_offsets_and_full_record_uses_continuous_position(self):
        class Model(torch.nn.Module):
            config=replace(small_config(),record_position=True)
        def fake(model,anchor,keys,metadata,config):
            return metadata['record_position'].expand(-1,7,-1)
        model=Model().eval();cfg=SamplingConfig(window_seconds=10,samples=1,steps=1)
        anchors=[torch.randn(1,5000),torch.randn(1,5000)]
        m=[view_metadata({},dict(window_start=k)) for k in [0,55000]]
        with patch('baselines.B7.inference.sample',side_effect=fake):
            out=predict_records(model,anchors,[1.]*12,['a','b'],m,cfg)
            full=predict_records(model,[torch.zeros(1,60000)],[1.]*12,['p'],config=cfg)
        torch.testing.assert_close(out[0,1],torch.arange(5000)/30000-1,rtol=0,atol=0)
        torch.testing.assert_close(out[1,1],(torch.arange(5000)+55000)/30000-1,rtol=0,atol=0)
        torch.testing.assert_close(full[0,1],torch.arange(60000)/30000-1,rtol=0,atol=0)
        torch.testing.assert_close(out[:,:1],torch.stack(anchors),rtol=0,atol=0)
        torch.testing.assert_close(out[:,2],out[:,1]-out[:,0])

    def test_ODE_batched_K_samples_preserve_position_metadata(self):
        class Velocity(torch.nn.Module):
            def forward(self,state,time,anchor,metadata):
                return metadata['record_position'].expand_as(state)
        model=Velocity().eval();a=torch.zeros(2,1,128)
        meta=dict(record_position=torch.stack([torch.full((1,128),-.5),torch.full((1,128),.8)]))
        cfg=SamplingConfig(samples=3,steps=2)
        x=sample(model,a,['a','b'],meta,cfg)
        y=sample(model,a,['a','b'],meta,replace(cfg,sample_batch_size=4))
        torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_actual_huawei10_position_training_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/'P.pt';save_checkpoint(p,payload(B7UNet(small_config())))
            cfg=json.loads((Path(__file__).resolve().parents[3]/'configs/round1/C5/A.json').read_text())
            cfg['model']=replace(small_config(),record_position=True).to_dict()
            cfg['training'].update(updates=2,validate_every=2,batch_size=1,checkpoint_every=1,
                                   initial_validation=False,early_stop_patience=0)
            config=root/'config.json';config.write_text(json.dumps(cfg))
            data=root/'parent.npz';y=np.random.default_rng(42).normal(size=(12,60000)).astype('float32')*100
            np.savez(data,anchor_uV=y[:1],target_uV=y,lead_mask=np.ones(12))
            rows=[dict(record_id=s,subject_id=s,target_record_id=s,source='huawei',group='task1',
                       split=s,input_reliable=True,path=str(data),window_start=k,window_stop=k+5000,
                       parent_record_id=s) for s,k in [('train',5000),('validation',55000)]]
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(protocol=DATA_PROTOCOL,sample_rate=500,
                                                scales_uV=[100.]*12,records=rows,
                                                huawei_window_protocol=VIEW_PROTOCOL)))
            args=Namespace(config=str(config),manifest=str(manifest),output=str(root/'A'),device='cpu',
                           init_checkpoint=str(p),resume=False)
            with patch.object(trainer,'evaluate',return_value=({'r':.5},[])):
                trainer.train(args)
                args.init_checkpoint=None;args.resume=True;trainer.train(args)
            ck=load_checkpoint(root/'A/last.pt')
            self.assertEqual(ck['update'],2)
            self.assertTrue(ck['initialization']['zero_initialized_position_upgrade'])
            self.assertGreater(ck['model']['position_projection.weight'].abs().sum().item(),0)
            self.assertTrue((root/'A/checkpoint_0000001.pt').exists())


if __name__=='__main__':
    unittest.main()
