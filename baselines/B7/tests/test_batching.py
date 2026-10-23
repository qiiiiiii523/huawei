from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import unittest
import torch

from baselines.B7.config import SamplingConfig
from baselines.B7.model import B7UNet
from baselines.B7.inference import predict_record, predict_records
from baselines.B7.flow import sample
from baselines.B7.pipeline import plan
from baselines.B7.tests.test_b7 import small_config, RecordingVelocity


class BatchingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_K16_chunking_is_same_keyed_average(self):
        q=torch.randn(3,1,40)
        model=RecordingVelocity(.2).eval()
        original=SamplingConfig(samples=16,steps=2)
        a=sample(model,q,['a','b','c'],config=original)
        b=sample(model,q,['a','b','c'],config=replace(original,sample_batch_size=7))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_record_batch_and_noise_batch_match_sequential_real_network(self):
        torch.manual_seed(7)
        model=B7UNet(small_config()).eval()
        anchors=[torch.randn(1,5000)*100 for _ in range(2)]
        cfg=SamplingConfig(window_seconds=10,samples=3,steps=2)
        reference=torch.stack([predict_record(model,a,[100.]*12,k,config=cfg)
                               for a,k in zip(anchors,['a','b'])])
        batched=predict_records(model,anchors,[100.]*12,['a','b'],config=replace(cfg,sample_batch_size=4))
        torch.testing.assert_close(reference,batched,rtol=2e-5,atol=3e-4)
        torch.testing.assert_close(batched[:,:1],torch.stack(anchors),rtol=0,atol=0)

    def test_reuse_runs_no_prepare_and_uses_new_weights_directories(self):
        args=Namespace(output=Path('/new'),data_root=Path('/raw'),metadata=Path('/meta'),ptb_root=Path('/ptb'),
                       gpu_p='0',gpu_a='0',gpu_b='0',reuse_data_from=Path('/old'))
        jobs=plan(args)
        self.assertEqual([j.name for j in jobs],['P','A','Q','B'])
        self.assertEqual([j.dependencies for j in jobs],[(),('P',),('P',),('Q',)])
        self.assertTrue(all(j.module=='train' for j in jobs))
        self.assertIn('public10',jobs[0].arguments[jobs[0].arguments.index('--manifest')+1])


if __name__=='__main__':
    unittest.main()
