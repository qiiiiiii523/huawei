"""Frozen M7 parameter budget, public10 config alias and P-only default route."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from baselines.B7 import pretrain
from baselines.B7.config import ModelConfig
from baselines.B7.model import B7UNet
from baselines.B7.presets import M7_CHANNELS, m7_config, model_preset
from baselines.B7.train import validate_settings


class M7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_head_parameter_counts_and_budget(self):
        for head,count in [('linear',14502815),('multiscale',14552486),('split',14527519)]:
            model=B7UNet(m7_config(head))
            actual=sum(p.numel()for p in model.parameters())
            self.assertEqual(actual,count)
            self.assertGreaterEqual(actual,10000000)
            self.assertLessEqual(actual,20000000)

    def test_default_M7_and_compatibility_alias_match(self):
        root=Path(__file__).resolve().parents[3]
        self.assertEqual(pretrain.DEFAULT_CONFIG,root/'configs/u7_m7/b7_public10.json')
        canonical=json.loads(pretrain.DEFAULT_CONFIG.read_text())
        alias=json.loads((root/'configs/b7_pprime.json').read_text())
        self.assertEqual(canonical,alias)
        mc,lc,sc=validate_settings(canonical)
        self.assertEqual(mc,m7_config(mc.output_head))
        self.assertEqual(mc.channels,M7_CHANNELS)
        self.assertIn((mc.output_head,lc.low_frequency_weight),
                      [('linear',0.),('linear',1.),('multiscale',1.),('split',0.)])
        self.assertEqual((sc.sample_rate,sc.window_seconds,sc.core_seconds),(500,10,10))

    def test_training_and_sampling_match_unchanged_M2(self):
        root=Path(__file__).resolve().parents[3]
        m2=json.loads((root/'configs/u7_m2/b7_public10.json').read_text())
        m7=json.loads(pretrain.DEFAULT_CONFIG.read_text())
        self.assertEqual(m7['training'],m2['training'])
        self.assertEqual(m7['sampling'],m2['sampling'])
        self.assertEqual(m7['training']['updates'],20000)
        self.assertEqual(m7['training']['validation_mode'],'final_only')
        self.assertEqual(m7['training']['checkpoint_every'],500)
        self.assertEqual((m7['sampling']['steps'],m7['sampling']['samples']),(16,16))

    def test_preset_identification_excludes_old_forty_million_model(self):
        for head in ('linear','multiscale','split'):
            self.assertEqual(model_preset(m7_config(head)),'M7')
        self.assertEqual(model_preset(ModelConfig(channels=(112,224,448,448,448,448,448))),'custom')
        self.assertEqual(model_preset(replace(m7_config(),head_channels=128)),'custom')

    def test_cli_no_override_selects_M7(self):
        with patch('sys.argv',['pretrain','--manifest','shared_qcv2.json','--output','new_M7_P']), \
             patch.object(pretrain,'run')as run:
            pretrain.main()
        args=run.call_args.args[0]
        self.assertEqual(args.config,str(pretrain.DEFAULT_CONFIG))
        self.assertFalse(args.resume)
        self.assertFalse(hasattr(args,'init_checkpoint'))


if __name__=='__main__':unittest.main()
