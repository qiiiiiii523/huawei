from __future__ import annotations

import copy
from contextlib import ExitStack
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5 import CACHE_PROTOCOL, EVALUATION_PROTOCOL
from baselines.B5.config import load_config
from baselines.B5.data import HuaweiValidationDataset, common_config, demographics_path
from baselines.B5.metadata import encode_demographics
from baselines.B5.runtime import validate, validate_resume_protocol
from baselines.B5.tests.test_b5 import preprocessor, tiny_model

ROOT = Path(__file__).resolve().parents[3]


class MainV2Tests(unittest.TestCase):
    def test_official_predict_requires_real_120_seconds_and_crops_long_inputs(self):
        from baselines.B5 import predict
        for length in (59999, 60000, 65000):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root=Path(tmp); path=root/'anchor.npy'; output=root/'output.npy'
                np.save(path,np.arange(length,dtype=np.float32))
                cp={'condition_schema':'fixture'}
                stack.enter_context(patch.object(sys,'argv',['predict','--checkpoint',str(root/'cp.pt'),
                    '--anchor',str(path),'--record-id','visible','--device','cpu','--output',str(output)]))
                stack.enter_context(patch('baselines.B5.checkpoint.load_checkpoint',return_value=cp))
                stack.enter_context(patch('baselines.B5.checkpoint.model_from_checkpoint',return_value=object()))
                stack.enter_context(patch('baselines.B5.checkpoint.checkpoint_preprocessor',return_value=object()))
                stack.enter_context(patch('baselines.B5.runtime.seed_all'))
                stack.enter_context(patch('sys.stdout',new=io.StringIO()))
                generate=stack.enter_context(patch.object(predict,'predict_record',return_value=np.zeros((12,60000),np.float32)))
                if length<60000:
                    with self.assertRaisesRegex(ValueError,'real 120-second'):predict.main()
                    generate.assert_not_called()
                else:
                    predict.main()
                    self.assertEqual(generate.call_args.args[2].shape,(1,60000))
                    self.assertEqual(np.load(output).shape,(12,60000))
                    sidecar=json.loads(output.with_suffix('.json').read_text())
                    self.assertEqual(sidecar['target_record_seconds'],120)
                    self.assertFalse(sidecar['context_used'])

    def test_cache_paths_and_legacy_resume_guard(self):
        config = load_config(ROOT/'configs/experiments/b5_finetune_meta.yaml')
        common = common_config(config)
        self.assertEqual(common.path('task1_output').name, 'task1_record_context_v3')
        self.assertEqual(common.path('task2_output').name, 'task2_record_context_v3')
        self.assertNotIn('E2_finetune_meta', config['paths']['output_dir'])
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            validate_resume_protocol({'config':config}, config)
        cp = {'cache_protocol':CACHE_PROTOCOL,'evaluation_protocol':EVALUATION_PROTOCOL,
              'sampling_config':copy.deepcopy(config['sampling'])}
        validate_resume_protocol(cp,config)
        cp['sampling_config']['samples'] += 1
        with self.assertRaisesRegex(ValueError,'sampling'):
            validate_resume_protocol(cp,config)

    def test_validation_adapter_keeps_device_and_raw_labels_without_context_inputs(self):
        config=load_config(ROOT/'configs/experiments/b5_finetune_meta.yaml')
        raw=np.tile(np.arange(5000,dtype=np.float32),(12,1))+950
        sample=SimpleNamespace(anchor_i_ecg=raw[:1],Y_12lead=raw,subject_id='patient',target_record_id='target')
        row={'pair_id':'pair','target_record_id':'target','subject_id':'patient','start_sample_500hz':'0',
             'expected_window_count':'12','input_type':'body_scale_d6'}
        class Base:
            _rows=[row]; _indices=[0]
            def __len__(self):return 1
            def __getitem__(self,i):return sample
        table=SimpleNamespace(get=lambda subject:encode_demographics(None))
        with patch('baselines.B5.data.JointAnchorDataset',return_value=Base()):
            dataset=HuaweiValidationDataset(config,'task2',preprocessor(),table)
            item=dataset[0]
        np.testing.assert_array_equal(item['target_uV'],raw)
        self.assertEqual(item['evaluation_metadata']['input_type'],'body_scale_d6')
        self.assertFalse(item['evaluation_metadata']['context_used'])
        self.assertEqual(item['evaluation_metadata']['expected_window_count'],'12')
        self.assertNotIn('context',item)
        np.testing.assert_allclose(item['anchor']*preprocessor().scale_uV_by_source['ecg_machine_i'][:,None],raw[:1],atol=.001)

    def test_demographics_primary_and_raw_upload_fallback(self):
        config=load_config(ROOT/'configs/experiments/b5_finetune_meta.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config['paths']['demographics']=str(root/'utf8.csv')
            config['paths']['data_root']=str(root)
            with self.assertRaises(FileNotFoundError):demographics_path(config)
            raw=root/'userinfobean.csv';raw.write_text('externalid,age\nHNU1,20\n')
            self.assertEqual(demographics_path(config),raw)
            primary=root/'utf8.csv';primary.write_text(raw.read_text())
            self.assertEqual(demographics_path(config),primary)

    def test_runtime_full_records_and_separate_device_bonus(self):
        processor=preprocessor(); model=tiny_model()
        settings={'solver':'euler','steps':1,'samples':1,'seed':42}
        config={'sampling':settings,'training':{'workers':0},'validation':{'batch_size':8}}
        wave=np.sin(np.arange(5000,dtype=np.float32)*.01)*100
        predictions={}
        class Dataset:
            def __init__(self,devices):
                self.rows=[]
                for pair,device in enumerate(devices):
                    for window in range(12):
                        key=f'{pair}:{device}:{window}'
                        raw=np.tile(wave,(12,1))
                        pred=raw.copy()
                        if device in ['ecg_machine_d6','body_scale_d6']:
                            pred[6:]+=70 if device=='ecg_machine_d6' else 140
                        predictions[key]=torch.from_numpy(pred/processor.scale_uV_by_source['d12'][:,None])
                        self.rows.append({'anchor':raw[:1]/processor.scale_uV_by_source['ecg_machine_i'][:,None],
                            'target_uV':raw,**encode_demographics(None),'key':key,
                            'evaluation_metadata':{'pair_id':f'{device}:{pair}','target_record_id':f'target:{pair}',
                                'start_sample_500hz':window*5000,'expected_window_count':12,'input_type':device}})
            def __len__(self):return len(self.rows)
            def __getitem__(self,i):return self.rows[i]
        datasets={'task1':Dataset(['watch_ecg']),
                  'task2':Dataset(['ecg_machine_d6','ecg_machine_d6','body_scale_d6'])}
        def generated(model,inputs,keys,*unused):return torch.stack([predictions[k] for k in keys])
        with tempfile.TemporaryDirectory() as tmp, patch('baselines.B5.runtime.sample',side_effect=generated):
            summary=validate(model,datasets,config,processor,torch.device('cpu'),Path(tmp))
            import csv
            with (Path(tmp)/'competition_score.csv').open() as h:score=next(csv.DictReader(h))
            self.assertAlmostEqual(float(score['task2_rmse_bonus_score']),7.5,places=5)
            self.assertAlmostEqual(float(score['competition_total_score']),8.5,places=5)
        self.assertEqual(summary['task2']['ecg_machine_d6_n_records'],2)
        self.assertEqual(summary['task2']['body_scale_d6_n_records'],1)
        self.assertEqual(summary['task2']['task2_device_aggregation'],EVALUATION_PROTOCOL)
        self.assertFalse(summary['task2']['context_used'])


if __name__=='__main__':unittest.main()
