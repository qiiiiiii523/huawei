import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from baselines.B7.data import DATA_PROTOCOL,read_manifest
from baselines.B7.huawei10 import VIEW_PROTOCOL
from baselines.B7.huawei120 import full120_validation_manifest,HUAWEI_SELECTION_PROTOCOL
from baselines.B7.checkpoint import file_sha256
from baselines.B7.config import SamplingConfig
from baselines.B7.evaluate import evaluate,pearson

class Full120SelectionTests(unittest.TestCase):
    def fixture(self,root):
        within=np.sin(np.arange(5000)*2*np.pi/100).astype('float32')
        wave=np.concatenate([within+100*k for k in range(12)])
        y=np.stack([wave+j*.2 for j in range(12)])
        rows=[]
        for group in ('body','machine','task1'):
            path=root/(group+'.npz')
            np.savez(path,anchor_uV=y[:1],target_uV=y,lead_mask=np.ones(12,bool))
            rows.append(dict(record_id=group,subject_id=group,target_record_id=group,source='huawei',
                             group=group,split='validation',input_reliable=True,path=str(path)))
        source=root/'source.json';source.write_text(json.dumps(dict(protocol=DATA_PROTOCOL,sample_rate=500,scales_uV=[200.]*12,records=rows)))
        children=[]
        for parent in rows:
            for start in range(0,60000,5000):
                item=copy.deepcopy(parent)
                item.update(record_id=f"{parent['record_id']}:{start}",target_record_id=f"{parent['target_record_id']}:{start}",
                            parent_record_id=parent['record_id'],parent_target_record_id=parent['target_record_id'],
                            window_start=start,window_stop=start+5000,view_lead_mask=[True]*12)
                children.append(item)
        path=root/'views.json';path.write_text(json.dumps(dict(protocol=DATA_PROTOCOL,sample_rate=500,scales_uV=[100.]*12,
            records=children,huawei_window_protocol=VIEW_PROTOCOL,view_source_manifest=str(source),view_source_sha256=file_sha256(source))))
        return read_manifest(path),source,y

    def test_complete_record_precedes_Pearson_and_can_reverse_checkpoint_ranking(self):
        with tempfile.TemporaryDirectory() as td:
            views,source,y=self.fixture(Path(td));within=np.sin(np.arange(5000)*2*np.pi/100)
            wrong=np.stack([np.concatenate([within+100*(11-k) for k in range(12)])+j*.2 for j in range(12)]).astype('float32')
            good=y+np.random.default_rng(4).normal(scale=.2,size=y.shape).astype('float32')
            scores=[];calls=[]
            for prediction in (wrong,good):
                def predict(model,anchors,scales,keys,metas,config):
                    self.assertTrue(all(a.shape==(1,60000) for a in anchors));calls.extend(keys)
                    return torch.from_numpy(np.stack([prediction]*len(anchors)))
                with patch('baselines.B7.evaluate.predict_records',side_effect=predict):
                    metrics,records=evaluate(None,views,SamplingConfig(window_seconds=10),'huawei10')
                self.assertEqual(metrics['pairs'],3);self.assertEqual(metrics['targets'],3)
                self.assertEqual(metrics['scored_seconds'],120);self.assertEqual(metrics['prediction_window_seconds'],10)
                self.assertEqual(metrics['selection_protocol'],HUAWEI_SELECTION_PROTOCOL)
                self.assertEqual({r['record_id'] for r in records},{'body','machine','task1'})
                scores.append(metrics['r'])
            self.assertGreater(np.mean([pearson(wrong[1,k:k+5000],y[1,k:k+5000]) for k in range(0,60000,5000)]),.99999)
            self.assertLess(scores[0],-.99);self.assertGreater(scores[1],.9999)

    def test_uses_parent_QC_scales_and_original_record_keys(self):
        with tempfile.TemporaryDirectory() as td:
            views,_,_=self.fixture(Path(td));views['records'][0]['view_lead_mask'][6]=False
            full=full120_validation_manifest(views)
            self.assertEqual(len(full['records']),3);self.assertEqual(full['scales_uV'],[100.]*12)
            self.assertTrue(all('window_start' not in r for r in full['records']))
            self.assertNotIn('selection_protocol',views)

    def test_rejects_missing_duplicate_wrong_parent_path_and_changed_source(self):
        with tempfile.TemporaryDirectory() as td:
            views,source,_=self.fixture(Path(td))
            broken=copy.deepcopy(views);broken['records'].pop()
            with self.assertRaises(ValueError):full120_validation_manifest(broken)
            broken=copy.deepcopy(views);broken['records'][-1]=copy.deepcopy(broken['records'][-2])
            with self.assertRaises(ValueError):full120_validation_manifest(broken)
            broken=copy.deepcopy(views);broken['records'][0]['subject_id']='other'
            with self.assertRaises(ValueError):full120_validation_manifest(broken)
            broken=copy.deepcopy(views);broken['records'][0]['resolved_path']='other.npz'
            with self.assertRaises(ValueError):full120_validation_manifest(broken)
            source.write_text(source.read_text()+' ')
            with self.assertRaises(ValueError):full120_validation_manifest(views)

    def test_public_scoring_is_not_redirected_and_precision_flags_restore(self):
        with patch('baselines.B7.evaluate._evaluate_records',return_value=({'r':.1},[])) as inner:
            self.assertEqual(evaluate(None,{},SamplingConfig(),'public10')[0]['r'],.1)
            self.assertEqual(inner.call_args.args[-1],'public10')
        with tempfile.TemporaryDirectory() as td:
            views,_,_=self.fixture(Path(td));old=torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32=True
            def check(*args):
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertEqual(args[-1],'huawei40')
                return {'r':.2},[]
            try:
                with patch('baselines.B7.evaluate._evaluate_records',side_effect=check):evaluate(None,views,SamplingConfig(window_seconds=10),'huawei10')
                self.assertTrue(torch.backends.cudnn.allow_tf32)
            finally:torch.backends.cudnn.allow_tf32=old

if __name__=='__main__':unittest.main()
