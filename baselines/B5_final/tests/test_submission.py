"""Organizer input/output integration; synthetic weights-free checks, no training."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from competition_interface import input as official_input,output as official_output
from baselines.B5_final.official_data import OfficialDataset,BODY_COLUMNS,xml_rate,resample_valid
from baselines.B5_final.submission import run,load_settings,executable_root
from baselines.B5_final.official_score import evaluator
from baselines.B5_final.predict import Predictor
from baselines.B5_best.common import read,write
from baselines.B5_final.profile import BASE_SHA,BODY_SHA
from baselines.B5_final.package import stage

ROOT=Path(__file__).resolve().parents[3]
CONFIG=load_settings(ROOT/'configs/b5_submission.json')


def xml(path,leads,fs=500,points=60000,annotation=''):
    path.parent.mkdir(parents=True,exist_ok=True)
    seq=[]
    if fs is not None:
        seq.append(f'<component><sequence><code code="TIME_ABSOLUTE"/><value><increment value="{1/fs}" unit="s"/></value></sequence></component>')
    for j,name in enumerate(leads):
        code=name.upper() if name.startswith('a') else name
        samples=(' '.join(str((i%100)-50+j*3) for i in range(points)))
        seq.append(f'<component><sequence><code code="MDC_ECG_LEAD_{code}"/><value><origin value="1" unit="mV"/>'
                   f'<scale value="0.5" unit="uV"/><digits>{samples}</digits></value></sequence></component>')
    path.write_text('<AnnotatedECG xmlns="urn:hl7-org:v3"><component><series><code code="RHYTHM"/>'
        '<component><sequenceSet>'+''.join(seq)+'</sequenceSet></component>'+annotation+'</series></component></AnnotatedECG>',encoding='utf-8')


def fixture(root,*,histories=True,anchor_fs=500,anchor_points=60000):
    root=Path(root);root.mkdir()
    (root/'userinfo.txt').write_text('externalid,height,weight,age,gender\nHNU00001,170,65,30,男\n',encoding='utf-8')
    xml(root/'Electrocardiograph/d12_test/HNU00001.xml',('I',),anchor_fs,anchor_points)
    if histories:
        xml(root/'Electrocardiograph/d6_test/HNU00001.xml',official_output.LEADS[:6],1000,6000)
        scale=root/'scale';scale.mkdir()
        with (scale/'SCALE.csv').open('w',encoding='utf-8',newline='') as stream:
            stream.write('被试ID:HNU00001\n采样率:500\n数据单位:mV\n')
            writer=csv.writer(stream);writer.writerow(['Index',*map(str,range(1,13))])
            for i in range(2000):
                # Preserve a genuine gap at positions 1000..1009.
                writer.writerow([i+17+(10 if i>=1000 else 0),*[j*.1+np.sin(i/20)*.01 for j in range(1,13)]])
    return root


class SubmissionTests(unittest.TestCase):
    def test_official_sources_preserved_and_example_is_inspection_only(self):
        folder=ROOT/'competition_interface';manifest=read(folder/'source_manifest.json')
        for name,digest in manifest['files'].items(): self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest)
        dataset=OfficialDataset(folder/'test_data',CONFIG)
        result=dataset.inspect();self.assertEqual(result['subjects'][0]['d12_I_lengths'],[4])
        self.assertFalse(result['model_forward_checked'])
        person=next(iter(dataset))
        with self.assertRaisesRegex(ValueError,'No usable 120s'): dataset.anchor(person)
        # Source unit conversion, not a second deployment-side conversion.
        self.assertEqual(person['ele']['d12'][0]['leads']['I'],[1.,2.,3.,4.])

    def test_xml_actual_rate_and_unit_then_exact_120s(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=fixture(Path(tmp)/'data',histories=False,anchor_fs=1000,anchor_points=120000)
            dataset=OfficialDataset(root,CONFIG);person=next(iter(dataset));anchor,info=dataset.anchor(person)
            self.assertEqual(anchor.shape,(1,60000));self.assertEqual(info['input_fs'],1000)
            self.assertEqual(xml_rate(root/'Electrocardiograph/d12_test/HNU00001.xml'),1000)
            # Source is near 1000uV (origin1mV), never accidentally scaled again.
            self.assertGreater(float(np.median(anchor)),900);self.assertLess(float(np.median(anchor)),1100)
            self.assertEqual(person['userinfo']['age'],30.)

    def test_scale_mapping_gap_mask_and_no_premature_detrending(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset=OfficialDataset(fixture(Path(tmp)/'data'),CONFIG);person=next(iter(dataset))
            history,info=dataset.history(person,'body_scale_d6')
            self.assertEqual(history.raw_uV.shape,(6,2010));self.assertFalse(history.valid_mask[:,1000:1010].any())
            self.assertEqual(info['scale_columns'],list(BODY_COLUMNS))
            np.testing.assert_allclose(history.raw_uV[:,0],[100,200,900,1000,1100,1200],atol=1e-4)
            self.assertGreater(float(np.median(history.raw_uV[0,history.valid_mask[0]])),90)
            machine,info=dataset.history(person,'ecg_machine_d6')
            self.assertEqual(machine.raw_uV.shape,(6,3000));self.assertEqual(info['input_fs'],1000)

    def test_missing_history_falls_back_but_missing_anchor_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=fixture(Path(tmp)/'data',histories=False)
            dataset=OfficialDataset(root,CONFIG);person=next(iter(dataset))
            self.assertIsNone(dataset.history(person,'body_scale_d6')[0])
            self.assertIsNone(dataset.history(person,'ecg_machine_d6')[0])
            person['ele']['d12']=[]
            with self.assertRaisesRegex(ValueError,'No usable'): dataset.anchor(person)

    def test_missing_rate_requires_explicit_fallback_and_short_anchor_never_padded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=fixture(Path(tmp)/'data',histories=False,anchor_fs=None)
            dataset=OfficialDataset(root,CONFIG);person=next(iter(dataset))
            with self.assertRaisesRegex(ValueError,'sampling rate metadata'): dataset.anchor(person)
            dataset=OfficialDataset(root,{**CONFIG,'d12_fs_fallback':500})
            self.assertTrue(dataset.anchor(person)[1]['fs_fallback_used'])
            person['ele']['d12'][0]['leads']['I']=person['ele']['d12'][0]['leads']['I'][:100]
            with self.assertRaisesRegex(ValueError,'padding is forbidden'): dataset.anchor(person)

    def test_machine_explicit_bad_lead_masks_history_not_chest_dropout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=fixture(Path(tmp)/'data',histories=False)
            annotation='<annotation><code code="MDC_ECG_INTERPRETATION_STATEMENT"/><value>信号质量差（II,V1,V2）</value></annotation>'
            xml(root/'Electrocardiograph/d6_test/HNU00001.xml',official_output.LEADS[:6],500,2000,annotation)
            dataset=OfficialDataset(root,CONFIG);person=next(iter(dataset))
            self.assertIsNone(dataset.history(person,'ecg_machine_d6')[0])
            xml(root/'Electrocardiograph/d6_test/HNU00001.xml',official_output.LEADS[:6],500,2000,
                annotation.replace('II,V1,V2','V1,V2'))
            history,_=dataset.history(next(iter(dataset)),'ecg_machine_d6')
            self.assertTrue(history.valid_mask.all())

    def test_resampling_does_not_bridge_invalid_gaps(self):
        raw=np.ones((6,10000),np.float32)*100;mask=np.ones(raw.shape,bool);mask[:,4000:5000]=False
        y,q=resample_valid(raw,mask,1000)
        self.assertEqual(y.shape,(6,5000));self.assertFalse(q[:,2000:2500].any())
        self.assertTrue(np.isfinite(y).all());self.assertTrue(q[:,:2000].all())

    def test_three_tasks_share_base_and_apply_original_router(self):
        from baselines.B5_best_context.adapters import ContextRouter
        class Base:
            count=0
            def predict(self,anchor,metadata,key,batch):
                self.count+=1
                return np.repeat(anchor,12,axis=0)*np.arange(1,13,dtype=np.float32)[:,None]
        base=Base();router=ContextRouter(machine_mode='fir25')
        engine=object.__new__(Predictor);engine.engine=SimpleNamespace(base=base,router=router)
        anchor=np.sin(np.arange(60000,dtype=np.float32)/60)[None]*100
        results,status=engine.predict_tasks(anchor,{},'id')
        self.assertEqual(base.count,1)
        original=np.repeat(anchor,12,axis=0)*np.arange(1,13,dtype=np.float32)[:,None]
        for name,task,source in (('gen1','task1','watch_ecg'),('gen6_scale','task2','body_scale_d6'),('gen6_d6','task2','ecg_machine_d6')):
            expected,_=router.apply(original,anchor,task_id=task,input_type=source)
            np.testing.assert_array_equal(results[name],expected)
        self.assertEqual(status['gen6_d6']['fallback'],'projected_base')

    def test_full_official_csv_readback_and_reference_score_no_gold_used_for_prediction(self):
        class FakePredictor:
            calls=[]
            def predict_tasks(self,anchor,metadata,key,**kwargs):
                self.calls.append((anchor.copy(),metadata,key,kwargs))
                base=anchor.repeat(12,axis=0)
                return {task:base.copy() for task in official_output.TASKS},{task:{'synthetic':True} for task in official_output.TASKS}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=fixture(root/'test_data');engine=FakePredictor();out=root/'results'
            report=run(data,out,CONFIG,engine=engine)
            self.assertEqual(len(report),1);self.assertEqual(len(engine.calls),1)
            self.assertNotIn('target',engine.calls[0][3]);self.assertNotIn('gold',engine.calls[0][3])
            with (out/'HNU00001.csv').open() as stream:
                self.assertEqual(len(next(csv.reader(stream))),39)
            prediction=official_output.read_result(out/'HNU00001.csv')
            self.assertEqual(len(prediction['gen6_scale']['leads']['V6']),60000)
            gold=root/'gold';gold.mkdir()
            with (gold/'HNU00001.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['userid',*official_output.LEADS])
                for value in engine.calls[0][0][0]: writer.writerow(['HNU00001',*[float(value)]*12])
            score=evaluator().evaluate_directory(out,gold)
            self.assertAlmostEqual(score['summary']['score'],1.)
            self.assertAlmostEqual(score['summary']['gen6']['mean_rmse_uv'],0.)
            self.assertTrue(read(out/'_audit/completed.json')['official_output_readback_checked'])
            with self.assertRaises(FileExistsError): run(data,out,CONFIG,engine=engine)
            with self.assertRaisesRegex(ValueError,'test_data'): run(data,data/'results',CONFIG,engine=engine)

    def test_executable_paths_ignore_current_working_directory(self):
        with patch.object(sys,'frozen',True,create=True),patch.object(sys,'executable',r'C:\competition\B5Submit.exe'):
            self.assertEqual(executable_root(),Path(r'C:\competition'))

    def test_packaging_copies_selected_bytes_and_requires_windows_for_exe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'selected.pt';body=root/'body.pt';base.write_bytes(b'base');body.write_bytes(b'body')
            assets=SimpleNamespace(base=base,body=body)
            def digest(path):
                name=Path(path).name
                return BASE_SHA if name in ('selected.pt','base.pt') else BODY_SHA if name=='body.pt' else hashlib.sha256(Path(path).read_bytes()).hexdigest()
            with patch('baselines.B5_final.package.resolve_assets',return_value=assets),patch('baselines.B5_final.package.sha',side_effect=digest):
                manifest=stage(root/'release',base,body)
            self.assertFalse(manifest['windows_exe_built']);self.assertFalse(manifest['checks']['real_inference_checked'])
            self.assertTrue((root/'release_package.zip').is_file())
            self.assertFalse((root/'release/metadata').exists());self.assertFalse((root/'release/competition_interface/test_data').exists())
            self.assertEqual((root/'release/models/base.pt').read_bytes(),b'base')


if __name__=='__main__': unittest.main()
