from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from baselines.B5.audit_raw_voltage import compare_record, read_xml_independent, resample_like_cache, resolve_xml
from baselines.B5.diagnose_evaluation import analyze_task
from baselines.B5.diagnostic_signals import LEADS, boxcar_width, record_groups, record_metrics, slow_component


def fixture_xml(path: Path, points=20000, fs=1000):
    ns = 'urn:hl7-org:v3'
    tag = lambda name: '{'+ns+'}'+name
    root = ET.Element(tag('AnnotatedECG'))
    # Two candidate sequence sets: parser must select the longest complete one.
    for count in (20, points):
        seqset = ET.SubElement(root, tag('sequenceSet'))
        seq = ET.SubElement(ET.SubElement(seqset, tag('component')), tag('sequence'))
        ET.SubElement(seq, tag('code'), {'code':'TIME_ABSOLUTE'})
        ET.SubElement(ET.SubElement(seq, tag('value')),tag('increment'),{'value':str(1/fs),'unit':'s'})
        for lead in LEADS:
            seq = ET.SubElement(ET.SubElement(seqset, tag('component')), tag('sequence'))
            ET.SubElement(seq,tag('code'),{'code':'MDC_ECG_LEAD_'+lead.upper()})
            value = ET.SubElement(seq,tag('value'))
            ET.SubElement(value,tag('scale'),{'value':'0.025','unit':'mV'})
            ET.SubElement(value,tag('origin'),{'value':'-2.1','unit':'mV'})
            ET.SubElement(value,tag('digits')).text = ' '.join(str(i%101-50) for i in range(count))
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def evaluation_fixture(path, offset=500.):
    path.mkdir(parents=True)
    wave = np.sin(np.arange(5000)/20.) * 100
    target = np.broadcast_to(wave,(4,12,5000)).copy().astype(np.float32)
    np.save(path/'target_uV.npy',target)
    np.save(path/'prediction_uV.npy',target+offset)
    with (path/'window_metadata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['pair_id','target_record_id','subject_id','start_sample_500hz','expected_window_count'])
        writer.writeheader()
        for i in range(4):
            writer.writerow({'pair_id':f'p{i//2}','target_record_id':f'r{i//2}', 'subject_id':f's{i//2}',
                             'start_sample_500hz':(i%2)*5000,'expected_window_count':2})


class SignalTests(unittest.TestCase):
    def test_running_boxcar_matches_convolution(self):
        x=np.random.default_rng(42).normal(size=5000)+1000
        expected=np.convolve(np.pad(x,(250,250),mode='edge'),np.ones(501)/501,mode='valid')
        np.testing.assert_allclose(slow_component(x,501),expected,atol=1e-8,rtol=1e-11)
        self.assertEqual(boxcar_width(1),501)

    def test_offset_does_not_change_r(self):
        y=np.sin(np.arange(10000)/20.)*100
        m=record_metrics(y+500,y,501)
        self.assertAlmostEqual(m['raw_r'],1)
        self.assertAlmostEqual(m['raw_rmse_uV'],500)
        self.assertAlmostEqual(m['offset_mse_fraction'],1)
        self.assertAlmostEqual(m['fast_r'],1)

    def test_slow_error_and_cross_term(self):
        t=np.arange(10000)
        y=200*np.sin(t/18)+1000*np.sin(t/1500)
        p=200*np.sin(t/18)
        m=record_metrics(p,y,501)
        self.assertGreater(m['fast_r'],m['raw_r'])
        self.assertAlmostEqual(m['raw_mse'],m['slow_mse']+m['fast_mse']+m['slow_fast_error_cross_term'],places=6)

    def test_constant_correlation_is_undefined(self):
        m=record_metrics(np.ones(5000),np.ones(5000),501)
        self.assertIsNone(m['raw_r'])
        self.assertIsNone(m['fast_r'])
        json.dumps(m,allow_nan=False)

    def test_incomplete_record_rejected(self):
        with self.assertRaises(ValueError):
            record_groups([{'pair_id':'p','target_record_id':'r','start_sample_500hz':'5000'}])


class EvaluationTests(unittest.TestCase):
    def test_all_records_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'task1'; evaluation_fixture(folder)
            rows,leads,contract=analyze_task(folder,501)
            self.assertEqual(len(rows),22)
            self.assertEqual(len(leads),11)
            self.assertEqual(contract['records'],2)
            self.assertAlmostEqual(contract['raw_r_missing11'],1)
            self.assertAlmostEqual(contract['raw_missing11_mean_rmse_uV'],500,places=5)

    def test_matches_main_record_macro_and_point_weighted_rmse(self):
        from ecg12gen.evaluate import evaluate_record_predictions
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'task1'; evaluation_fixture(folder)
            target=np.load(folder/'target_uV.npy')
            prediction=target.copy(); prediction[0]+=500; prediction[1:]=-target[1:]+200
            metadata=[]
            for i in range(4):
                metadata.append({'pair_id':'short' if i==0 else 'long', 'target_record_id':'r0' if i==0 else 'r1',
                                 'subject_id':'s0' if i==0 else 's1', 'start_sample_500hz':str(0 if i==0 else (i-1)*5000),
                                 'expected_window_count':str(1 if i==0 else 3)})
            with (folder/'window_metadata.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(metadata[0])); writer.writeheader(); writer.writerows(metadata)
            np.save(folder/'prediction_uV.npy',prediction)
            official,_=evaluate_record_predictions(prediction,target,'task1',metadata)
            with (folder/'overall_metrics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(official)); writer.writeheader(); writer.writerow(official)
            _,_,contract=analyze_task(folder,501)
            self.assertTrue(contract['official_report_checked'])
            self.assertAlmostEqual(contract['raw_r_missing11'],0,places=6)
            self.assertAlmostEqual(contract['raw_missing11_mean_rmse_uV'],official['missing11_mean_rmse_uV'],places=5)

    def test_cli_and_target_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for label in ('A','B'):
                for task in ('task1','task2'):
                    evaluation_fixture(root/label/task)
            command=[sys.executable,'-m','baselines.B5.diagnose_evaluation','--evaluation',f'A={root/"A"}',
                     '--evaluation',f'B={root/"B"}','--output-dir',str(root/'report')]
            subprocess.run(command,check=True,capture_output=True,text=True)
            manifest=json.loads((root/'report/manifest.json').read_text())
            self.assertTrue(all(manifest['cross_model_target_and_metadata_match'].values()))
            self.assertFalse(manifest['training_started'])
            array=np.load(root/'B/task1/target_uV.npy'); array[0,8,10]+=1
            np.save(root/'B/task1/target_uV.npy',array)
            command[-1]=str(root/'mismatch')
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Different targets',result.stderr)


class RawVoltageTests(unittest.TestCase):
    def test_units_origin_sequence_and_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'signal.xml'; fixture_xml(path)
            raw,fs,headers=read_xml_independent(path)
            self.assertEqual(raw.shape,(12,20000))
            self.assertEqual(fs,1000)
            self.assertEqual(raw[0,0],-3350)
            self.assertEqual(headers[8]['origin_unit'],'mV')
            cached=resample_like_cache(raw,fs).reshape(12,2,5000).transpose(1,0,2).copy()
            refs=[{'array':cached,'array_index':i,'start':i*5000,'task':'task1','split':'validation'} for i in range(2)]
            result,_,_=compare_record('record',refs,path,.01,1e-5)
            self.assertTrue(result['independent_parser_matches_production'])
            self.assertTrue(result['cache_matches_reparsed_xml'])
            cached[0,8]-=500
            result,_,details=compare_record('record',refs,path,.01,1e-5)
            self.assertFalse(result['cache_matches_reparsed_xml'])
            self.assertTrue(any(not r['matches_reparsed_xml'] for r in details))

    def test_windows_manifest_and_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            expected=root/'心电图机d12/a.xml'
            self.assertEqual(resolve_xml(root,r'C:\old\Data\心电图机d12\a.xml'),expected)
            self.assertFalse(expected.exists())
            with self.assertRaises(ValueError):
                resolve_xml(root,'../outside.xml')

    def test_missing_xml_reports_incomplete_instead_of_pass(self):
        from baselines.B5 import audit_raw_voltage
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'metadata').mkdir()
            with (root/'metadata/raw_record_manifest.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['record_id','source_path']); writer.writeheader()
                writer.writerow({'record_id':'r','source_path':'Data/心电图机d12/missing.xml'})
            refs={('r','validation'):[{'task':'task1','split':'validation','array_index':0,'start':0,
                                       'subject_id':'s','array':np.ones((1,12,5000),dtype=np.float32)}]}
            config={'repository_root':str(root),'paths':{'data_root':str(root)}}
            argv=['audit','--config','unused.yaml','--output-dir',str(root/'out')]
            with patch.object(sys,'argv',argv), patch('baselines.B5.config.load_config',return_value=config), patch.object(audit_raw_voltage,'cached_records',return_value=refs):
                with self.assertRaises(SystemExit) as caught:
                    audit_raw_voltage.main()
            self.assertEqual(caught.exception.code,2)
            report=json.loads((root/'out/audit.json').read_text())
            self.assertFalse(report['checks_complete_and_matching'])
            self.assertEqual(report['records'][0]['status'],'missing_raw')
            self.assertTrue((root/'out/required_raw_files.csv').is_file())


if __name__ == '__main__':
    unittest.main()
