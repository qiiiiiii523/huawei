"""Post-hoc rescoring on synthetic 120s records; optimizer updates prohibited."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5.tests.test_b5 import tiny_model,checkpoint,preprocessor
from baselines.B5.reassessment_cache import (SAMPLING,write,read,csv_write,csv_read,file_sha256,state_digest,
    checked_arrays,verified_prepared,trusted_cached_evaluation,evaluation_provenance)
from baselines.B5.reassess_qc import run,score_evaluation,references,slow_entries,generate
from ecg12gen.evaluate import QC_COMPETITION_SCORE_PROTOCOL,COMPETITION_SCORE_PROTOCOL


def arrays(task):
    rows=[];p=[];y=[];quality=[]
    devices=['watch_ecg'] if task=='task1' else ['ecg_machine_d6','body_scale_d6']
    for device in devices:
        for target in ('A','B'):
            for start in range(0,60000,5000):
                row={'pair_id':device+'_'+target,'target_record_id':target,'subject_id':target+'_subject',
                     'input_type':device,'start_sample_500hz':str(start)}
                wave=np.sin(np.arange(start,start+5000)/80).astype(np.float32)*100
                truth=wave[None].repeat(12,axis=0)+np.arange(12,dtype=np.float32)[:,None]*20
                pred=truth.copy()+10
                q=np.ones(12,bool)
                if target=='A': pred[9]=-truth[9]+10000;q[9]=False
                rows.append(row);p.append(pred);y.append(truth);quality.append(q)
    return np.stack(p),np.stack(y),rows,np.stack(quality)


def export(folder,task,values):
    p,y,rows,q=values;folder=Path(folder)/task;folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'prediction_uV.npy',p);np.save(folder/'target_uV.npy',y)
    np.save(folder/'target_quality_mask.npy',q);csv_write(folder/'window_metadata.csv',rows)


def current_reference(root,values):
    result={}
    for task,(p,y,rows,q) in values.items():
        folder=Path(root)/task;folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'target_uV.npy',y)
        result[task]={'target':np.load(folder/'target_uV.npy',mmap_mode='r'),'rows':rows,'quality':q}
    return result


class ReassessmentTests(unittest.TestCase):
    def test_same_predictions_full_and_qc_and_bad_lead_only_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);values={t:arrays(t) for t in ('task1','task2')}
            for task,v in values.items(): export(root/'old',task,v)
            ref=current_reference(root/'ref',values)
            before={t:file_sha256(root/'old'/t/'prediction_uV.npy') for t in values}
            scores,inputs=score_evaluation(root/'old',ref,root/'new')
            self.assertLess(scores['full_population']['task1_r_missing11'],scores['target_lead_qc']['task1_r_missing11'])
            self.assertAlmostEqual(scores['target_lead_qc']['task1_r_missing11'],1.)
            self.assertEqual(scores['full_population']['competition_score_protocol'],COMPETITION_SCORE_PROTOCOL)
            self.assertEqual(scores['target_lead_qc']['competition_score_protocol'],QC_COMPETITION_SCORE_PROTOCOL)
            details={r['lead']:r for r in csv_read(root/'new/target_lead_qc/task1/lead_metrics.csv')}
            self.assertEqual(int(details['V4']['n_unique_target_records']),1)
            self.assertEqual(int(details['V3']['n_unique_target_records']),2)
            for task in values:
                self.assertEqual(before[task],file_sha256(root/'old'/task/'prediction_uV.npy'))
                ref[task]['target']._mmap.close()

    def test_reordered_complete_windows_align_and_target_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);v=arrays('task1');p,y,rows,q=v
            ref=current_reference(root/'ref',{'task1':v})['task1']
            order=np.arange(len(p))[::-1]
            export(root/'old','task1',(p[order],y[order],[rows[i] for i in order],q[order]))
            a,b,old_rows,masks=checked_arrays(root/'old/task1',ref)
            np.testing.assert_array_equal(masks,q[order]);a._mmap.close();b._mmap.close()
            broken=y[order].copy();broken[0,0,0]+=1;np.save(root/'old/task1/target_uV.npy',broken)
            with self.assertRaisesRegex(ValueError,'Cached targets differ'): checked_arrays(root/'old/task1',ref)
            ref['target']._mmap.close()

    def test_folder_name_and_checkpoint_path_are_not_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);root.joinpath('best.pt').write_bytes(b'checkpoint')
            folder=root/'evaluation_best';folder.mkdir()
            write(folder/'inference_sampling.json',{'checkpoint':str(root/'best.pt'),'sampling':SAMPLING,'context_used':False})
            self.assertFalse(trusted_cached_evaluation(folder,root/'best.pt','modelA',None,None))
            for task in ('task1','task2'): export(folder,task,arrays(task))
            model=tiny_model();cp={'ema':model.state_dict()}
            write(folder/'inference_sampling.json',evaluation_provenance(folder,root/'best.pt',cp))
            self.assertTrue(trusted_cached_evaluation(folder,root/'best.pt',state_digest(cp['ema']),None,None))
            with (folder/'task1/prediction_uV.npy').open('ab') as stream: stream.write(b'changed')
            self.assertFalse(trusted_cached_evaluation(folder,root/'best.pt',state_digest(cp['ema']),None,None))

    def test_current_reference_reads_raw_targets_and_masks_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            values={t:arrays(t) for t in ('task1','task2')}
            class Dataset:
                def __init__(self,common,task,split):
                    self.p,self.y,self._rows,self.q=values[task];self._indices=list(range(len(self._rows)))
                def __len__(self): return len(self._rows)
                def __getitem__(self,i): return SimpleNamespace(Y_12lead=self.y[i],target_quality_mask=self.q[i])
            cp={'manifests':{t:hashlib.sha256(json.dumps(v[2],sort_keys=True).encode()).hexdigest() for t,v in values.items()}}
            with patch('baselines.B5.data.common_config',return_value=None),patch('ecg12gen.dataset.JointAnchorDataset',Dataset):
                refs=references({},cp,Path(tmp)/'ref')
            for task,ref in refs.items():
                np.testing.assert_array_equal(ref['target'],values[task][1]);np.testing.assert_array_equal(ref['quality'],values[task][3]);ref['target']._mmap.close()

    def test_generate_uses_existing_validator_ema_and_fixed_sampler_without_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=tiny_model();processor=preprocessor();cp=checkpoint(model,processor)
            cp['manifests']={'task1':'one','task2':'two'};path=root/'best.pt';torch.save(cp,path)
            config={'sampling':{'samples':1},'training':{'deterministic':True}}
            datasets={t:SimpleNamespace(manifest_digest=cp['manifests'][t]) for t in ('task1','task2')}
            def validator(net,ds,cfg,pc,device,output):
                self.assertIs(net,model);self.assertIs(pc,processor);self.assertEqual(cfg['sampling'],SAMPLING)
                for task in ds: export(output,task,arrays(task))
            with patch('baselines.B5.checkpoint.checkpoint_preprocessor',return_value=processor), \
                 patch('baselines.B5.data.preprocessing_config',return_value=None), \
                 patch('baselines.B5.checkpoint.model_from_checkpoint',return_value=model) as load_ema, \
                 patch('baselines.B5.runtime.build_validation_datasets',return_value=datasets), \
                 patch('baselines.B5.runtime.validate',side_effect=validator), \
                 patch('torch.optim.AdamW',side_effect=AssertionError('No optimizer may be constructed')):
                generate(config,path,cp,'cpu',root/'generated')
                load_ema.assert_called_once_with(cp,torch.device('cpu'))
            self.assertEqual(config['sampling'],{'samples':1})
            self.assertTrue(trusted_cached_evaluation(root/'generated',path,state_digest(cp['ema']),None,None))

    def test_four_candidates_hybrids_generation_dedup_resume_and_selection_no_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();out=root/'result';prepared=root/'prepared';prepared.mkdir()
            model=tiny_model();processor=preprocessor();config={'model':asdict(model.config),'repository_root':str(root),
                'training':{'deterministic':True},'sampling':SAMPLING}
            write(old/'resolved_config.json',config)
            demo=root/'demographics.csv';demo.write_text('synthetic',encoding='utf-8')
            source_files={key:root/(key+'.csv') for key in ('device_interpretation_qc_csv','subject_split_csv','task1_pair_manifest_csv','task2_pair_manifest_csv')}
            for path in source_files.values(): path.write_text('synthetic',encoding='utf-8')
            for task in ('task1','task2'):
                folder=root/task;folder.mkdir();source_files[task+'_output']=folder
                (folder/(task+'_window_metadata.csv')).write_text('synthetic',encoding='utf-8')
                np.save(folder/(task+'_validation_target.npy'),np.ones(1))
            manifests={'task1':'one','task2':'two','demographics_csv':file_sha256(demo)}
            cps={}
            for i,name in enumerate(('best','best_r','best_bonus','last')):
                cp=checkpoint(model,processor);cp.update(cache_protocol='independent-record-context-v1',manifests=manifests,epoch=40+i)
                cp['ema']={k:v.clone() for k,v in cp['ema'].items()}
                # best_bonus aliases best EMA but is a different checkpoint file.
                if i in (1,3): cp['ema']['output.bias']+=i*.01
                torch.save(cp,old/(name+'.pt'));cps[name]=cp
            source=old/'evaluation_best';source.mkdir()
            values={t:arrays(t) for t in ('task1','task2')}
            for task,v in values.items(): export(source,task,v)
            write(source/'inference_sampling.json',{'checkpoint':str(old/'best.pt'),'sampling':SAMPLING,'context_used':False})
            write(prepared/'records.json',{'synthetic':True})
            eval_files={task+'/'+name:file_sha256(source/task/name) for task in values for name in ('prediction_uV.npy','target_uV.npy','window_metadata.csv')}
            info={'protocol':'B5-slow-probe-v1','completed':True,'checkpoint_sha256':file_sha256(old/'best.pt'),
                'files':{'records.json':file_sha256(prepared/'records.json')},'evaluation_dir':str(source),'evaluation_files':eval_files}
            write(prepared/'prepared.json',info)
            slow_args=[]
            for label,group in (('S50','C'),('S50-F','F')):
                probe=root/(label+'.pt');probe.write_bytes(label.encode());folder=root/(label+'_evaluation');folder.mkdir()
                for task,v in values.items():
                    p,y,rows,q=v;export(folder/'learned_slow_hybrid',task,(p+2,y,rows,q))
                write(folder/'evaluation.json',{'protocol':'B5-slow-probe-v1','completed':True,'context_seconds':50,
                    'group':group,'target_used_at_inference':False,'B5_checkpoint_sha256':file_sha256(old/'best.pt'),
                    'prepared_sha256':file_sha256(prepared/'prepared.json'),'probe_checkpoint':str(probe),
                    'probe_checkpoint_sha256':file_sha256(probe),'epoch':20})
                slow_args.append(label+'='+str(folder))
            args=SimpleNamespace(b5_run=str(old),config=None,data_repo=None,prepared=str(prepared),slow_evaluation=slow_args,
                                 output_dir=str(out),generate_missing=True,device='cpu',resume=False)
            common=SimpleNamespace(path=lambda key:source_files[key])
            def build_ref(config,cp,folder): return current_reference(folder,values)
            def fake_generate(config,path,cp,device,folder):
                for task,v in values.items(): export(folder,task,v)
                write(folder/'inference_sampling.json',evaluation_provenance(folder,path,cp))
            with patch('baselines.B5.data.common_config',return_value=common),patch('baselines.B5.data.demographics_path',return_value=demo), \
                 patch('baselines.B5.reassess_qc.references',side_effect=build_ref), \
                 patch('baselines.B5.reassess_qc.generate',side_effect=fake_generate) as generator, \
                 patch('torch.optim.AdamW.step',side_effect=AssertionError('No training allowed')):
                args.generate_missing=False
                with self.assertRaisesRegex(FileNotFoundError,'generate-missing'): run(args)
                self.assertEqual(generator.call_count,0)
                args.resume=True;args.generate_missing=True
                run(args)
                self.assertEqual(generator.call_count,2)
                self.assertTrue(read(out/'completed.json')['completed'])
                rows=csv_read(out/'comparison.csv');self.assertEqual(len(rows),12)
                self.assertEqual(read(out/'selection.json')['scope'],'only_the_four_existing_candidates_not_all_historical_epochs')
                self.assertEqual(file_sha256(old/'best.pt'),info['checkpoint_sha256'])
                run(args);self.assertEqual(generator.call_count,2)
                source_files['device_interpretation_qc_csv'].write_text('changed_QC',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Resume changed'): run(args)

    def test_missing_predictions_do_not_silently_start_gpu_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);p=folder/'best.pt';p.write_bytes(b'old')
            with self.assertRaises(FileNotFoundError): verified_prepared(folder,p)
            with self.assertRaisesRegex(ValueError,'S50='): slow_entries(['wrong'],None,'x')


if __name__=='__main__': unittest.main()
