"""Synthetic CPU tests, with no optimizer steps or real ECG training."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from baselines.B5.checkpoint import atomic_save
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best.common import read,write,sha,state_sha
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.slow import predict_native
from baselines.B5_public_slow import PROTOCOL as P_PROTOCOL
from baselines.B5_public_slow.tests.test_public_slow import study_fixture,synthetic_record
from baselines.B5_public_slow.common import model_from_bundle,SourceSampler
from baselines.B5_public_slow.core import generate,batch_record,FIELDS,objective as original_objective
from baselines.B5_public_slow.evaluate import evaluate as p_evaluate
from baselines.B5_p_all import PROTOCOL
from baselines.B5_p_all.prepare import prepare
from baselines.B5_p_all.common import load_study,load_checkpoint,objective,group_settings
from baselines.B5_p_all.evaluate import evaluate
from baselines.B5_p_all.export import export
from baselines.B5_p_all.compare import compare
from baselines.B5_p_all.multiscale import scale_bands,multiscale_shape_loss


def source_fixture(root):
    root.mkdir()
    path,bundle=study_fixture(root)
    old=read(path);cache=Path(old['huawei']['root']);manifest=read(old['huawei']['manifest'])
    # One flagged V4, one fully good target and one bad-I target in both tasks/devices.
    a=synthetic_record();a['mask'][9]=False
    np.savez(cache/'validation.npz',**a)
    for e in manifest['validation']:e['sha256']=sha(cache/'validation.npz')
    for key,patient,badI in [('good2','Hval2',False),('badI','Hbad',True)]:
        a=synthetic_record();a['mask'][0]=not badI;np.savez(cache/(key+'.npz'),**a)
        for task,rows in manifest['validation_rows'].items():
            devices=['ecg_machine_d6'] if task=='task1' else ['ecg_machine_d6','body_scale_d6']
            for device in devices:
                pair=task+'_'+device+'_'+key
                manifest['validation'].append({'cache_key':task+':'+pair,'pair_id':pair,
                    'file':key+'.npz','sha256':sha(cache/(key+'.npz')),'subject_id':patient,'target_record_id':key})
                rows.extend({**rows[0],'pair_id':pair,'target_record_id':key,'subject_id':patient,'input_type':device,
                             'start_sample_500hz':str(s)} for s in range(0,60000,5000))
    write(old['huawei']['manifest'],manifest)
    old['huawei']['sha256']=sha(old['huawei']['manifest'])
    old['training'].update(stage1_updates=0,stage2_updates=780,seed=42)
    write(path,old)
    model=model_from_bundle(bundle,torch.device('cpu')).eval()
    with torch.no_grad():model.output.bias.add_(.15)
    score=p_evaluate(model,old,bundle,torch.device('cpu'),root/'reference')
    selected=copy.deepcopy(bundle)
    selected['slow']['model']={k:v.detach().clone() for k,v in model.state_dict().items()}
    selected['slow_state_sha256']=state_sha(selected['slow']['model'])
    selected['reference_qc_r']={k:score[k] for k in ('task1_r_missing11','task2_r_missing11')}
    selected['reference_score_protocol']=SCORE_PROTOCOL
    cp={'protocol':P_PROTOCOL,'group':'P','study_sha256':sha(path),'training_seed':42,
        'stage':2,'step':390,'validation_score':score,'model':model.state_dict(),'FM_state_sha256':bundle['fm_state_sha256']}
    atomic_save(root/'P/best.pt',cp);atomic_save(root/'P/model.pt',selected)
    write(root/'P/model.json',{'group':'P','stage':2,'step':390,'training_seed':42,'score':score,
          'bundle_sha256':sha(root/'P/model.pt'),'checkpoint_sha256':sha(root/'P/best.pt')})
    write(root/'P/completed.json',{'completed':True,'group':'P','best':{'stage':2,'step':390,'score':score}})
    return root


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_control_exactly_preserves_P_objective_and_ALL_masked_gradients(self):
        a=synthetic_record();a['mask'][9]=False;batch=batch_record(a,torch.device('cpu'))
        p=(batch['slow_target']*.8+.02).detach().requires_grad_()
        cfg={'weight':.1,'enabled_bands':[0,1,2],'std_floor':np.full((3,11),.001).tolist(),'shape_huber_weight':.1}
        study={'residual_weight':2.5890171538829843,'combo_weight':.1,'multiscale':cfg}
        scales=torch.ones(12)*100
        base,_=original_objective(p,batch,scales,study['residual_weight'],study['combo_weight'])
        control,_=objective(p,batch,scales,study,'control')
        torch.testing.assert_close(control,base,rtol=0,atol=0)
        all_loss,parts=objective(p,batch,scales,study,'all')
        extra,_=multiscale_shape_loss(p,batch['slow_target'],batch['mask'][:,1:],cfg)
        torch.testing.assert_close(all_loss,base+.1*extra,rtol=0,atol=0)
        self.assertGreater(float(parts['weighted_multiscale_loss'].detach()),0.)
        all_loss.backward();self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue((p.grad[:,8]==0).all()) # V4 is index8 among missing11.
        torch.testing.assert_close(sum(scale_bands(p.detach())),p.detach(),rtol=1e-5,atol=1e-6)
        self.assertFalse(batch['fm_uV'].requires_grad)

    def test_prepare_initial_score_export_QC_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp, patch('torch.optim.AdamW.step',side_effect=AssertionError('No training')):
            root=Path(tmp);source=source_fixture(root/'source')
            files=[p for p in source.rglob('*') if p.is_file()];before={str(p):sha(p) for p in files}
            args=SimpleNamespace(source_run=str(source),output=str(root/'run/shared'),updates=390,validate_every=78,training_seed=43)
            report=prepare(args)
            self.assertTrue(report['ready']);self.assertEqual(report['source_step'],390)
            self.assertEqual(report['training']['learning_rate'],1e-4)
            self.assertEqual(report['multiscale_weights'],{'control':0.,'all':.1})
            self.assertEqual(report['public_training_updates'],0)
            self.assertEqual(report['validation_populations']['task2']['body_scale_d6']['eligible_pairs'],2)
            study_path=root/'run/shared/study.json';s,bundle,manifest=load_study(study_path,deep=True)
            self.assertNotEqual(bundle['slow_state_sha256'],load_bundle(source/'shared/components.pt')['slow_state_sha256'])
            digests=[]
            for group in ('control','all'):
                model=model_from_bundle(bundle,torch.device('cpu'));digests.append(state_sha(model.state_dict()))
                folder=root/'run'/group/'eval'
                score=evaluate(model,s,bundle,torch.device('cpu'),folder,group,0,initial=True)
                self.assertTrue(read(folder/'evaluation.json')['reference_matched'])
                cp={'protocol':PROTOCOL,'study_sha256':sha(study_path),'group':group,
                    'FM_state_sha256':bundle['fm_state_sha256'],'multiscale':group_settings(s,group),
                    'sampling_config':s['sampling'],'score_protocol':SCORE_PROTOCOL,
                    'step':0,'model':model.state_dict(),'validation_score':score}
                cp_path=root/'run'/group/'best.pt';atomic_save(cp_path,cp)
                load_checkpoint(cp_path,study_path,torch.device('cpu'))
                result=export(study_path,cp_path,cp_path.parent/'model.pt')
                self.assertEqual(result['fm_state_sha256'],bundle['fm_state_sha256'])
                # Portable inference loads only the full bundle, not study/cache or targets.
                loaded=load_bundle(cp_path.parent/'model.pt')
                deployed=model_from_bundle(loaded,torch.device('cpu')).eval()
                inputs=batch_record(synthetic_record(),torch.device('cpu'),labels=False)
                self.assertEqual(set(inputs),set(FIELDS))
                with torch.no_grad():torch.testing.assert_close(generate(model.eval(),inputs),predict_native(deployed,inputs),rtol=0,atol=0)
                write(cp_path.parent/'completed.json',{'completed':True})
                for name in ('best','last'):(cp_path.parent/f'latest_{name}_evaluation.txt').write_text(str(folder))
            self.assertEqual(digests[0],digests[1])
            self.assertEqual(len(compare(root/'run')),5)
            self.assertEqual(before,{str(p):sha(p) for p in files})
            p=source/'source/cache/train.npz'
            p.write_bytes(p.read_bytes()+b'modified')
            with self.assertRaises(ValueError):load_study(study_path,deep=True)

    def test_sampler_pair_and_exact_restore_without_optimizer_updates(self):
        entries=[{'file':str(i),'subject_id':str(i)} for i in range(8)]
        a,b=SourceSampler(entries,243),SourceSampler(entries,243)
        for _ in range(9):self.assertEqual(a.draw(),b.draw())
        saved=a.state();expected=[a.draw() for _ in range(5)]
        restored=SourceSampler(entries,999);restored.restore(saved)
        self.assertEqual(expected,[restored.draw() for _ in range(5)])

    def test_training_requires_execute_flag(self):
        from baselines.B5_p_all.train import main
        with patch('sys.argv',['train','--study','unused','--group','all','--output','unused']),patch('baselines.B5_p_all.train.run') as run:
            with self.assertRaises(SystemExit) as e:main()
            self.assertEqual(e.exception.code,2);run.assert_not_called()


if __name__=='__main__':unittest.main()
