"""CPU synthetic functions only; no ECG training or optimizer updates."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .fixtures import Fixture,CONFIG,LOSS,inputs,fake_sample
from baselines.B5.checkpoint import load_checkpoint
from baselines.B5.flow import sample
from baselines.B5.losses import flow_loss
from baselines.B5_scratch import PROTOCOL,SCORE_PROTOCOL,FULL_SCORE_PROTOCOL,INPUTS
from baselines.B5_scratch.prepare import validate_public,public_initializer,load_shared
from baselines.B5_scratch.model import make_model,BoundModel
from baselines.B5_scratch.common import read,write,sha,state_hash,code_hashes
from baselines.B5_scratch.runtime import datasets,validate,update_best,CANDIDATES,run_config
from baselines.B5_scratch.check import check
from baselines.B5_scratch.checkpoint import load
from baselines.B5_scratch.evaluate import evaluate as final_evaluate
from baselines.B5_scratch.scoring import score_tasks
from baselines.B5_scratch.hybrid import evaluate_hybrid
from baselines.B5_scratch.compare import compare
from baselines.B5_scratch.continuation import archive_boundary

torch.set_num_threads(2)


class EarlyTests(unittest.TestCase):
    def test_prepare_binds_public_weights_not_Huawei_and_freezes_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp)
            shared,info=f.prepare()
            self.assertEqual(info['fm_sha256'],sha(f.public))
            self.assertEqual(info['data_reference_sha256'],sha(f.fm))
            self.assertNotEqual(info['fm_initial_ema_sha256'],state_hash(load_checkpoint(f.fm)['ema']))
            self.assertEqual(info['fm_initial_ema_sha256'],state_hash(load_checkpoint(f.public)['ema']))
            self.assertEqual(info['training']['epochs'],100)
            self.assertEqual(info['initialization_mode'],'public_pretrained_early_slow')
            self.assertEqual(info['public_source_stage'],'public')
            self.assertEqual(info['optimizer_updates'],0)
            self.assertNotIn('competition_total_score',info['public_initial_score'])
            for key in ['slow','slow_f']:self.assertTrue(info['slow_sources'][key]['frozen'])
            report=Path(tmp)/'preflight.json'
            check(SimpleNamespace(shared=str(shared),report=str(report)))
            r=read(report)
            self.assertEqual(r['initializer_stage'],'public')
            self.assertTrue(r['experimental_adapter_same_initialization'])

    def test_public_initializer_rejects_Huawei_architecture_and_scale_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);ref=load_checkpoint(f.fm);pub=load_checkpoint(f.public)
            info=read(f.prepared/'prepared.json')
            validate_public(pub,ref,info)
            with self.assertRaisesRegex(ValueError,'public-pretrained'):validate_public(ref,ref,info)
            bad=copy.deepcopy(pub);bad['scales_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'scales'):validate_public(bad,ref,info)
            bad=copy.deepcopy(pub);bad['architecture_hash']='changed'
            with self.assertRaisesRegex(ValueError,'architecture'):validate_public(bad,ref,info)
            with self.assertRaises(FileNotFoundError):public_initializer(None,ref,info)
            ref['initialization']={'source_checkpoint':str(f.public),'source_checkpoint_sha256':sha(f.public)}
            path,cp=public_initializer(None,ref,info)
            self.assertEqual(path,f.public);self.assertEqual(cp['stage'],'public')
            ref['initialization']['source_checkpoint_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'changed'):public_initializer(None,ref,info)

    def test_zero_adapter_preserves_public_ODE_and_gets_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);state=load_checkpoint(f.public)['ema']
            a=make_model(CONFIG,'control',state).eval();b=make_model(CONFIG,'slow',state).eval()
            data=inputs();slow=torch.randn(2,11,64)
            with torch.no_grad():
                original=sample(BoundModel(a),data,['a','b'],steps=2,samples=2)
                injected=sample(BoundModel(b,slow),data,['a','b'],steps=2,samples=2)
            self.assertTrue(torch.equal(original,injected))
            before=state_hash(b.state_dict());b.train()
            batch={**data,'target':torch.randn(2,12,64),'quality_mask':torch.ones(2,12,dtype=torch.bool)}
            flow_loss(BoundModel(b,slow),batch,LOSS,torch.ones(12))[0].backward()
            self.assertGreater(b.slow_adapter[-1].weight.grad.norm().item(),0)
            self.assertEqual(before,state_hash(b.state_dict()))

    def test_datasets_audit_Huawei_reference_while_training_initializes_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);shared,info=f.prepare()
            with patch('baselines.B5_scratch.runtime.source_datasets',return_value=(None,f.train,f.val)) as source:
                train,val,_=datasets(shared,info,'slow')
            self.assertEqual(source.call_args.args[1]['stage'],'finetune')
            self.assertIn('slow',train[0])
            self.assertEqual(val['task2'][0]['slow'].shape,(11,5000))
            c=run_config('control',shared,info);s=run_config('slow',shared,info)
            for key in ['fm_sha256','training','sampling','loss','model_config','initialization_mode']:
                self.assertEqual(c[key],s[key])
            train.native._mmap.close();val['task1'].native._mmap.close()

    def test_r_only_selector_ignores_retired_bonus_and_retains_ties(self):
        self.assertEqual(CANDIDATES,{'r':'best_r.pt'})
        best={'r':{'value':-float('inf'),'epoch':None}}
        changed=update_best(best,{'score_protocol':SCORE_PROTOCOL,'main_correlation':.7},0)
        self.assertEqual(changed,['r'])
        changed=update_best(best,{'score_protocol':SCORE_PROTOCOL,'main_correlation':.69,'competition_total_score':99},5)
        self.assertEqual(changed,[]);self.assertEqual(best['r']['epoch'],0)
        with self.assertRaises(ValueError):update_best(best,{'competition_score_protocol':'old','main_correlation':.8},10)

    def test_old_S50_checkpoint_and_cache_cannot_resume_here(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);shared,info=f.prepare()
            old=Path(tmp)/'old.pt';torch.save({'protocol':'B5-S50-three-arm-QC-v2','score_protocol':SCORE_PROTOCOL},old)
            with self.assertRaisesRegex(ValueError,'checkpoint'):load(old)
            changed=copy.deepcopy(info);changed['protocol']='B5-S50-three-arm-QC-v2';write(shared/'shared.json',changed)
            with self.assertRaisesRegex(ValueError,'B5-scratch'):load_shared(shared)

    def test_full_size_final_eval_and_fixed_S50F_hybrid_score_by_same_r(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);shared,info=f.prepare()
            state=load_checkpoint(f.public)['ema'];model=make_model(CONFIG,'slow',state).eval()
            with patch('baselines.B5_scratch.runtime.source_datasets',return_value=(None,f.train,f.val)):
                train,val,_=datasets(shared,info,'slow')
            out=Path(tmp)/'evaluation'
            with patch('baselines.B5_scratch.runtime.sample',side_effect=fake_sample):
                result=validate(model,val,shared,info,torch.device('cpu'),out)
            hybrid=evaluate_hybrid(out,shared,info)
            self.assertEqual(result['score']['score_protocol'],SCORE_PROTOCOL)
            self.assertEqual(result['full_score']['score_protocol'],FULL_SCORE_PROTOCOL)
            self.assertEqual(hybrid['score_protocol'],SCORE_PROTOCOL)
            self.assertEqual(np.load(out/'hybrid_slow_f/task2/prediction_uV.npy').shape,(24,12,5000))
            self.assertNotIn('competition_total_score',hybrid)
            self.assertFalse((out/'competition_score.csv').exists())
            self.assertEqual(evaluate_hybrid(out,shared,info),hybrid)
            train.native._mmap.close();val['task1'].native._mmap.close()

    def test_saved_candidate_evaluation_reuse_and_two_arm_comparison_without_training(self):
        # Synthetic epoch markers exercise file contracts only; weights are never optimized.
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);shared,info=f.prepare();public=load_checkpoint(f.public)
            folders={}
            for group in ('control','slow'):
                folder=Path(tmp)/group;folders[group]=folder
                config=run_config(group,shared,info)
                model=make_model(CONFIG,group,public['ema'])
                slow=torch.load(f.slow,weights_only=True)['model']
                value={'protocol':PROTOCOL,'score_protocol':SCORE_PROTOCOL,'group':group,'config':config,
                       'epoch':100,'model_config':info['model_config'],'base_architecture_hash':CONFIG.fingerprint,
                       'model':model.state_dict(),'model_ema_sha256':state_hash(model.state_dict()),
                       'scales':info['scales'],'scales_sha256':info['scales_sha256'],'slow_state':slow,
                       'slow_state_sha256':state_hash(slow),'slow_variant':config['slow_variant'],
                       'sampling':info['sampling']}
                (folder/'checkpoints').mkdir(parents=True)
                write(folder/'checkpoints/resolved_config.json',config)
                write(folder/'checkpoints/completed.json',{'completed':True,'total_epochs':100})
                for name in ('best_r','last'):
                    cp=folder/'checkpoints'/(name+'.pt');torch.save(value,cp)
                    with patch('baselines.B5_scratch.evaluate.datasets',return_value=(None,{},None)),patch(
                            'baselines.B5_scratch.evaluate.validate') as validate_mock:
                        def generate(model,validation,shared,info,device,output,plots):
                            from baselines.B5_scratch.data import ConditionedDataset
                            meta=read(shared/'records.json')
                            native=np.load(shared/'validation_slow.npy')
                            ds={k:ConditionedDataset(v,native,meta['validation']) for k,v in f.val.items()}
                            with patch('baselines.B5_scratch.runtime.sample',side_effect=fake_sample):
                                return validate(model,ds,shared,info,device,output)
                        validate_mock.side_effect=generate
                        final_evaluate(SimpleNamespace(shared=str(shared),checkpoint=str(cp),
                            output=str(folder/'evaluations'/name),reuse_root=str(folder/'evaluations'),device='cpu',plots=False))
                        if name=='last':validate_mock.assert_not_called()
                self.assertTrue(read(folder/'evaluations/last/evaluation.json')['reused_identical_ema_from'])
            compare(SimpleNamespace(shared=str(shared),control=str(folders['control']),slow=str(folders['slow']),
                                    slow_f=None,total_epochs=100,output=str(Path(tmp)/'comparison')))
            delta=read(Path(tmp)/'comparison/deltas.json')
            self.assertEqual(delta['mean_r_deltas']['slow_minus_control'],0.)


if __name__=='__main__':unittest.main()
