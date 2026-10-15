"""Synthetic S50/F audit and workflow tests. No optimizer updates or ECG training."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from baselines.B5_exp.core import Records, json_write, sha
from baselines.B5_exp.context_audit import read, CANDIDATES
from baselines.B5_exp.check_s50 import check as check_s50
from baselines.B5_exp.check_s50_f import check as check_s50_f
from baselines.B5_exp.check_loss import initializer, implementation_hashes
from baselines.B5_exp.loss_metrics import Reference
from baselines.B5_exp.evaluate import run as evaluate
from baselines.B5_exp import train_loss
from baselines.B5_exp.train import state_hash
from baselines.B5_exp.tests.test_losses import fixture
from baselines.B5_exp.tests.test_s50 import mock_control
from baselines.B5_exp.s50_f_audit import (REFERENCE, FIXED_F_WEIGHT, verify_unchanged_code, canonical,
                                         verify_control, assert_initial_equal)
from baselines.B5_exp.compare_s50_f import compare


def synthetic_s50_control(root, prepared, info, init):
    """Fabricate completion metadata around actual no-training predictions."""
    old_s30 = mock_control(root,prepared,info,init)
    calibration = root/'ordinary_s50_preflight.json'
    check_s50(SimpleNamespace(prepared=str(prepared),init_checkpoint=str(init),control_run=str(old_s30),
                             report=str(calibration),device='cpu'))
    cfg=train_loss.configuration(SimpleNamespace(group='C',context_seconds=50,epochs=20,batch_records=2,
        validate_every=5,calibration=str(calibration),init_checkpoint=str(init)),prepared)
    snapshot=read(calibration)
    snapshot['implementation_sha256']=read(REFERENCE)['implementation_sha256']
    snapshot['weights']['residual']=FIXED_F_WEIGHT
    control=root/'synthetic_S50'
    control.mkdir()
    json_write(control/'context_preflight.json',snapshot)
    cfg.update(implementation_sha256=snapshot['implementation_sha256'],
               calibrated_weights=snapshot['weights'],calibration_sha256=sha(control/'context_preflight.json'))
    json_write(control/'resolved_config.json',cfg)
    model,source=initializer(init,prepared)
    result=train_loss.validation(model,DataLoader(Records(prepared/'validation.npz'),batch_size=2),
                                 torch.device('cpu'),Reference(prepared,info),50)
    json_write(control/'initial_evaluation.json',result)
    history=[{'epoch':epoch,**({'validation':result} if epoch%5==0 else {})} for epoch in range(21)]
    (control/'history.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history),encoding='utf-8')
    json_write(control/'completed.json',{'completed':True,'additional_epochs':20,'group':'C'})
    for candidate in CANDIDATES:
        torch.save({'protocol':cfg['protocol'],'config':cfg,'epoch':20,'model':source['model']},control/(candidate+'.pt'))
    evaluate(SimpleNamespace(prepared=str(prepared),checkpoint=str(control/'best_score.pt'),
                             output=str(control/'evaluation_best_score'),device='cpu',no_plots=True))
    for candidate in ('best_slow_r','last'):
        folder=control/('evaluation_'+candidate)
        shutil.copytree(control/'evaluation_best_score',folder)
        evaluation=read(folder/'evaluation.json')
        evaluation.update(probe_checkpoint=str(control/(candidate+'.pt')),probe_checkpoint_sha256=sha(control/(candidate+'.pt')))
        json_write(folder/'evaluation.json',evaluation)
    return control


class S50FTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_fixed_F_activation_is_ignored_by_pair_comparison(self):
        c={'group':'C','context_seconds':50,'mean_weight':0.,'residual_weight':0.,'epochs':20,'seed':42}
        f={**c,'group':'F','residual_weight':FIXED_F_WEIGHT}
        self.assertEqual(canonical(c),canonical(f))
        with self.assertRaises(ValueError): canonical({**f,'residual_weight':.1})
        with self.assertRaises(ValueError): canonical({**f,'mean_weight':.1})
        self.assertNotEqual(canonical(c),canonical({**f,'epochs':40}))
        verify_unchanged_code(implementation_hashes())
        broken=dict(implementation_hashes())
        broken['baselines/B5_exp/core.py']='changed CNN/context'
        with self.assertRaises(ValueError): verify_unchanged_code(broken)

    def test_initial_score_guard(self):
        fields=('task1_r_missing11','task2_r_missing11','ecg_machine_d6_task2_missing_lead_mean_rmse_uV',
                'body_scale_d6_task2_missing_lead_mean_rmse_uV','competition_total_score')
        a={'hybrid_score':dict.fromkeys(fields,1.),'baseline_score':dict.fromkeys(fields,1.),
           'slow_r_unique_quality_good_50Hz':.3}
        assert_initial_equal(a,copy.deepcopy(a))
        b=copy.deepcopy(a)
        b['hybrid_score']['competition_total_score']+=.001
        with self.assertRaises(ValueError): assert_initial_equal(a,b)

    def test_preflight_epoch_zero_resume_evaluate_compare_without_training(self):
        class StopBeforeTraining(RuntimeError): pass
        class NoTrainingLoader:
            def __iter__(self): raise StopBeforeTraining()

        with tempfile.TemporaryDirectory() as directory, patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
            root=Path(directory)
            prepared,info,init=fixture(root)
            control=synthetic_s50_control(root,prepared,info,init)
            before={p.relative_to(control).as_posix():sha(p) for p in control.rglob('*') if p.is_file()}
            preflight=root/'s50_f_preflight.json'
            check_s50_f(SimpleNamespace(prepared=str(prepared),init_checkpoint=str(init),control_run=str(control),
                                       report=str(preflight),device='cpu'))
            report=read(preflight)
            self.assertEqual(report['optimizer_updates'],0)
            self.assertEqual(report['fixed_F_weight'],FIXED_F_WEIGHT)
            self.assertEqual(report['prediction_shape'],[2,11,6000])
            args=SimpleNamespace(group='F',context_seconds=50,epochs=20,batch_records=2,validate_every=5,
                prepared=str(prepared),init_checkpoint=str(init),calibration=str(preflight),
                output=str(root/'S50_F'),device='cpu',resume=None)
            cfg=train_loss.configuration(args,prepared)
            self.assertEqual(canonical(cfg),canonical(read(control/'resolved_config.json')))
            self.assertEqual(cfg['residual_weight'],FIXED_F_WEIGHT)
            wrong=copy.copy(args); wrong.epochs=40
            with self.assertRaisesRegex(ValueError,'more than'): train_loss.configuration(wrong,prepared)
            wrong=copy.copy(args); wrong.init_checkpoint=str(control/'best_score.pt')
            with self.assertRaises(ValueError): train_loss.configuration(wrong,prepared)
            loader=train_loss.DataLoader
            def checked_loader(dataset,**kwargs):
                return NoTrainingLoader() if kwargs.get('shuffle') else loader(dataset,**kwargs)
            run=Path(args.output)
            with patch.object(train_loss,'DataLoader',side_effect=checked_loader):
                with self.assertRaises(StopBeforeTraining): train_loss.run(args)
                last=torch.load(run/'last.pt',weights_only=True)
                self.assertEqual(last['epoch'],0)
                self.assertEqual(last['optimizer']['state'],{})
                self.assertEqual(state_hash(last['model']),state_hash(torch.load(init,weights_only=True)['model']))
                assert_initial_equal(read(run/'initial_evaluation.json'),read(control/'initial_evaluation.json'))
                args.resume=str(run/'last.pt')
                with self.assertRaises(StopBeforeTraining): train_loss.run(args)
            # Completion metadata is fabricated only for comparison plumbing;
            # tensors remain initializer weights, and no optimizer ran.
            last['epoch']=20
            torch.save(last,run/'last.pt')
            json_write(run/'completed.json',{'completed':True,'additional_epochs':20,'group':'F'})
            (run/'history.jsonl').write_text(''.join(json.dumps({'epoch':e})+'\n' for e in range(21)),encoding='utf-8')
            evaluate(SimpleNamespace(prepared=str(prepared),checkpoint=str(run/'last.pt'),
                                     output=str(run/'evaluation_last'),device='cpu',no_plots=True))
            for candidate in ('best_score','best_slow_r'):
                folder=run/('evaluation_'+candidate)
                shutil.copytree(run/'evaluation_last',folder)
                evaluation=read(folder/'evaluation.json')
                evaluation.update(epoch=0,probe_checkpoint=str(run/(candidate+'.pt')),
                                  probe_checkpoint_sha256=sha(run/(candidate+'.pt')))
                json_write(folder/'evaluation.json',evaluation)
            compare(control,run,root/'comparison')
            self.assertTrue(read(root/'comparison/audit.json')['epoch_zero_scores_equal'])
            (run/'last.pt').write_bytes(b'changed candidate')
            with self.assertRaisesRegex(ValueError,'Stale'): compare(control,run,root/'stale')
            self.assertEqual(before,{p.relative_to(control).as_posix():sha(p) for p in control.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError): check_s50_f(SimpleNamespace(report=str(preflight)))
            (control/'completed.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'changed'): verify_control(report['s50_f_pair']['control_reference'])


if __name__=='__main__':
    unittest.main()
