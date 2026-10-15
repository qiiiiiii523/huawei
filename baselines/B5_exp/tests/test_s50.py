"""Synthetic context/compatibility/workflow checks. Never execute optimizer.step."""
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

from baselines.B5_exp.core import contexts, generate, SlowCNN, Records, json_write, sha
from baselines.B5_exp.legacy_context import contexts_s30_reference, generate_s30_reference
from baselines.B5_exp.context_audit import (read, REFERENCE_FILE, model_code_hash, verify_model_and_fixed_code,
                                           verify_control_files, audit_control, canonical)
from baselines.B5_exp.check_loss import check as check_loss, implementation_hashes, initializer
from baselines.B5_exp.check_s50 import check as check_s50
from baselines.B5_exp.compare_context import compare
from baselines.B5_exp import train_loss
from baselines.B5_exp.loss_metrics import Reference
from baselines.B5_exp.evaluate import run as evaluate
from baselines.B5_exp.tests.test_losses import fixture
from baselines.B5_exp.train import state_hash


def inputs(batch):
    return {k:batch[k] for k in ('anchor','numeric','sex','field_mask','age_topcoded')}


def mock_control(root, prepared, info, init):
    """Fabricated 20-epoch report fixture, with actual no-training model evaluation."""
    calibration = root / 'ordinary_calibration.json'
    check_loss(SimpleNamespace(prepared=str(prepared), init_checkpoint=str(init), report=str(calibration),
                               device='cpu', gradient_fraction=.1, residual_width=51))
    config = train_loss.configuration(SimpleNamespace(group='C', epochs=20, batch_records=2,
        validate_every=5, calibration=str(calibration), init_checkpoint=str(init), context_seconds=30), prepared)
    config['implementation_sha256'] = read(REFERENCE_FILE)['implementation_sha256']
    control = root / 'synthetic_C'
    control.mkdir()
    json_write(control / 'resolved_config.json', config)
    model, source = initializer(init, prepared)
    report = train_loss.validation(model, DataLoader(Records(prepared/'validation.npz'), batch_size=2),
                                   torch.device('cpu'), Reference(prepared, info), 30)
    json_write(control/'initial_evaluation.json', report)
    for name in ('best_slow_r','best_score','last'):
        torch.save({'protocol':config['protocol'], 'config':config, 'epoch':20,
                    'model':source['model']}, control/(name+'.pt'))
    (control/'history.jsonl').write_text(''.join(json.dumps({'epoch':epoch})+'\n' for epoch in range(21)), encoding='utf-8')
    json_write(control/'completed.json', {'completed':True,'additional_epochs':20,'group':'C'})
    evaluate(SimpleNamespace(prepared=str(prepared), checkpoint=str(control/'best_score.pt'),
                             output=str(control/'evaluation_best_score'), device='cpu', no_plots=True))
    for name in ('best_slow_r','last'):
        folder = control/('evaluation_'+name)
        shutil.copytree(control/'evaluation_best_score',folder)
        value = read(folder/'evaluation.json')
        value['probe_checkpoint'] = str(control/(name+'.pt'))
        value['probe_checkpoint_sha256'] = sha(control/(name+'.pt'))
        json_write(folder/'evaluation.json',value)
    return control


class S50Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_context50_center_order_edges_and_patient_separation(self):
        anchor = torch.arange(12000.).reshape(2,1,6000)
        short, long = contexts(anchor,10),contexts(anchor,50)
        self.assertEqual(tuple(long.shape),(24,1,2500))
        self.assertTrue(torch.equal(long[...,1000:1500],short))
        self.assertTrue(torch.equal(long[2,0],anchor[0,0,:2500]))
        self.assertTrue(torch.equal(long[0,0,:1000],torch.zeros(1000)))
        self.assertTrue(torch.equal(long[11,0,-1000:],torch.full((1000,),5999.)))
        self.assertTrue(torch.equal(long[12,0,:1000],torch.full((1000,),6000.)))
        for seconds in (10,30):
            self.assertTrue(torch.equal(contexts(anchor,seconds),contexts_s30_reference(anchor,seconds)))
        for seconds in (20,40,60):
            with self.assertRaises(ValueError): contexts(anchor,seconds)
        with self.assertRaises(ValueError): contexts(anchor[...,:5999],50)

    def test_stitching_gradients_and_target_free_interface(self):
        class Echo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor(1.))
            def forward(self,anchor,**metadata):
                return anchor.expand(-1,11,-1)*self.weight
        batch={'anchor':torch.arange(6000.)[None,None], 'numeric':torch.zeros(1,3),
               'sex':torch.tensor([2]), 'field_mask':torch.zeros(1,4,dtype=torch.bool),
               'age_topcoded':torch.zeros(1,1), 'target':torch.full((1,11,6000),float('nan'))}
        model=Echo()
        p=generate(model,batch,50)
        self.assertTrue(torch.equal(p[:,:1],batch['anchor']))
        p.mean().backward()  # Derivative only; no optimizer.
        self.assertAlmostEqual(float(model.weight.grad),2999.5,places=2)
        self.assertEqual(float(model.weight),1.)
        with torch.no_grad():
            for seconds in (10,30):
                self.assertTrue(torch.equal(generate(model,batch,seconds),generate_s30_reference(model,batch,seconds)))

    def test_model_and_score_unchanged_reference_contract(self):
        ref=verify_model_and_fixed_code(implementation_hashes())
        core=Path(__file__).resolve().parents[1]/'core.py'
        source=core.read_text(encoding='utf-8')
        self.assertEqual(model_code_hash(source),ref['model_code_sha256'])
        self.assertEqual(sum(p.numel() for p in SlowCNN().parameters()),213511)
        with self.assertRaises(ValueError):
            changed=dict(implementation_hashes())
            changed['ecg12gen/evaluate.py']='changed score'
            verify_model_and_fixed_code(changed)
        self.assertNotEqual(model_code_hash(source.replace('nn.Embedding(3, 4)','nn.Embedding(3, 5)')),ref['model_code_sha256'])

    def test_s50_requires_matching_preflight_and_original_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            prepared, info, init=fixture(root)
            calibration=root/'calibration.json'
            check_loss(SimpleNamespace(prepared=str(prepared),init_checkpoint=str(init),report=str(calibration),
                                      device='cpu',gradient_fraction=.1,residual_width=51))
            args=SimpleNamespace(group='C',context_seconds=50,epochs=20,batch_records=2,validate_every=5,
                                 calibration=str(calibration),init_checkpoint=str(init))
            with self.assertRaisesRegex(ValueError,'check_s50'):
                train_loss.configuration(args,prepared)

    def test_preflight_train_entry_resume_and_comparison_no_updates(self):
        class StopBeforeTraining(RuntimeError): pass
        class NoTrainingLoader:
            def __iter__(self): raise StopBeforeTraining()

        with tempfile.TemporaryDirectory() as directory, patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('Forbidden local training')):
            root=Path(directory)
            prepared,info,init=fixture(root)
            control=mock_control(root,prepared,info,init)
            control_before={p.relative_to(control).as_posix():sha(p) for p in control.rglob('*') if p.is_file()}
            preflight=root/'s50_preflight.json'
            check_s50(SimpleNamespace(prepared=str(prepared),init_checkpoint=str(init),control_run=str(control),
                                     report=str(preflight),device='cpu'))
            proof=read(preflight)
            self.assertTrue(proof['s50_compatibility']['s30_bitwise_equal'])
            self.assertEqual(proof['s50_compatibility']['input_shapes']['50'],[12,1,2500])
            self.assertEqual(proof['optimizer_updates'],0)
            run=root/'S50'
            args=SimpleNamespace(prepared=str(prepared),init_checkpoint=str(init),calibration=str(preflight),
                group='C',context_seconds=50,epochs=20,batch_records=2,validate_every=5,
                output=str(run),device='cpu',resume=None)
            config=train_loss.configuration(args,prepared)
            self.assertEqual(canonical(config),canonical(read(control/'resolved_config.json')))
            wrong=copy.copy(args)
            wrong.epochs=25
            with self.assertRaisesRegex(ValueError,'optimization'):
                train_loss.configuration(wrong,prepared)
            wrong=copy.copy(args)
            wrong.group='F'
            with self.assertRaisesRegex(ValueError,'original loss'):
                train_loss.configuration(wrong,prepared)
            loader=train_loss.DataLoader
            def checked_loader(dataset,**kwargs):
                return NoTrainingLoader() if kwargs.get('shuffle') else loader(dataset,**kwargs)
            with patch.object(train_loss,'DataLoader',side_effect=checked_loader):
                with self.assertRaises(StopBeforeTraining): train_loss.run(args)
                last=torch.load(run/'last.pt',weights_only=True)
                self.assertEqual(last['config']['context_seconds'],50)
                self.assertEqual(last['epoch'],0)
                self.assertEqual(last['optimizer']['state'],{})
                self.assertEqual(state_hash(last['model']),state_hash(torch.load(init,weights_only=True)['model']))
                args.resume=str(run/'last.pt')
                with self.assertRaises(StopBeforeTraining): train_loss.run(args)
                wrong=copy.copy(args)
                wrong.context_seconds=30
                with self.assertRaisesRegex(ValueError,'Exact resume'):
                    train_loss.run(wrong)
            # Fabricate completion metadata solely to exercise comparison plumbing;
            # all model tensors remain the initializer. No training occurred.
            last['epoch']=20
            torch.save(last,run/'last.pt')
            json_write(run/'completed.json',{'completed':True,'additional_epochs':20,'group':'C'})
            (run/'history.jsonl').write_text(''.join(json.dumps({'epoch':e})+'\n' for e in range(21)),encoding='utf-8')
            evaluate(SimpleNamespace(prepared=str(prepared),checkpoint=str(run/'last.pt'),
                                     output=str(run/'evaluation_last'),device='cpu',no_plots=True))
            for candidate in ('best_score','best_slow_r'):
                folder=run/('evaluation_'+candidate)
                shutil.copytree(run/'evaluation_last',folder)
                evaluation=read(folder/'evaluation.json')
                evaluation['epoch']=0
                evaluation['probe_checkpoint']=str(run/(candidate+'.pt'))
                evaluation['probe_checkpoint_sha256']=sha(run/(candidate+'.pt'))
                json_write(folder/'evaluation.json',evaluation)
            compare(control,run,root/'comparison')
            self.assertTrue(read(root/'comparison/audit.json')['s30_compatibility_verified'])
            (run/'last.pt').write_bytes(b'changed candidate')
            with self.assertRaisesRegex(ValueError,'Stale'):
                compare(control,run,root/'stale_comparison')
            self.assertEqual(control_before,{p.relative_to(control).as_posix():sha(p) for p in control.rglob('*') if p.is_file()})
            (control/'completed.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'changed'):
                verify_control_files(proof['s50_compatibility']['control_reference'])


if __name__=='__main__':
    unittest.main()
