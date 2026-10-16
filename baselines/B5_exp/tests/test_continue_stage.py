"""Continuation/QC regressions on synthetic fixtures; optimizer.step is forbidden."""
import copy
import gc
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import pack_rng
from baselines.B5_exp import PROTOCOL
from baselines.B5_exp import continue_stage as stage
from baselines.B5_exp.core import SlowCNN, Records, json_write, sha
from baselines.B5_exp.losses import LOSS_PROTOCOL
from baselines.B5_exp.stage_metrics import QCReference, SCORE_PROTOCOL, mean_r, validate
from baselines.B5_exp.stage_evaluator import evaluate_record_predictions
from baselines.B5_exp.tests.test_losses import fixture


def source_fixture(root, group='C'):
    prepared, info, init = fixture(root)
    info['source_checks'] = {}
    json_write(prepared/'prepared.json', info)
    # The fixture source is fabricated, not produced by 20 optimizer epochs.
    source = root/'legacy'
    source.mkdir()
    calibration = source/('context_preflight.json' if group == 'C' else 'pair_preflight.json')
    json_write(calibration, {'fixture': True})
    cfg = {'protocol': PROTOCOL, 'experiment_protocol': LOSS_PROTOCOL, 'context_seconds': 50,
           'epochs': 20, 'seed': 42, 'model_channels': 48, 'learning_rate': 1e-4, 'weight_decay': .01,
           'huber_weight': 1., 'pcc_weight': .1, 'mean_weight': 0., 'residual_width_native': 51,
           'group': group, 'calibrated_weights': {'mean': .3, 'residual': 2.5890171538829843},
           'residual_weight': 0. if group == 'C' else 2.5890171538829843,
           'prepared_sha256': sha(prepared/'prepared.json'), 'implementation_sha256': stage.implementation_hashes(),
           'init_checkpoint': str(init), 'init_checkpoint_sha256': sha(init), 'source_epoch': 50,
           'calibration_sha256': sha(calibration), 'batch_records': 2, 'validate_every': 5}
    model = SlowCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    # Construct nonempty Adam state without taking a training step.
    for parameter in model.parameters():
        optimizer.state[parameter] = {'step': torch.tensor(7.), 'exp_avg': torch.full_like(parameter, .03),
                                      'exp_avg_sq': torch.full_like(parameter, .04)}
    original = {'protocol': PROTOCOL, 'experiment_protocol': LOSS_PROTOCOL, 'config': cfg, 'epoch': 20,
                'source_epoch': 50, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'rng': pack_rng(), 'loader_rng': torch.Generator().manual_seed(123).get_state(),
                'output_directory': str(source)}
    candidate = {k: original[k] for k in ('protocol','experiment_protocol','config','epoch','source_epoch','model')}
    original['best_payloads'] = {'slow_r': candidate, 'score': candidate}
    for name in ('best_slow_r','best_score'):
        torch.save(candidate, source/(name+'.pt'))
    torch.save(original, source/'last.pt')
    json_write(source/'resolved_config.json', cfg)
    json_write(source/'completed.json', {'completed': True, 'additional_epochs': 20, 'group': group})
    json_write(source/'initial_evaluation.json', {'fixture': True})
    (source/'history.jsonl').write_text(''.join(json.dumps({'epoch':i})+'\n' for i in range(21)), encoding='utf-8')
    return prepared, info, source, original


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_new_budget_and_directory_safety_and_r_only_selector(self):
        for budget in (0, 20):
            with self.assertRaises(ValueError): stage.stage_config({'epochs':20}, budget)
        cfg = stage.stage_config({'epochs':20,'learning_rate':1e-4},40)
        self.assertEqual(cfg['epochs'],40)
        self.assertEqual(cfg['learning_rate'],1e-4)
        self.assertEqual(cfg['score_protocol'],SCORE_PROTOCOL)
        for path in (Path('/legacy'), Path('/legacy/new'), Path('/')):
            with self.assertRaises(ValueError): stage.eligible_output(path,Path('/legacy'))
        high_r = {'hybrid_score': {**mean_r(.8,.7), 'competition_total_score': 0}}
        low_r = {'hybrid_score': {**mean_r(.6,.6), 'competition_total_score': 60}}
        self.assertGreater(stage.score_value(high_r),stage.score_value(low_r))
        with self.assertRaises(ValueError): mean_r(float('nan'),.3)

    def test_audit_complete_sources_and_reject_budget_or_alias_changes(self):
        for group in ('C','F'):
            with tempfile.TemporaryDirectory() as d:
                prepared, _, source, cp = source_fixture(Path(d),group)
                last,candidates,files = stage.audit_source(source,prepared)
                self.assertEqual(last['epoch'],20)
                self.assertEqual(set(candidates),{'best_slow_r','best_score','last'})
                self.assertTrue(all(sha(p)==h for p,h in files.items()))
                wrong=copy.deepcopy(candidates['best_score']); wrong['model']['output.bias'] += 1
                torch.save(wrong,source/'best_score.pt')
                with self.assertRaisesRegex(ValueError,'ledger'): stage.audit_source(source,prepared)

    def test_restore_adam_loader_python_numpy_torch_rng_without_updates(self):
        with tempfile.TemporaryDirectory() as d:
            _,_,_,cp=source_fixture(Path(d))
            model=SlowCNN(); opt=torch.optim.AdamW(model.parameters(),lr=.02)
            loader_rng=torch.Generator().manual_seed(999)
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                stage.restore_training(cp,model,opt,loader_rng,torch.device('cpu'))
                expected=(random.random(),np.random.rand(),torch.rand(3),torch.randperm(8,generator=loader_rng))
                stage.restore_training(cp,model,opt,loader_rng,torch.device('cpu'))
                actual=(random.random(),np.random.rand(),torch.rand(3),torch.randperm(8,generator=loader_rng))
            self.assertEqual(expected[:2],actual[:2])
            self.assertTrue(torch.equal(expected[2],actual[2]))
            self.assertTrue(torch.equal(expected[3],actual[3]))
            self.assertEqual(opt.param_groups[0]['lr'],1e-4)
            state=next(iter(opt.state.values()))
            self.assertEqual(state['step'].item(),7.)
            self.assertTrue(torch.equal(state['exp_avg'],torch.full_like(state['exp_avg'],.03)))

    def test_qc_record_and_device_aggregation_with_flagged_v4(self):
        with tempfile.TemporaryDirectory() as d:
            prepared,info,_=fixture(Path(d));info['source_checks']={}
            # Flag b/V4; bad predictions remain in output, only scoring eligibility differs.
            with np.load(prepared/'validation.npz') as data:
                values={k:data[k] for k in data.files}
            values['mask'][1,8]=False
            np.savez_compressed(prepared/'validation.npz',**values)
            reference=QCReference(prepared,info)
            try:
                for task in ('task1','task2'):
                    p=np.array(reference.stores[task]['prediction'])
                    for i,row in enumerate(reference.rows[task]):
                        if row['target_record_id']=='b': p[i,9] *= -1
                    # Close mmap before replacing the synthetic source file.
                    reference.stores[task]['prediction']._mmap.close()
                    np.save(Path(info['evaluation_dir'])/task/'prediction_uV.npy',p)
                    reference.stores[task]['prediction']=np.load(Path(info['evaluation_dir'])/task/'prediction_uV.npy',mmap_mode='r')
                score=reference.score(None)
                self.assertAlmostEqual(score['main_correlation'],1.,places=7)
                task='task2'; p=reference.stores[task]['prediction'];y=reference.stores[task]['target'];rows=reference.rows[task]
                full,_=evaluate_record_predictions(p,y,task,rows)
                self.assertAlmostEqual(full['r_missing11'],10/11,places=7)
                self.assertTrue(score['summaries'][task]['quality_mask_applied'])
                self.assertNotIn('task2_rmse_bonus_score',score['summaries'][task])
                # Machine is perfect, body is sign-reversed: equal device weight gives zero.
                p=np.array(y)
                for i,row in enumerate(rows):
                    if row['input_type']=='body_scale_d6': p[i,1:] *= -1
                summary,_=evaluate_record_predictions(p,y,task,rows,target_quality_mask=np.ones(p.shape[:2],bool))
                self.assertAlmostEqual(summary['r_missing11'],0.,places=7)
            finally:
                for store in reference.stores.values():
                    for array in store.values(): array._mmap.close()

    def test_validation_visible_inputs_rng_and_legacy_stage_boundary_resume(self):
        class StopBeforeTraining(RuntimeError): pass
        class InterruptedLoader:
            def __iter__(self): raise StopBeforeTraining()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prepared,info,source,cp=source_fixture(root)
            output=root/'new_stage'
            args=SimpleNamespace(prepared=str(prepared),source_run=str(source),output=str(output),
                                 total_epochs=40,device='cpu',resume=False)
            original_hashes={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
            real_loader=DataLoader
            def loaders(dataset,**kwargs):
                return InterruptedLoader() if kwargs.get('shuffle') else real_loader(dataset,**kwargs)
            # Full functional setup and checkpoint writes, but stop before any optimizer update.
            with patch.object(stage,'DataLoader',side_effect=loaders), \
                 patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                with self.assertRaises(StopBeforeTraining): stage.run(args)
                saved=torch.load(output/'checkpoints/last.pt',weights_only=True)
                self.assertEqual(saved['epoch'],20)
                self.assertEqual(saved['config']['epochs'],40)
                self.assertEqual(saved['optimizer']['state'][0]['step'].item(),7.)
                self.assertTrue(torch.equal(saved['loader_rng'],cp['loader_rng']))
                self.assertEqual(read_json(output/'stage.json')['score_protocol'],SCORE_PROTOCOL)
                best_path=output/'checkpoints/best_r.pt'
                best_hash=sha(best_path)
                args.resume=True
                with self.assertRaises(StopBeforeTraining): stage.run(args)
                self.assertEqual(sha(best_path),best_hash)
                # Alias recovery uses last.pt's ledger, never resets the r-only best.
                best_path.write_bytes(b'Interrupted candidate alias')
                with self.assertRaises(StopBeforeTraining): stage.run(args)
                self.assertEqual(stage.state_hash(torch.load(best_path,weights_only=True)['model']),
                                 stage.state_hash(saved['best_payload']['model']))
                history=[json.loads(r) for r in (output/'history.jsonl').read_text().splitlines()]
                self.assertEqual([r['epoch'] for r in history],[20])
                # Simulate interruption during initial setup: retry before any epoch was trained.
                (output/'checkpoints/last.pt').unlink()
                with self.assertRaises(StopBeforeTraining): stage.run(args)
                history=[json.loads(r) for r in (output/'history.jsonl').read_text().splitlines()]
                self.assertEqual([r['epoch'] for r in history],[20])
                # Export actual QC hybrid arrays and local-detail diagnostic tables, still no updates.
                eval_args=SimpleNamespace(stage=str(output),candidate='best_r',device='cpu')
                with patch('baselines.B5_exp.loss_metrics.plot_fast_details',return_value=[]):
                    stage.evaluate(eval_args)
                    marker=output/'evaluations/best_r/evaluation.json'
                    done=read_json(marker)
                    self.assertTrue(done['completed'])
                    self.assertFalse(done['target_used_at_inference'])
                    self.assertTrue(all(sha(marker.parent/name)==h for name,h in done['files'].items()))
                    marker_hash=sha(marker)
                    stage.evaluate(eval_args)
                    self.assertEqual(sha(marker),marker_hash)
            self.assertTrue(all(sha(p)==h for p,h in original_hashes.items()))
            reference=QCReference(prepared,info)
            try:
                calls=[]
                def visible(model,batch,seconds):
                    calls.append(set(batch))
                    self.assertEqual(set(batch),{'anchor','numeric','sex','field_mask','age_topcoded'})
                    self.assertEqual(seconds,50)
                    return batch['anchor'].repeat(1,11,1)
                rng=torch.get_rng_state().clone()
                with patch('baselines.B5_exp.core.generate',side_effect=visible):
                    validate(SlowCNN(),real_loader(Records(prepared/'validation.npz'),batch_size=2),
                             torch.device('cpu'),reference)
                self.assertTrue(calls)
                # Construction of SlowCNN consumed RNG before validate; directly test validation restoration.
                model=SlowCNN();rng=torch.get_rng_state().clone()
                with patch('baselines.B5_exp.core.generate',side_effect=visible):
                    validate(model,real_loader(Records(prepared/'validation.npz'),batch_size=2),torch.device('cpu'),reference)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            finally:
                for store in reference.stores.values():
                    for array in store.values():array._mmap.close()
            gc.collect()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


if __name__=='__main__': unittest.main()
