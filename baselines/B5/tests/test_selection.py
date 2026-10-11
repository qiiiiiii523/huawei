from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from ecg12gen.evaluate import competition_score, COMPETITION_SCORE_PROTOCOL
from baselines.B5.config import load_config
from baselines.B5.selection import (SELECTION_PROTOCOL, empty_best_metrics, restore_best_metrics,
    update_best_metrics, save_selected_checkpoints, scores_from_summaries)
from baselines.B5.runtime import validate_resume_protocol

ROOT=Path(__file__).resolve().parents[3]


def payload(best, epoch=0):
    return {'selection_protocol':SELECTION_PROTOCOL,'competition_score_protocol':COMPETITION_SCORE_PROTOCOL,
            'selection_metric':'competition_total_score','best_metrics':best,'checkpoint_role':'last',
            'best_score':None if best['competition_total_score'] is None else best['competition_total_score']['value'],
            'epoch':epoch,'model':{'marker':torch.tensor(epoch)},'ema':{'marker':torch.tensor(epoch)}}


class SelectionTests(unittest.TestCase):
    def test_official_points_and_separate_device_bonus(self):
        s=competition_score(.8,.6,70,140)
        self.assertAlmostEqual(s['main_correlation'],.7)
        self.assertAlmostEqual(s['main_score'],35)
        self.assertAlmostEqual(s['task2_rmse_bonus_score'],7.5)
        self.assertAlmostEqual(s['competition_total_score'],42.5)
        self.assertEqual(competition_score(1,1,0,70)['competition_total_score'],60)
        derived=scores_from_summaries({'task1':{'r_missing11':.8},'task2':{'r_missing11':.6,
            'ecg_machine_d6_task2_missing_lead_mean_rmse_uV':70,
            'body_scale_d6_task2_missing_lead_mean_rmse_uV':140,
            'task2_missing_lead_mean_rmse_uV':105}})
        self.assertEqual(derived,s)

    def test_three_selected_files_update_independently_and_last_always_updates(self):
        best=empty_best_metrics()
        candidates=[(5,.9,700),(10,.88,140),(15,.91,1400),(20,.7,70)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for epoch,r,rmse in candidates:
                s=competition_score(r,r,rmse,rmse)
                best,changed=update_best_metrics(best,s,epoch)
                save_selected_checkpoints(root,payload(best,epoch),changed)
            self.assertEqual({p.name for p in root.iterdir()},{'best.pt','best_r.pt','best_bonus.pt','last.pt'})
            expected={'best.pt':10,'best_r.pt':15,'best_bonus.pt':20,'last.pt':20}
            for file,epoch in expected.items():
                cp=torch.load(root/file,weights_only=True)
                self.assertEqual(cp['epoch'],epoch)
                self.assertEqual(cp['model']['marker'].item(),epoch)
                self.assertEqual(cp['ema']['marker'].item(),epoch)
            restored=restore_best_metrics(torch.load(root/'last.pt',weights_only=True))
            self.assertEqual(restored,best)
            self.assertEqual(best['competition_total_score']['epoch'],10)
            self.assertEqual(best['main_correlation']['epoch'],15)
            self.assertEqual(best['task2_rmse_bonus_score']['epoch'],20)

    def test_resume_keeps_best_records_and_no_validation_does_not_create_fake_best(self):
        best=empty_best_metrics()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save_selected_checkpoints(root,payload(best),[])
            self.assertEqual({p.name for p in root.iterdir()},{'last.pt'})
            self.assertEqual(restore_best_metrics(torch.load(root/'last.pt',weights_only=True)),best)
            best,changed=update_best_metrics(best,competition_score(.8,.8,70,70),5)
            save_selected_checkpoints(root,payload(best,5),changed)
            resumed=restore_best_metrics(torch.load(root/'last.pt',weights_only=True))
            original=copy.deepcopy(resumed)
            resumed,changed=update_best_metrics(resumed,competition_score(.7,.7,140,140),10)
            self.assertEqual(changed,[]);self.assertEqual(resumed,original)
            save_selected_checkpoints(root,payload(resumed,10),changed)
            self.assertEqual(torch.load(root/'best.pt',weights_only=True)['epoch'],5)
            self.assertEqual(torch.load(root/'last.pt',weights_only=True)['epoch'],10)

    def test_ties_do_not_replace_existing_selected_models(self):
        s=competition_score(.8,.6,70,140)
        best,_=update_best_metrics(empty_best_metrics(),s,5)
        state,changed=update_best_metrics(best,s,10)
        self.assertEqual(changed,[]);self.assertEqual(state,best)

    def test_nonfinite_metrics_are_rejected_without_mutating_state(self):
        best=empty_best_metrics();s=competition_score(.8,.6,70,140)
        s['main_correlation']=float('nan')
        with self.assertRaises(FloatingPointError):update_best_metrics(best,s,5)
        self.assertEqual(best,empty_best_metrics())

    def test_legacy_and_invalid_resume_state_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Legacy'):
            restore_best_metrics({'best_score':.66,'selection_metric':'r_missing11'})
        cp=payload(empty_best_metrics())
        for bad in [{**cp,'best_score':1}, {**cp,'best_metrics':{}},
                    {**cp,'competition_score_protocol':'old'}, {**cp,'checkpoint_role':'main_correlation'}]:
            with self.assertRaises(ValueError):restore_best_metrics(bad)
        cfg=load_config(ROOT/'configs/experiments/b5_main_v2_scratch_meta.yaml')
        cp.update(cache_protocol='independent-record-context-v1',evaluation_protocol='machine_body_equal_weight_score_v1',sampling_config=copy.deepcopy(cfg['sampling']))
        validate_resume_protocol(cp,cfg)
        cp['sampling_config']['samples']=1
        with self.assertRaisesRegex(ValueError,'sampling'):validate_resume_protocol(cp,cfg)

    def test_new_run_outputs_and_sampling_policy_are_explicit(self):
        for name in ['b5_main_v2_scratch_meta.yaml','b5_main_v2_public_finetune_meta.yaml']:
            c=load_config(ROOT/'configs/experiments'/name)
            self.assertIn('score60_k16',c['paths']['output_dir'])
            self.assertEqual(c['validation']['selection_metric'],'competition_total_score')
            self.assertEqual(c['sampling']['samples'],16)  # match final generation


if __name__=='__main__':unittest.main()
