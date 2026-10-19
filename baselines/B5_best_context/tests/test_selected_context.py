"""Pinned B43 transfer guards and actual three-route reliable-I/QC scoring."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from baselines.B5_best.common import sha
from baselines.B5_best_context.selected_study import check_contract, cases
from baselines.B5_best_context.four_routes import SAMPLING
from baselines.B5_best_context.body_features import BODY_PROTOCOL, BEAT16_ENCODING
from baselines.B5_best_context.body_history import B_POLICY, A_POLICY
from baselines.B5_best_context.body_alpha_core import SharedCorrection, FixedAlphaAdapter
from baselines.B5_best_context.body_study import run_study
from . import test_context


class SelectedContextTests(unittest.TestCase):
    def fixture(self):
        population = {task: {key: task+key for key in ('metadata_sha256','target_sha256','QC_sha256')}
                      for task in ('task1','task2')}
        cache = dict(feature_encoding=BEAT16_ENCODING, history_preprocessing=B_POLICY,
            frozen=dict(base_checkpoint_sha256='old_P', sampling=SAMPLING, slow_beta=.75,
                        baseline_population=deepcopy(population)))
        payload = dict(protocol=BODY_PROTOCOL, group='context', training_seed=43, epoch=15,
            feature_encoding=BEAT16_ENCODING, history_preprocessing=B_POLICY, cache_sha256='B_cache',
            base_checkpoint_sha256='old_P', sampling=SAMPLING, slow_beta=.75)
        baseline = dict(checkpoint_sha256='new_FM_P45', sampling=SAMPLING, slow_beta=.75)
        return payload, cache, baseline, population

    def test_explicit_transfer_pins_B43_without_rewriting_its_training_base(self):
        p,c,b,pop = self.fixture()
        original = deepcopy(p)
        with self.assertRaisesRegex(ValueError, 'allow-base-transfer'):
            check_contract(p,c,'B_cache',b,pop,allow_base_transfer=False)
        report = check_contract(p,c,'B_cache',b,pop,allow_base_transfer=True)
        self.assertTrue(report['frozen_body_transfer'])
        self.assertEqual(report['body_training_base_sha256'], 'old_P')
        self.assertEqual(report['evaluated_base_sha256'], 'new_FM_P45')
        self.assertEqual(report['body_training_seed'],43)
        self.assertEqual(p,original)
        self.assertFalse(check_contract(p,c,'B_cache',{**b,'checkpoint_sha256':'old_P'},pop,
                                        allow_base_transfer=False)['frozen_body_transfer'])

    def test_wrong_body_preprocessing_seed_epoch_data_or_noise_cannot_be_transferred(self):
        p,c,b,pop = self.fixture()
        for key,value in (('epoch',27),('training_seed',42),('history_preprocessing',A_POLICY),
                          ('cache_sha256','other'),('base_checkpoint_sha256','other'),('record_gate','G')):
            with self.assertRaises(ValueError):
                check_contract({**p,key:value},c,'B_cache',b,pop,allow_base_transfer=True)
        for key in ('metadata_sha256','target_sha256','QC_sha256'):
            changed=deepcopy(pop); changed['task2'][key]='changed'
            with self.assertRaisesRegex(ValueError,'Changed validation'):
                check_contract(p,c,'B_cache',b,changed,allow_base_transfer=True)
        with self.assertRaises(ValueError):
            check_contract(p,{**c,'history_preprocessing':A_POLICY},'B_cache',b,pop,allow_base_transfer=True)
        with self.assertRaises(ValueError):
            check_contract(p,c,'B_cache',{**b,'sampling':{**SAMPLING,'seed':43}},pop,allow_base_transfer=True)

    def test_actual_three_groups_share_predictions_and_keep_routes_separate(self):
        class Engine:
            def __call__(self,p,i,h):
                p[1] += .02*i[0]
                return p,{}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            test_context.StudyTests().baseline(root/'base')
            before=sha(root/'base/task2/prediction_uV.npy')
            shared=SharedCorrection(Engine())
            rows=run_study(root/'base',root/'study',lambda r:None,None,None,case_specs=cases(),
                adapter_map={'body':FixedAlphaAdapter(shared,1.)},additive_routes=[],notes=['synthetic selected context'])
            self.assertEqual([r['group'] for r in rows],['baseline','d6_only','both'])
            self.assertEqual(len({r['task1_r'] for r in rows}),1)
            self.assertEqual(rows[1]['machine_r'],rows[2]['machine_r'])
            self.assertEqual(rows[0]['body_r'],rows[1]['body_r'])
            self.assertNotEqual(rows[1]['body_r'],rows[2]['body_r'])
            self.assertEqual(sha(root/'base/task2/prediction_uV.npy'),before)
            self.assertEqual(shared.forward_calls,1)


if __name__=='__main__':
    unittest.main()
