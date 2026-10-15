from __future__ import annotations

from contextlib import ExitStack
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from baselines.B5.cli import training_cli, validate_training_start
from baselines.B5.config import load_config, resolve_path
from baselines.B5.prepare import preflight
from baselines.B5.sampling_settings import inference_sampling
from ecg12gen.evaluate import QC_COMPETITION_SCORE_PROTOCOL as COMPETITION_SCORE_PROTOCOL

ROOT=Path(__file__).resolve().parents[3]


class PublicVsScratchTests(unittest.TestCase):
    def test_preflight_exposes_sampling_score_and_output_before_training(self):
        config=load_config(ROOT/'configs/experiments/b5_main_v2_scratch_meta.yaml')
        with patch('baselines.B5.prepare.load_preprocessor', side_effect=FileNotFoundError('fixture scales absent')):
            report=preflight(config)
        self.assertFalse(report['ready'])
        self.assertFalse(report['training_started'])
        self.assertEqual(report['sampling'],inference_sampling())
        self.assertTrue(report['sampling_matches_final_default'])
        self.assertEqual(report['selection_metric'],'main_correlation')
        self.assertEqual(report['metric_aggregation_protocol'],COMPETITION_SCORE_PROTOCOL)
        self.assertIn('mean_r_k16',report['output_dir'])

    def test_paired_settings_match_except_initialization_and_output(self):
        scratch=load_config(ROOT/'configs/experiments/b5_main_v2_scratch_meta.yaml')
        fine=load_config(ROOT/'configs/experiments/b5_main_v2_public_finetune_meta.yaml')
        for key in ['model','training','loss','validation','sampling']:
            self.assertEqual(scratch[key],fine[key])
        a,b=dict(scratch['paths']),dict(fine['paths'])
        self.assertNotEqual(a.pop('output_dir'),b.pop('output_dir'))
        self.assertEqual(a,b)
        self.assertEqual(scratch['stage'],'local');self.assertEqual(fine['stage'],'finetune')
        self.assertEqual(fine['initialization']['allowed_source_stage'],'public')
        self.assertEqual(scratch['sampling']['samples'],16)
        self.assertEqual(scratch['training']['epochs'],100)
        self.assertEqual(scratch['training']['validate_every'],5)
        self.assertEqual(resolve_path(scratch,'demographics'), ROOT/'data/userinfobean.csv')

    def test_start_modes_reject_hidden_pretraining_and_ambiguous_arguments(self):
        validate_training_start({'stage':'local'},None,None)
        validate_training_start({'stage':'local'},None,'local-last.pt')
        validate_training_start({'stage':'finetune'},'public.pt',None)
        validate_training_start({'stage':'finetune'},None,'fine-last.pt')
        for cfg,init,resume in [({'stage':'local'},'public.pt',None),
                                ({'stage':'finetune'},None,None),
                                ({'stage':'local'},'a','b'),({'stage':'public'},None,None)]:
            with self.assertRaises(ValueError):validate_training_start(cfg,init,resume)

    def test_cli_dispatches_scratch_and_finetune_without_optimizer_or_updates(self):
        # Mock the whole training function; no training run or optimizer exists.
        for stage,extra,expected in [('local',[],None),('finetune',['--init-checkpoint','public.pt'],'public.pt')]:
            with self.subTest(stage=stage), ExitStack() as stack:
                cfg={'stage':stage}
                stack.enter_context(patch.object(sys,'argv',['train','--config','fixture.yaml','--device','cpu','--execute-training',*extra]))
                stack.enter_context(patch('baselines.B5.config.load_config',return_value=cfg))
                run=stack.enter_context(patch('baselines.B5.runtime.run_training'))
                training_cli()
                run.assert_called_once_with(cfg,'cpu',expected,None)

    def test_cli_no_opt_in_never_loads_config_or_runs_training(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys,'argv',['train','--config','nonexistent.yaml']))
            stack.enter_context(patch('sys.stderr',new=io.StringIO()))
            load=stack.enter_context(patch('baselines.B5.config.load_config'))
            with self.assertRaises(SystemExit):training_cli()
            load.assert_not_called()


if __name__=='__main__':unittest.main()
