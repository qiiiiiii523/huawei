from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from baselines.B7 import train as trainer
from baselines.B7.config import LossConfig, SamplingConfig
from baselines.B7.checkpoint import load_checkpoint
from baselines.B7.tests.test_b7 import make_manifest, small_config
from scripts.reschedule_b7_validation import migrate, verify_schedule_upgrade, package_hash


class RescheduleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_code_upgrade_guard_allows_only_committed_schedule_and_reader_files(self):
        original={'model.py':b'model','train.py':b'old training'}
        updated={'model.py':b'model','train.py':b'checkpoint scheduling'}
        with patch('scripts.reschedule_b7_validation.package_at_revision',side_effect=[updated,original]), \
             patch.object(trainer,'code_sha',return_value=package_hash(updated)):
            self.assertEqual(verify_schedule_upgrade(package_hash(original)),'032fc7d')
        updated['model.py']=b'changed network'
        with patch('scripts.reschedule_b7_validation.package_at_revision',side_effect=[updated,original]), \
             patch.object(trainer,'code_sha',return_value=package_hash(updated)):
            with self.assertRaisesRegex(ValueError,'Computational model'):
                verify_schedule_upgrade(package_hash(original))

    def source(self, root):
        manifest=make_manifest(root)
        config={'stage':'public10','model':small_config().to_dict(),'loss':asdict(LossConfig()),
                'sampling':asdict(SamplingConfig(window_seconds=10,steps=1,samples=1)),
                'training':{'seed':4,'updates':4,'validate_every':2,'batch_size':1,'learning_rate':.0001,
                            'weight_decay':.01,'ema_decay':.9,'gradient_clip':1.,'boundary_probability':0.}}
        cp=root/'cfg.json'
        cp.write_text(json.dumps(config))
        args=Namespace(config=str(cp),manifest=str(manifest),output=str(root/'old/P'),device='cpu',resume=False,init_checkpoint=None)
        original=trainer.evaluate
        count=0
        def interrupt(*a,**kw):
            nonlocal count
            count+=1
            if count==3:
                raise RuntimeError('interrupted')
            return original(*a,**kw)
        with patch.object(trainer,'evaluate',interrupt):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                trainer.train(args)
        self.assertEqual(load_checkpoint(root/'old/P/last.pt')['update'],2)
        return args,manifest,cp,config

    def test_resume_preserves_state_best_and_finishes_like_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args,manifest,cp,config=self.source(root)
            original_bytes=(root/'old/P/last.pt').read_bytes()
            original_best=(root/'old/P/best_r.pt').read_bytes()
            before=load_checkpoint(root/'old/P/last.pt')
            migrate(root/'old',root/'new',manifest,cp,4)
            migrated=load_checkpoint(root/'new/P/last.pt')
            self.assertEqual(original_bytes,(root/'old/P/last.pt').read_bytes())
            self.assertEqual(original_best,(root/'old/P/best_r.pt').read_bytes())
            self.assertEqual(migrated['update'],before['update'])
            self.assertEqual(migrated['best_r'],before['best_r'])
            for key in ('model','ema'):
                for name in before[key]: self.assertTrue(torch.equal(before[key][name],migrated[key][name]))
            self.assertTrue(torch.equal(before['rng']['torch'],migrated['rng']['torch']))
            self.assertEqual(migrated['identity']['config']['training']['validate_every'],4)
            args.output=str(root/'new/P');args.resume=True
            trainer.train(args)
            cp.write_text(json.dumps(config));args.output=str(root/'reference/P');args.resume=False
            trainer.train(args)
            a,b=[load_checkpoint(root/f'{name}/P/last.pt') for name in ('new','reference')]
            for name in a['model']: self.assertTrue(torch.equal(a['model'][name],b['model'][name]),name)
            for name in a['ema']: self.assertTrue(torch.equal(a['ema'][name],b['ema'][name]),name)
            for key,values in a['optimizer']['state'].items():
                for name,value in values.items(): self.assertTrue(torch.equal(value,b['optimizer']['state'][key][name]))

    def test_rejects_unapproved_learning_rate_change_without_writing_new_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args,manifest,cp,config=self.source(root)
            config['training']['learning_rate']=.002
            cp.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'Current config differs'):
                migrate(root/'old',root/'new',manifest,cp,4)
            self.assertFalse((root/'new').exists())

    def test_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args,manifest,cp,config=self.source(root)
            with (root/'train.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Prepared data/scales changed'):
                migrate(root/'old',root/'new',manifest,cp,4)
            self.assertFalse((root/'new').exists())

    def test_periodic_checkpoint_without_validation_can_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args,manifest,cp,config=self.source(root)
            migrate(root/'old',root/'new',manifest,cp,4,checkpoint_every=1,allow_code_upgrade=True)
            args.output=str(root/'new/P');args.resume=True
            original=trainer.balanced_indices
            calls=0
            def interrupt(*a,**kw):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('between validations')
                return original(*a,**kw)
            with patch.object(trainer,'balanced_indices',interrupt):
                with self.assertRaisesRegex(RuntimeError,'between validations'):
                    trainer.train(args)
            saved=load_checkpoint(root/'new/P/last.pt')
            self.assertEqual(saved['update'],3)
            self.assertEqual(saved['last_validated_update'],2)
            self.assertFalse(saved['pending_validation'])
            self.assertEqual(saved['best_r'],load_checkpoint(root/'old/P/last.pt')['best_r'])
            trainer.train(args)
            self.assertTrue((root/'new/P/completed.json').is_file())

    def test_interrupted_scheduled_validation_is_retried_before_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args,manifest,cp,config=self.source(root)
            migrate(root/'old',root/'new',manifest,cp,4,checkpoint_every=1,allow_code_upgrade=True)
            args.output=str(root/'new/P');args.resume=True
            with patch.object(trainer,'evaluate',side_effect=RuntimeError('during validation')):
                with self.assertRaisesRegex(RuntimeError,'during validation'):
                    trainer.train(args)
            saved=load_checkpoint(root/'new/P/last.pt')
            self.assertEqual(saved['update'],4)
            self.assertTrue(saved['pending_validation'])
            self.assertFalse((root/'new/P/completed.json').exists())
            original=trainer.evaluate
            with patch.object(trainer,'evaluate',wraps=original) as spy:
                trainer.train(args)
                self.assertEqual(spy.call_count,1)
            self.assertFalse(load_checkpoint(root/'new/P/last.pt')['pending_validation'])
            self.assertTrue((root/'new/P/completed.json').is_file())


if __name__=='__main__':
    unittest.main()
