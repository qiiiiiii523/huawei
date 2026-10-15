"""Continuation bookkeeping tests, no torch dependency or training."""
import importlib.util
import json
import csv
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('s50_continuation',Path(__file__).parents[1]/'continuation.py')
continuation=importlib.util.module_from_spec(spec);spec.loader.exec_module(continuation)


class ContinuationTests(unittest.TestCase):
    def test_budget_extension_and_interrupted_resume(self):
        self.assertEqual(continuation.resolve_budget(20), (20,20))
        self.assertEqual(continuation.resolve_budget(20,{'epoch':20},40), (40,20))
        self.assertEqual(continuation.resolve_budget(20,{'epoch':23,'total_epochs':40}), (40,40))
        for cp,total in [(None,40),({'epoch':10},40),({'epoch':20},10),({'epoch':40,'total_epochs':40},20)]:
            with self.assertRaises(ValueError): continuation.resolve_budget(20,cp,total)

    def test_current_and_only_registered_legacy_implementations(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'profile.json';p.write_text(json.dumps({'legacy':{'a':'old'},'current':{'a':'new'}}))
            continuation.check_implementation({'a':'new'},{'a':'new'},p)
            continuation.check_implementation({'a':'old'},{'a':'new'},p)
            for old,new in [({'a':'other'},{'a':'new'}),({'a':'old'},{'a':'unreviewed'})]:
                with self.assertRaises(ValueError):continuation.check_implementation(old,new,p)

    def test_archive_is_idempotent_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'checkpoints';out.mkdir()
            for name in ['last.pt','best.pt','best_r.pt','best_bonus.pt','history.jsonl','resolved_config.json','initial_evaluation.json']:
                (out/name).write_text(name)
            (out/'completed.json').write_text(json.dumps({'completed':True,'additional_epochs':20}))
            original=continuation.file_digest(out/'last.pt')
            archive=continuation.archive_boundary(out,20)
            self.assertEqual(continuation.file_digest(archive/'last.pt'),original)
            (out/'completed.json').write_text(json.dumps({'completed':False,'additional_epochs':40}))
            self.assertEqual(continuation.archive_boundary(out,20),archive)
            (archive/'best.pt').write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'archive changed'):continuation.archive_boundary(out,20)

    def test_evaluation_budget_paths_and_checkpoint_identity(self):
        self.assertEqual(continuation.evaluation_directory('run',20),Path('run/evaluations'))
        self.assertEqual(continuation.evaluation_directory('run',40),Path('run/evaluations_epoch40'))
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'last.pt';cp.write_bytes(b'epoch20');marker=Path(tmp)/'evaluation.json'
            self.assertFalse(continuation.completed_evaluation_matches(cp,marker))
            marker.write_text(json.dumps({'completed':True,'checkpoint_sha256':continuation.file_digest(cp)}))
            self.assertTrue(continuation.completed_evaluation_matches(cp,marker))
            cp.write_bytes(b'epoch40')
            self.assertFalse(continuation.completed_evaluation_matches(cp,marker))

    def test_real_compare_two_or_three_arms_requires_matched_budget_and_fresh_weights(self):
        """Exercise the real comparison function with file fixtures, no model imports."""
        package='s50_bookkeeping_fixture'
        protocol='correlation50_bonus10_target_record_qc_v3';full='correlation50_bonus10_target_record_v2'
        def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
        def write(p,v):
            Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v),encoding='utf-8')
        def csv_write(p,rows):
            Path(p).parent.mkdir(parents=True,exist_ok=True)
            with Path(p).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        def module(name,**attributes):
            item=types.ModuleType(name)
            for k,v in attributes.items():setattr(item,k,v)
            return item
        def score(qc=True):
            return {'task1_r_missing11':.6,'task2_r_missing11':.5,
                'ecg_machine_d6_task2_missing_lead_mean_rmse_uV':600.,'body_scale_d6_task2_missing_lead_mean_rmse_uV':500.,
                'competition_total_score':28.783333,'competition_score_protocol':protocol if qc else full}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shared=root/'shared';write(shared/'shared.json',{'source':'unchanged'})
            info={'source_summaries':score(),'source_full_summaries':score(False),'fixed_hybrid_references':{},'training':{'epochs':20}}
            cfg=lambda g,s,i:{'group':g,'shared_sha256':continuation.file_digest(s/'shared.json')}
            mods={package:module(package,GROUPS=('control','slow','slow_f'),SCORE_PROTOCOL=protocol,FULL_SCORE_PROTOCOL=full),
                package+'.common':module(package+'.common',read=read,write=write,csv_write=csv_write,sha=continuation.file_digest),
                package+'.prepare':module(package+'.prepare',load_shared=lambda s:(Path(s),info)),
                package+'.runtime':module(package+'.runtime',run_config=cfg,CANDIDATES={'score':'best.pt','r':'best_r.pt','bonus':'best_bonus.pt'}),
                package+'.scoring':module(package+'.scoring',score_tasks=lambda summaries,**kw:summaries),
                package+'.continuation':continuation,
                'numpy':module('numpy',isclose=lambda a,b,atol,rtol:math.isclose(a,b,abs_tol=atol,rel_tol=rtol))}
            spec=importlib.util.spec_from_file_location(package+'.compare',Path(__file__).parents[1]/'compare.py')
            compare_module=importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules,mods):spec.loader.exec_module(compare_module)
            for group in ['control','slow','slow_f']:
                folder=root/group;out=folder/'checkpoints'
                write(out/'resolved_config.json',cfg(group,shared,info))
                write(out/'completed.json',{'completed':True,'additional_epochs':40,'total_epochs':40})
                write(out/'initial_evaluation.json',{'score':score(),'full_score':score(False)})
                for name in ['best','best_r','best_bonus','last']:
                    (out/(name+'.pt')).write_text(group+name,encoding='utf-8')
                    write(folder/'evaluations_epoch40'/name/'evaluation.json',{'completed':True,
                        'fingerprint':{'group':group,'score_protocol':protocol,'shared_sha256':continuation.file_digest(shared/'shared.json')},
                        'checkpoint_sha256':continuation.file_digest(out/(name+'.pt')),'epoch':40 if name=='last' else 0,
                        'score':score(),'full_score':score(False)})
            args=SimpleNamespace(shared=shared,control=root/'control',slow=root/'slow',slow_f=None,total_epochs=40,output=root/'two')
            compare_module.compare(args)
            self.assertEqual(set(read(root/'two/deltas.json')['QC_best_deltas']),{'S50_minus_control'})
            args.slow_f=root/'slow_f';args.output=root/'three';compare_module.compare(args)
            self.assertEqual(len(read(root/'three/deltas.json')['QC_best_deltas']),3)
            write(root/'slow_f/checkpoints/completed.json',{'completed':True,'additional_epochs':20})
            args.output=root/'mixed'
            with self.assertRaisesRegex(ValueError,'same requested total epoch budget'):compare_module.compare(args)
            args.slow_f=None;args.output=root/'stale';(root/'slow/checkpoints/best.pt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Checkpoint changed'):compare_module.compare(args)


if __name__=='__main__':unittest.main()
