"""CPU tests for new-base adaptation data, provenance, epoch0 reference and score routes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from baselines.B5_best.common import read,write,sha
from baselines.B5_best_context.body_features import BODY_PROTOCOL,BEAT16_ENCODING
from baselines.B5_best_context.body_history import B_POLICY
from baselines.B5_best_context.body_prepare import load_cache
from baselines.B5_best_context.body_adapt_prepare import prepare,check_source,PROTOCOL
from baselines.B5_best_context.body_adapt_study import cases
from baselines.B5_best_context.selected_study import SELECTED_B43_SHA256
from baselines.B5_best_context.four_routes import SAMPLING
from baselines.B5_best_context.body_alpha_core import SharedCorrection,FixedAlphaAdapter
from baselines.B5_best_context.body_alpha_study import reference_check
from baselines.B5_best_context.body_study import run_study
from . import test_context


def fixture(root):
    old,new=root/'old',root/'new'
    population={t:{k:t+k for k in ('metadata_sha256','target_sha256','QC_sha256')} for t in ('task1','task2')}
    records={}
    for name,folder in (('old',old),('new',new)):
        rr=[]
        for split in ('train','validation'):
            file=split+'/record.npz'; (folder/split).mkdir(parents=True)
            anchor=np.sin(np.arange(60000)/30).astype(np.float32)*100
            base=anchor*(.7 if name=='old' else .75)
            beat=np.ones((2,300),np.float32)*(2 if name=='old' else 1)
            stats=np.arange(27,dtype=np.float32)*(2 if name=='old' else 1)
            np.savez_compressed(folder/file,anchor=anchor,base_ii=base,target_ii=anchor*.8,
                quality=np.ones(12,bool),beat=beat,stats=stats,available=np.array(True))
            rr.append(dict(split=split,pair_id=split,subject_id=split,target_record_id=split,history_record_id='h_'+split,
                history_visible_sha256='raw_'+split,processed_history_sha256='processed_'+split,
                I_II_eligible=True,history_available=True,file=file,sha256=sha(folder/file)))
        frozen=dict(base_checkpoint_sha256=name+'_base',sampling=SAMPLING,slow_beta=.75,baseline_population=population)
        m=dict(protocol=BODY_PROTOCOL,completed=True,frozen=frozen,records=rr)
        if name=='old':m.update(feature_encoding=BEAT16_ENCODING,history_preprocessing=B_POLICY)
        write(folder/'cache.json',m); records[name]=m
    cp=root/'source.pt';cp.write_bytes(b'synthetic body weight; neural forward is server preflight')
    return old,new,cp


class BodyAdaptTests(unittest.TestCase):
    def test_cache_only_replaces_base_prediction_and_preserves_B_features_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,new,cp=fixture(root)
            args=SimpleNamespace(old_body_cache=old,cache=new,source_checkpoint=cp,output=root/'adapt_cache',resume=False)
            actual_sha=sha
            mock_sha=lambda p:SELECTED_B43_SHA256 if Path(p)==cp else actual_sha(p)
            before={str(p):sha(p) for folder in (old,new) for p in folder.rglob('*') if p.is_file()}
            with patch('baselines.B5_best_context.body_adapt_prepare.sha',side_effect=mock_sha):
                prepare(args)
                m=load_cache(args.output)
                self.assertEqual(m['frozen']['base_checkpoint_sha256'],'new_base')
                for row in m['records']:
                    with np.load(args.output/row['file']) as x,np.load(old/row['file']) as a,np.load(new/row['file']) as b:
                        for key in ('anchor','target_ii','quality','beat','stats','available'):
                            np.testing.assert_array_equal(x[key],a[key])
                        np.testing.assert_array_equal(x['base_ii'],b['base_ii'])
                        self.assertFalse(np.array_equal(x['base_ii'],a['base_ii']))
                hashes={str(p):sha(p) for p in args.output.rglob('*.npz')}
                args.resume=True;prepare(args)
                self.assertEqual(hashes,{str(p):sha(p) for p in args.output.rglob('*.npz')})
            self.assertEqual(before,{str(p):sha(p) for folder in (old,new) for p in folder.rglob('*') if p.is_file()})

    def test_changed_hidden_target_or_raw_history_rejected_even_with_valid_file_hashes(self):
        for mode in ('target','history'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);old,new,cp=fixture(root)
                m=read(new/'cache.json');r=m['records'][0]
                if mode=='history':r['history_visible_sha256']='changed'
                else:
                    path=new/r['file']
                    with np.load(path) as h:a={k:h[k].copy() for k in h.files}
                    a['target_ii'][0]+=1
                    np.savez_compressed(path,**a);r['sha256']=sha(path)
                write(new/'cache.json',m)
                args=SimpleNamespace(old_body_cache=old,cache=new,source_checkpoint=cp,output=root/'adapt',resume=False)
                real_sha=sha
                with patch('baselines.B5_best_context.body_adapt_prepare.sha',side_effect=lambda p:SELECTED_B43_SHA256 if Path(p)==cp else real_sha(p)):
                    with self.assertRaisesRegex(ValueError,'changed'):
                        prepare(args)

    def test_source_contract_prevents_loading_wrong_body_or_dropping_B_policy(self):
        info=dict(protocol=PROTOCOL,source_checkpoint_sha256=SELECTED_B43_SHA256,source_cache_sha256='old_cache',source_base_sha256='old_P')
        cache=dict(body_adaptation=info,feature_encoding=BEAT16_ENCODING,history_preprocessing=B_POLICY,
                   frozen=dict(base_checkpoint_sha256='new_FM',sampling=SAMPLING,slow_beta=.75))
        cp=dict(protocol=BODY_PROTOCOL,group='context',training_seed=43,epoch=15,feature_encoding=BEAT16_ENCODING,
                history_preprocessing=B_POLICY,base_checkpoint_sha256='old_P',cache_sha256='old_cache',sampling=SAMPLING,slow_beta=.75)
        before=deepcopy(cp)
        report=check_source(cp,cache)
        self.assertTrue(report['optimizer_reset']);self.assertEqual(report['target_base_sha256'],'new_FM')
        self.assertEqual(cp,before)
        for key,value in (('epoch',27),('training_seed',42),('cache_sha256','wrong'),('history_preprocessing',{})):
            with self.assertRaises(ValueError):check_source({**cp,key:value},cache)

    def test_complete_scoring_and_epoch0_reproduction_preserve_task1_machine(self):
        class Engine:
            def __init__(self,scale):self.scale=scale
            def __call__(self,p,i,h):p[1]+=self.scale*i[0];return p,{}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);test_context.StudyTests().baseline(root/'base')
            before=sha(root/'base/task2/prediction_uV.npy')
            # Epoch0 fallback must reproduce the current transferred model exactly.
            shared=SharedCorrection(Engine(.02))
            rows=run_study(root/'base',root/'study',lambda r:None,None,None,case_specs=cases(),additive_routes=[],
                adapter_map={k:FixedAlphaAdapter(shared,1.) for k in ('current','adapted')},notes=['synthetic adaptation epoch0'])
            self.assertEqual([r['group'] for r in rows],['baseline','d6_only','current_both','adapted_both'])
            self.assertEqual(rows[2]['mean_r'],rows[3]['mean_r'])
            self.assertEqual(len({r['task1_r'] for r in rows}),1)
            self.assertEqual(rows[1]['machine_r'],rows[3]['machine_r'])
            self.assertEqual(sha(root/'base/task2/prediction_uV.npy'),before)
            records=[dict(pair_id='p',target_record_id='t',alpha=1.,r_II=.78)]
            self.assertTrue(reference_check(records,dict(pairs=records,validation_full120s_II_r=.78))['reproduced'])
            with self.assertRaises(ValueError):reference_check(records,dict(pairs=records,validation_full120s_II_r=.79))


if __name__=='__main__':unittest.main()
