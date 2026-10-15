from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.B5 import ARCHITECTURE_ID, CONDITION_SCHEMA, PREPROCESSING_VERSION
from baselines.B5.checkpoint import scales_digest
from baselines.B5.config import ModelConfig
from baselines.B5.metadata import encode_demographics
from baselines.B5.model import B5UNet
from baselines.B5_best import BUNDLE_PROTOCOL, SCORE_PROTOCOL, SAMPLING
from baselines.B5_best.bundle import export_bundle, load_bundle, validate_bundle
from baselines.B5_best.combine import combine, slow_curve, upsample_native
from baselines.B5_best.common import write, read, sha
from baselines.B5_best.predict import Predictor
from baselines.B5_best.scoring import correlation_score, scores_from_summaries
from baselines.B5_best.select import select
from baselines.B5_best.slow import SlowCNN, predict_native
from baselines.B5_best.validate import evaluate, pair_indices, verify_reference_inputs
from ecg12gen.evaluate import (QC_COMPETITION_SCORE_PROTOCOL, TARGET_LEAD_QC_PROTOCOL,
                              QC_TASK2_DEVICE_AGGREGATION)

torch.set_num_threads(2)


def fixture(root):
    torch.manual_seed(9)
    cfg = ModelConfig(channels=(8,16,32), time_dim=16, metadata_dim=16)
    scales = {"d12": [100.] * 12, "ecg_machine_i": [100.]}
    fm = {"architecture_id":ARCHITECTURE_ID,"condition_schema":CONDITION_SCHEMA,
          "preprocessing_version":PREPROCESSING_VERSION,"architecture_hash":cfg.fingerprint,
          "model_config":asdict(cfg),"scales":scales,"scales_sha256":scales_digest(scales),
          "ema":B5UNet(cfg).state_dict(),"epoch":44}
    fp=root/"fm.pt";torch.save(fm,fp)
    ip=root/"prepared.json"
    write(ip,{"protocol":"B5-slow-probe-v1","completed":True,"scale_uV":scales["d12"],
              "checkpoint_sha256":sha(fp),"slow_width":501,"native_rate_hz":50,"target_means_removed":False})
    slow={"protocol":"B5-slow-probe-v1","config":{"context_seconds":50,"group":"F",
          "model_channels":48,"prepared_sha256":sha(ip)},"model":SlowCNN().state_dict(),"epoch":20}
    sp=root/"slow.pt";torch.save(slow,sp)
    profile={"bundle_protocol":BUNDLE_PROTOCOL,"sampling":dict(SAMPLING),"fm_sha256":sha(fp),
             "slow_sha256":sha(sp),"reference_qc_r":{"task1_r_missing11":.7,"task2_r_missing11":.69}}
    return fp,sp,ip,profile


class HybridTests(unittest.TestCase):
    def test_full_record_filter_and_combination_match_original_probe(self):
        rng=np.random.default_rng(4)
        base=rng.normal(size=(12,60000)).astype(np.float32)
        base[:,30000:]+=500
        predicted=rng.normal(size=(11,60000))
        padded=np.pad(base[1:].astype(float),((0,0),(250,250)),mode="edge")
        acc=np.concatenate((np.zeros((11,1)),np.cumsum(padded,axis=1)),axis=1)
        expected=base.astype(float);expected[1:]=base[1:]-(acc[:,501:]-acc[:,:-501])/501+predicted
        np.testing.assert_array_equal(combine(base,predicted),expected.astype(np.float32))
        np.testing.assert_array_equal(combine(base,predicted)[0],base[0])
        # Full-record filtering must cross the 10s seam instead of resetting there.
        wave=np.zeros((1,60000));wave[:,5000:]=1
        self.assertGreater(slow_curve(wave)[0,4999],0)
        self.assertLess(slow_curve(wave)[0,5000],1)

    def test_interpolation_uses_full_record_block_centers(self):
        native=np.tile(np.arange(6000),(11,1)).astype(float)
        actual=upsample_native(native)
        expected=np.interp(np.arange(60000),np.arange(6000)*10+4.5,native[0])
        np.testing.assert_array_equal(actual[0],expected)
        self.assertAlmostEqual(actual[0,5000]-actual[0,4999],.1)
        with self.assertRaises(ValueError):upsample_native(native[:,:500])

    def test_s50_uses_correct_50s_context_and_center_10s(self):
        class Echo:
            training=False
            def __call__(self,anchor,**metadata):return anchor.repeat(1,11,1)
        fields={k:torch.from_numpy(v[None]) for k,v in encode_demographics({}).items()}
        anchor=torch.arange(6000,dtype=torch.float32).reshape(1,1,6000)
        result=predict_native(Echo(),{"anchor":anchor,**fields})
        torch.testing.assert_close(result,anchor.repeat(1,11,1))

    def test_frozen_cnn_native_inference_equals_manual_windows(self):
        torch.manual_seed(6)
        model=SlowCNN().requires_grad_(False).eval()
        fields={k:torch.from_numpy(v[None]) for k,v in encode_demographics({"age":40}).items()}
        anchor=torch.sin(torch.arange(6000)/30).reshape(1,1,6000)
        with torch.no_grad():
            result=predict_native(model,{"anchor":anchor,**fields})
            padded=torch.cat((anchor[...,:1].expand(-1,-1,1000),anchor,anchor[...,-1:].expand(-1,-1,1000)),dim=-1)
            manual=[]
            for i in range(12):manual.append(model(padded[...,i*500:i*500+2500],**fields)[...,1000:1500])
        torch.testing.assert_close(result,torch.cat(manual,dim=-1),atol=2e-5,rtol=2e-5)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_export_embeds_two_models_and_needs_no_source_files_afterwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fp,sp,ip,profile=fixture(root)
            out=root/"bundle.pt"
            export_bundle(fp,sp,ip,out,profile)
            bundle=load_bundle(out)
            self.assertNotIn("optimizer",bundle["fm"])
            self.assertNotIn("files",bundle["slow"]["config"])
            fp.unlink();sp.unlink();ip.unlink()
            engine=Predictor(load_bundle(out),torch.device("cpu"))
            keys=[]
            def sample_stub(model,inputs,record_keys,**settings):
                self.assertEqual(settings,SAMPLING);keys.extend(record_keys)
                return inputs["anchor"].repeat(1,12,1)
            anchor=(100*np.sin(np.arange(60000)/70)).astype(np.float32)
            with patch("baselines.B5_best.predict.sample",side_effect=sample_stub),patch(
                    "baselines.B5_best.predict.predict_native",return_value=torch.zeros(1,11,6000)):
                result=engine.predict(anchor,{},"new",8)
            expected=combine(np.tile(anchor,(12,1)),np.zeros((11,60000)))
            np.testing.assert_allclose(result,expected,atol=2e-5)
            self.assertEqual(keys,[f"new:{i}" for i in range(0,60000,5000)])
            self.assertTrue(np.isfinite(result).all())
            self.assertEqual(result.shape,(12,60000))
            with self.assertRaises(ValueError):engine.predict(anchor[:-1],{},"new")
            with self.assertRaises(FileExistsError):export_bundle(out,out,out,out,profile)

    def test_bad_sources_scales_and_tampered_bundle_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fp,sp,ip,profile=fixture(root)
            wrong={**profile,"slow_sha256":"wrong"}
            with self.assertRaisesRegex(ValueError,"Stable components"):
                export_bundle(fp,sp,ip,root/"x.pt",wrong)
            bundle=export_bundle(fp,sp,ip,root/"candidate.pt",wrong,candidate=True)
            self.assertIsNone(bundle["reference_qc_r"])
            broken=copy.deepcopy(bundle);broken["fm"]["scales"]["d12"][0]=0
            with self.assertRaises(ValueError):validate_bundle(broken)
            broken=copy.deepcopy(bundle);key=next(iter(broken["slow"]["model"]));broken["slow"]["model"][key].add_(1)
            with self.assertRaisesRegex(ValueError,"digest"):validate_bundle(broken)
            data=read(ip);data["scale_uV"][1]=101;write(ip,data)
            slow=torch.load(sp,weights_only=True);slow["config"]["prepared_sha256"]=sha(ip);torch.save(slow,sp)
            with self.assertRaisesRegex(ValueError,"identical frozen scales"):
                export_bundle(fp,sp,ip,root/"mismatch.pt",profile,candidate=True)

    def test_qc_mean_r_never_uses_rmse_bonus(self):
        qc={"competition_score_protocol":QC_COMPETITION_SCORE_PROTOCOL,
            "quality_mask_applied":True,"target_lead_qc_protocol":TARGET_LEAD_QC_PROTOCOL}
        summaries={"task1":{**qc,"r_missing11":.8},"task2":{**qc,"r_missing11":.6,
                   "task2_device_aggregation":QC_TASK2_DEVICE_AGGREGATION}}
        score=scores_from_summaries(summaries)
        self.assertEqual(score,correlation_score(.8,.6))
        summaries["task2"].update(task2_rmse_bonus_score=10,competition_total_score=99)
        self.assertEqual(scores_from_summaries(summaries),score)
        summaries["task1"]["quality_mask_applied"]=False
        with self.assertRaises(ValueError):scores_from_summaries(summaries)
        with self.assertRaises(FloatingPointError):correlation_score(float("nan"),.6)

    def test_rank_only_same_population_full_hybrid_r(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,a,b in [("a",.8,.6),("b",.75,.7)]:
                write(root/name/"evaluation.json",{"completed":True,"score_protocol":SCORE_PROTOCOL,
                      "population":{"QC":"same"},"sampling":SAMPLING,"checkpoint_sha256":name,
                      "score":correlation_score(a,b)})
            ranked=select([root/"a",root/"b"],root/"rank")
            self.assertEqual(ranked[0]["checkpoint_sha256"],"b")
            report=read(root/"b/evaluation.json");report["population"]={"QC":"different"};write(root/"b/evaluation.json",report)
            with self.assertRaisesRegex(ValueError,"different data"):
                select([root/"a",root/"b"],root/"rank2")


class ValidationTests(unittest.TestCase):
    def test_reference_hash_check_catches_replaced_QC_or_demographics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            current={k:root/k for k in ["demographics_csv","subject_split_csv","device_interpretation_qc_csv",
                    "task1_pair_manifest","task2_pair_manifest"]}
            for task in ["task1","task2"]:
                for suffix in ["window_metadata.csv","validation_target.npy"]:
                    current[task+"_"+suffix.removesuffix(".csv").removesuffix(".npy")]=root/(task+"_"+suffix)
            for key,path in current.items():path.write_text(key,encoding="utf-8")
            bundle={"provenance":{"slow_source_checks":{k:{"sha256":sha(p)} for k,p in current.items()}}}
            common_paths={"subject_split_csv":current["subject_split_csv"],
                          "device_interpretation_qc_csv":current["device_interpretation_qc_csv"],
                          "task1_pair_manifest_csv":current["task1_pair_manifest"],
                          "task2_pair_manifest_csv":current["task2_pair_manifest"],
                          "task1_output":root,"task2_output":root}
            with patch("baselines.B5_best.validate.common_config",return_value=SimpleNamespace(path=lambda k:common_paths[k])),patch(
                    "baselines.B5_best.validate.demographics_path",return_value=current["demographics_csv"]):
                verify_reference_inputs(bundle,{})
                current["device_interpretation_qc_csv"].write_text("changed",encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"device_interpretation_qc_csv"):
                    verify_reference_inputs(bundle,{})

    def test_end_to_end_scoring_reuses_only_identical_visible_records(self):
        raw=np.tile(np.sin(np.arange(60000)/30).astype(np.float32),(12,1))
        class Dataset:
            def __init__(self,task):
                devices=["watch_ecg"] if task=="task1" else ["ecg_machine_d6","body_scale_d6"]
                self.rows=[{"pair_id":dev,"target_record_id":"record","subject_id":"S","input_type":dev,
                            "start_sample_500hz":str(s),"expected_window_count":"12"}
                           for dev in devices for s in range(0,60000,5000)]
                self.base=self;self.demographics=SimpleNamespace(get=lambda _:encode_demographics({}))
            def __len__(self):return len(self.rows)
            def __getitem__(self,i):
                row=self.rows[i];s=int(row["start_sample_500hz"])
                return SimpleNamespace(anchor_i_ecg=raw[:1,s:s+5000]) if self._base else {
                    "target_uV":raw[:,s:s+5000],"quality_mask":np.ones(12,bool),"evaluation_metadata":row}
        # Separate observed base adapter from the label-bearing scoring adapter.
        datasets={}
        for task in ["task1","task2"]:
            ds=Dataset(task);ds._base=False
            ds.base=SimpleNamespace()
            class Base:
                def __init__(self,rows):self.rows=rows
                def __getitem__(self,i):
                    s=int(self.rows[i]["start_sample_500hz"])
                    return SimpleNamespace(anchor_i_ecg=raw[:1,s:s+5000])
            ds.base=Base(ds.rows);datasets[task]=ds
        calls=[]
        class Engine:
            def __init__(self,*args):pass
            def predict_encoded(self,anchor,metadata,key,batch_size):
                calls.append(key);return np.tile(anchor,(12,1))
        bundle={"sampling":SAMPLING,"reference_qc_r":None,"reference_tolerance":2e-5,"provenance":{}}
        with tempfile.TemporaryDirectory() as tmp,patch("baselines.B5_best.validate.Predictor",Engine):
            score=evaluate(bundle,datasets,torch.device("cpu"),Path(tmp)/"eval","bundle")
            self.assertAlmostEqual(score["main_correlation"],1.)
            self.assertEqual(calls,["record"])
            self.assertFalse((Path(tmp)/"eval/competition_score.csv").exists())
            self.assertTrue(read(Path(tmp)/"eval/evaluation.json")["completed"])
            broken=datasets["task1"].rows[:-1]
            with self.assertRaises(ValueError):pair_indices(broken)


if __name__=="__main__":
    unittest.main()
