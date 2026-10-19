"""Selected model identity, raw-history policy, reference scope and route invariance."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace
import sys

from baselines.B5_best.common import read, sha, write
from baselines.B5_final.profile import (BASE_SHA, BODY_SHA, SOURCE_BODY_SHA, BODY_CACHE_SHA,
    DEFAULT_PROFILE, SAMPLING, load_profile, resolve_assets, check_payloads)
from baselines.B5_final.validate import cases, reference_check, check_baseline
from baselines.B5_best_context.body_features import BODY_PROTOCOL, BEAT16_ENCODING
from baselines.B5_best_context.body_history import B_POLICY, A_POLICY
from baselines.B5_best_context.tests.test_context import StudyTests
from baselines.B5_best_context.body_alpha_core import FixedAlphaAdapter, SharedCorrection
from baselines.B5_best_context.body_study import run_study
from baselines.B5_best_context.evaluate import load_baseline


class FinalTests(unittest.TestCase):
    def payload(self):
        return dict(protocol=BODY_PROTOCOL, group="context", training_seed=43, epoch=3,
            base_checkpoint_sha256=BASE_SHA, cache_sha256=BODY_CACHE_SHA, sampling=SAMPLING,
            slow_beta=.75, feature_encoding=BEAT16_ENCODING, history_preprocessing=deepcopy(B_POLICY),
            body_adaptation=dict(protocol="B5-body-new-base-adaptation-v1",
                                 source_checkpoint_sha256=SOURCE_BODY_SHA, target_base_sha256=BASE_SHA))

    def test_profile_rejects_old_body_or_silent_setting_changes(self):
        profile = load_profile()
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)/"profile.json"
            for key, value in (("sampling", {**SAMPLING, "seed":43}), ("slow_beta",1.),
                               ("body_alpha",.75), ("machine_mode","none")):
                write(path,{**profile,key:value})
                with self.assertRaises(ValueError): load_profile(path)
            changed=deepcopy(profile); changed["body"]["sha256"]=SOURCE_BODY_SHA
            write(path,changed)
            with self.assertRaises(ValueError): load_profile(path)

    def test_weight_paths_can_move_but_hash_cannot(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); base=root/"model.pt"; body=root/"body.pt"
            base.write_bytes(b"test base"); body.write_bytes(b"test body")
            def digest(path): return BASE_SHA if Path(path)==base else BODY_SHA
            with patch("baselines.B5_final.profile.sha",side_effect=digest):
                assets=resolve_assets(root,checkpoint="model.pt",body_checkpoint="body.pt")
            self.assertEqual(assets.base,base.resolve())
            self.assertEqual(assets.body,body.resolve())
            self.assertEqual(assets.report()["sampling"],SAMPLING)
            with self.assertRaisesRegex(ValueError,"SHA256"):
                resolve_assets(root,checkpoint=base,body_checkpoint=body)
            with self.assertRaises(FileNotFoundError):
                resolve_assets(root,checkpoint=root/"missing.pt",body_checkpoint=body)

    def test_payload_pins_adapted_not_transferred_or_seed42(self):
        bundle=dict(sampling=SAMPLING,slow_beta=.75)
        payload=self.payload()
        check_payloads(bundle,payload)
        for key,value in (("epoch",15),("training_seed",42),("cache_sha256","old_cache"),
                          ("history_preprocessing",A_POLICY),("record_gate","G"),
                          ("base_checkpoint_sha256","old_P")):
            with self.assertRaises(ValueError): check_payloads(bundle,{**payload,key:value})
        with self.assertRaises(ValueError): check_payloads({**bundle,"slow_beta":1.},payload)
        with self.assertRaises(ValueError): check_payloads(bundle,{**payload,"body_adaptation":{}})

    def test_reference_requires_same_population_and_all_device_scores(self):
        profile=load_profile(); ref=profile["reference"]
        row={k:ref[k] for k in ("task1_r","machine_r","body_r","task2_r","mean_r")}
        self.assertTrue(reference_check(profile,ref["population"],row)["matched"])
        pop=deepcopy(ref["population"]); pop["task2"]["QC_sha256"]="other"
        self.assertFalse(reference_check(profile,pop,row)["matched"])
        self.assertFalse(reference_check(profile,ref["population"],{**row,"machine_r":row["machine_r"]+.001})["matched"])
        report=dict(checkpoint_sha256=BASE_SHA,sampling=SAMPLING,slow_beta=.75)
        check_baseline(report)
        for change in ({"checkpoint_sha256":"old"},{"sampling":{**SAMPLING,"seed":43}}):
            with self.assertRaises(ValueError): check_baseline({**report,**change})

    def test_actual_four_routes_keep_base_and_other_device_outputs_unchanged(self):
        class Engine:
            def __call__(self,p,i,h):
                p[1] += .02*i[0]
                return p,{}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); StudyTests().baseline(root/"base")
            loaded=load_baseline(root/"base")
            before=sha(root/"base/task2/prediction_uV.npy")
            try:
                rows=run_study(root/"base",root/"out",lambda r:None,None,None,loaded=loaded,
                    case_specs=cases(),adapter_map={"body":FixedAlphaAdapter(SharedCorrection(Engine()),1.)})
                self.assertEqual([r["group"] for r in rows],["baseline","d6_only","body_only","both"])
                self.assertEqual(len({r["task1_r"] for r in rows}),1)
                self.assertEqual(rows[0]["body_r"],rows[1]["body_r"])
                self.assertEqual(rows[2]["body_r"],rows[3]["body_r"])
                self.assertEqual(rows[1]["machine_r"],rows[3]["machine_r"])
                self.assertNotEqual(rows[0]["body_r"],rows[2]["body_r"])
                self.assertEqual(before,sha(root/"base/task2/prediction_uV.npy"))
            finally:
                for item in loaded[1].values():
                    for key in ("prediction","target","quality"): item[key]._mmap.close()

    def test_validate_entry_wires_selected_body_and_writes_final_provenance(self):
        from baselines.B5_final.validate import main
        payload=self.payload()
        class Body:
            def __init__(self,*args,**kwargs): self.payload=payload
            def __call__(self,p,i,h):
                p[1] += .02*i[0]
                return p,{}
        closed=[]
        class Provider:
            def __init__(self,*args): pass
            def __call__(self,row): return None
            def close(self): closed.append(True)
        runtime=ModuleType("baselines.B5.runtime"); runtime.device_from_name=lambda x:x
        model=ModuleType("baselines.B5_best_context.body_model"); model.BodyAdapter=Body
        bundle=ModuleType("baselines.B5_best.bundle")
        bundle.load_bundle=lambda p:dict(sampling=SAMPLING,slow_beta=.75)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); StudyTests().baseline(root/"base")
            report=read(root/"base/evaluation.json"); report["checkpoint_sha256"]=BASE_SHA
            write(root/"base/evaluation.json",report)
            class FakeAssets:
                base=root/"selected.pt"
                body=root/"body.pt"
                validation_config=root/"config.yaml"
                baseline_evaluation=root/"base"
                profile=load_profile()
                def report(self): return dict(base_sha256=BASE_SHA,body_sha256=BODY_SHA)
            args=["validate","--output",str(root/"study"),"--device","cpu"]
            with patch.dict(sys.modules,{runtime.__name__:runtime,model.__name__:model,bundle.__name__:bundle}), \
                    patch("baselines.B5_final.validate.assets_from_args",return_value=FakeAssets()), \
                    patch("baselines.B5_best_context.evaluate.HistoryProvider",Provider),patch.object(sys,"argv",args):
                main()
            result=read(root/"study/final.json")
            self.assertTrue(result["completed"])
            self.assertEqual(result["body_sha256"],BODY_SHA)
            self.assertEqual([r["group"] for r in result["summary"]],["baseline","d6_only","body_only","both"])
            self.assertFalse(result["reference_check"]["matched"])
            self.assertEqual(closed,[True])


if __name__ == "__main__": unittest.main()
