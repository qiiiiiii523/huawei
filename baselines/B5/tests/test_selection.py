"""R-only policy replaces the retired 60-point checkpoint policy in this branch."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from baselines.B5.scoring import SCORE_PROTOCOL, correlation_score
from baselines.B5.selection import (SELECTION_PROTOCOL, empty_best_metrics, update_best_metrics,
                                   restore_best_metrics, save_selected_checkpoints)
from baselines.B5.config import load_config


def payload(best, epoch=0):
    return {"selection_protocol": SELECTION_PROTOCOL, "score_protocol": SCORE_PROTOCOL,
            "selection_metric": "main_correlation", "best_metrics": best, "checkpoint_role": "last",
            "best_score": None if best["main_correlation"] is None else best["main_correlation"]["value"],
            "epoch": epoch, "model": {"x": torch.tensor(epoch)}, "ema": {"x": torch.tensor(epoch)}}


class SelectionTests(unittest.TestCase):
    def test_only_best_r_and_last_are_saved_and_resume_preserves_best(self):
        best = empty_best_metrics()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_selected_checkpoints(out, payload(best), [])
            self.assertEqual({p.name for p in out.iterdir()}, {"last.pt"})
            for epoch, a, b in [(1,.8,.6),(2,.7,.9),(3,.9,.5)]:
                best, improved = update_best_metrics(best, correlation_score(a,b), epoch)
                save_selected_checkpoints(out, payload(best,epoch), improved)
            self.assertEqual({p.name for p in out.iterdir()}, {"best_r.pt","last.pt"})
            self.assertEqual(torch.load(out/"best_r.pt",weights_only=True)["epoch"],2)
            last = torch.load(out/"last.pt",weights_only=True)
            self.assertEqual(last["epoch"],3)
            self.assertEqual(restore_best_metrics(last),best)

    def test_ties_and_rmse_fields_do_not_change_selection(self):
        s=correlation_score(.8,.6)
        best,_=update_best_metrics(empty_best_metrics(),s,1)
        original=copy.deepcopy(best)
        s.update(competition_total_score=60,task2_rmse_bonus_score=10)
        best,changed=update_best_metrics(best,s,2)
        self.assertEqual(best,original);self.assertEqual(changed,[])

    def test_legacy_and_selected_checkpoints_cannot_exact_resume(self):
        with self.assertRaisesRegex(ValueError,"Legacy"):
            restore_best_metrics({"selection_metric":"competition_total_score"})
        cp=payload(empty_best_metrics())
        with self.assertRaises(ValueError):restore_best_metrics({**cp,"checkpoint_role":"main_correlation"})
        with self.assertRaises(ValueError):restore_best_metrics({**cp,"best_score":35})
        with self.assertRaises(ValueError):update_best_metrics(empty_best_metrics(),{"main_correlation":.8},1)
        with self.assertRaises(FloatingPointError):
            update_best_metrics(empty_best_metrics(),{"score_protocol":SCORE_PROTOCOL,"main_correlation":float("nan")},1)

    def test_new_configs_select_mean_r_and_use_separate_outputs(self):
        root=Path(__file__).resolve().parents[3]
        for name in ["b5_main_v2_scratch_meta.yaml","b5_main_v2_public_finetune_meta.yaml"]:
            cfg=load_config(root/"configs/experiments"/name)
            self.assertEqual(cfg["validation"]["selection_metric"],"main_correlation")
            self.assertIn("results/B5/",cfg["paths"]["output_dir"])
            self.assertEqual(cfg["sampling"]["samples"],16)

    def test_old_checkout_config_is_read_only_data_input_not_legacy_resume(self):
        root=Path(__file__).resolve().parents[3]
        path=root/"configs/experiments/b5_finetune_meta.yaml"
        old=load_config(path)
        old["validation"]["selection_metric"]="competition_total_score"
        with patch("baselines.B5.config._read",side_effect=lambda *args:copy.deepcopy(old)):
            with self.assertRaisesRegex(ValueError,"main_correlation"):load_config(path)
            self.assertEqual(load_config(path,data_only=True)["validation"]["selection_metric"],"main_correlation")
        self.assertEqual(old["validation"]["selection_metric"],"competition_total_score")


if __name__=="__main__":unittest.main()
