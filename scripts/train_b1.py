"""Train protocol-native B1 P0 or P1-C3."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ecg12gen.b1_train import train_b1

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(ROOT/"configs/common.yaml")); p.add_argument("--task-id",choices=("task1","task2"),required=True); p.add_argument("--stage",choices=("P0_anchor_only","P1-C3"),required=True); p.add_argument("--fusion-mode",choices=("none","film_gated_residual"),default="none"); p.add_argument("--p0-checkpoint"); p.add_argument("--context-source-type",choices=("watch_ecg","ecg_machine_d6","body_scale_d6")); p.add_argument("--body-scale-variant",choices=("A_raw_window","B_detrend_0p2Hz_then_window"),default="A_raw_window"); p.add_argument("--variant",choices=("base","wide","dilated","core7"),default="base"); p.add_argument("--baseline-weight",type=float,default=.10); p.add_argument("--output-dir",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--epochs",type=int,default=60); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--dropout",type=float,default=.10); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=5e-4); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    if a.variant=="core7" and a.stage!="P0_anchor_only": p.error("core7 is enabled for P0 only")
    if a.task_id=="task1" and a.stage=="P1-C3" and a.context_source_type!="watch_ecg": p.error("task1 P1 requires watch_ecg")
    if a.task_id=="task2" and a.stage=="P1-C3" and a.context_source_type not in {"ecg_machine_d6","body_scale_d6"}: p.error("task2 P1 requires one d6 source")
    print(f"B1 complete: {train_b1(a)}")
if __name__=="__main__": main()
