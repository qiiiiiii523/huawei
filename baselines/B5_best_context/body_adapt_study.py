"""Full-record comparison: frozen new FM+d6, transferred B43, adapted body best."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baselines.B5_best.common import read, write, sha
from .body_prepare import load_cache
from .body_adapt_prepare import PROTOCOL
from .body_ab_study import validate_features
from .body_alpha_study import reference_check
from .body_alpha_core import FixedAlphaAdapter, SharedCorrection
from .body_study import run_study
from .evaluate import load_baseline, HistoryProvider
from .selected_study import SELECTED_B43_SHA256, check_contract
from .body_features import BODY_PROTOCOL, BEAT16_ENCODING, feature_encoding
from .body_history import B_POLICY, history_policy


def cases():
    return [("baseline","none",None,None),("d6_only","fir25",None,None),
            ("current_both","fir25","current",None),("adapted_both","fir25","adapted",None)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    p.add_argument("--cache",required=True)
    p.add_argument("--adapt-run",required=True)
    p.add_argument("--source-checkpoint",required=True)
    p.add_argument("--old-body-cache",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda")
    args=p.parse_args()
    cache=load_cache(args.cache); old_cache=load_cache(args.old_body_cache)
    root=Path(args.adapt_run)
    cfg,done=read(root/"config.json"),read(root/"completed.json")
    if (cfg.get("body_adaptation",{}).get("protocol")!=PROTOCOL
            or cfg["body_adaptation"].get("source_checkpoint_sha256")!=SELECTED_B43_SHA256
            or cfg.get("protocol")!=BODY_PROTOCOL or cfg.get("group")!="context"
            or cfg.get("record_gate") is not None or feature_encoding(cfg)!=BEAT16_ENCODING or history_policy(cfg)!=B_POLICY
            or cfg.get("cache_sha256")!=sha(Path(args.cache)/"cache.json")
            or done.get("completed") is not True or done.get("epochs")!=cfg["epochs"]
            or done.get("training_seed")!=cfg["seed"] or done.get("best_checkpoint_sha256")!=sha(root/"best.pt")
            or sha(args.source_checkpoint)!=SELECTED_B43_SHA256):
        raise ValueError("Complete the matching body adaptation run before evaluation")
    frozen=cache["frozen"]
    loaded=load_baseline(frozen["baseline_evaluation"])
    if loaded[0]["checkpoint_sha256"]!=frozen["base_checkpoint_sha256"] or loaded[2]!=frozen["baseline_population"]:
        raise ValueError("New FM prediction cache/population changed")
    from baselines.B5.runtime import device_from_name
    from .body_model import BodyAdapter
    device=device_from_name(args.device)
    current=BodyAdapter(args.source_checkpoint,old_cache["frozen"]["base_checkpoint_sha256"],device,expected_group="context")
    transfer=check_contract(current.payload,old_cache,sha(Path(args.old_body_cache)/"cache.json"),
                            loaded[0],loaded[2],allow_base_transfer=True)
    adapted=BodyAdapter(root/"best.pt",frozen["base_checkpoint_sha256"],device,expected_group="context")
    cp=adapted.payload
    if (cp.get("body_adaptation")!=cfg["body_adaptation"] or cp.get("cache_sha256")!=cfg["cache_sha256"]
            or cp.get("initial_weights_sha256")!=cfg["initial_weights_sha256"]
            or cp.get("epoch")!=done["best_epoch"] or cp.get("training_seed")!=cfg["seed"]
            or history_policy(cp)!=B_POLICY or feature_encoding(cp)!=BEAT16_ENCODING):
        raise ValueError("Selected adapted checkpoint does not match its training provenance")
    provider=HistoryProvider(args.config,loaded[1]["task2"])
    try:
        validate_features(args.cache,cache,loaded[1]["task2"],provider)
        validate_features(args.old_body_cache,old_cache,loaded[1]["task2"],provider)
        adapters={name:FixedAlphaAdapter(SharedCorrection(engine),1.) for name,engine in (("current",current),("adapted",adapted))}
        summary=run_study(frozen["baseline_evaluation"],args.output,provider,None,None,loaded=loaded,
            case_specs=cases(),adapter_map=adapters,additive_routes=[],
            notes=["Same new FM/P slow predictions and d6 FIR25. Only body II weights may change.",
                   "Current is frozen B43-best15; adapted best also considers the unchanged epoch0.",
                   "B history/O16/alpha1 unchanged; target used only for training supervision and scoring."])
        by_name={r["group"]:r for r in summary}
        if any(r["task1_r"]!=summary[0]["task1_r"] for r in summary):
            raise ValueError("Body adaptation changed Task1")
        if any(by_name[g]["machine_r"]!=by_name["d6_only"]["machine_r"] for g in ("current_both","adapted_both")):
            raise ValueError("Body adaptation contaminated machine route")
        records={}
        for group in ("current_both","adapted_both"):
            with (Path(args.output)/group/"pair_metrics.csv").open(encoding="utf-8-sig",newline="") as h:
                records[group]=[dict(pair_id=r["pair_id"],target_record_id=r["target_record_id"],alpha=1.,r_II=float(r["r_II"]))
                    for r in csv.DictReader(h) if r["task_id"]=="task2" and r["input_type"]=="body_scale_d6"
                    and r["input_I_reliable"]=="True" and r["r_II"]!=""]
        epoch0=reference_check(records["current_both"],read(root/"validation/epoch_000.json"))
        selected=reference_check(records["adapted_both"],read(root/f"validation/epoch_{cp['epoch']:03d}.json"))
        delta=by_name["adapted_both"]["mean_r"]-by_name["current_both"]["mean_r"]
        write(Path(args.output)/"body_adapt_study.json",dict(completed=True,training_started=False,
            FM_updates=0,slow_CNN_updates=0,d6_frozen=True,sampling=frozen["sampling"],slow_beta=.75,body_alpha=1.,
            base_checkpoint_sha256=frozen["base_checkpoint_sha256"],source_body_sha256=SELECTED_B43_SHA256,
            adapted_checkpoint=str((root/"best.pt").resolve()),adapted_checkpoint_sha256=sha(root/"best.pt"),
            best_epoch=cp["epoch"],additional_epoch_budget=cfg["epochs"],training_config=cfg,
            source_transfer=transfer,epoch0_reference=epoch0,selected_reference=selected,
            population=loaded[2],summary=summary,adaptation_delta_mean_r=delta,
            adapted_best_improved=delta>1e-8))
    finally:
        provider.close()
    print("Completed: "+str(Path(args.output)/"summary.csv"),flush=True)


if __name__=="__main__": main()
