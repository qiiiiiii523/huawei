#!/usr/bin/env bash
# New FM frozen predictions -> same B features -> B43 warm start -> ten extra epochs -> compare.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
step=all
gpu=1
run=
checkpoint=
baseline=
source=
old_cache=
config=
resume=0
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --step) step="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --baseline-evaluation) baseline="$2"; shift 2 ;;
    --source-checkpoint) source="$2"; shift 2 ;;
    --old-body-cache) old_cache="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$step" in all|prepare|train|evaluate) ;; *) echo 'Use --step all|prepare|train|evaluate' >&2; exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
[[ -n "$run" ]] || run="$repository/results/B5-best-context/newFM_body_adapt_v1"
[[ -n "$checkpoint" ]] || checkpoint="$repository/results/B5-best-exp-PublicFM/cross_P_20261019_185724_3338378/bundles/FM_P_best45__P.pt"
[[ -n "$baseline" ]] || baseline="$repository/results/B5-best-context/FM_P_best45_B43_v1/base_inference"
[[ -n "$source" ]] || source="$repository/results/B5-best-context/P_body_AB_seed43_v1/B/best.pt"
[[ -n "$old_cache" ]] || old_cache="$repository/results/B5-best-context/P_body_AB_v1/prepared/B_cache"
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
for f in "$checkpoint" "$baseline/evaluation.json" "$source" "$old_cache/cache.json" "$config"; do
  test -f "$f" || { echo "Missing: $f" >&2; exit 1; }
done
if [[ "$step" == all || "$step" == prepare ]]; then
  if [[ -e "$run" && "$resume" == 0 ]]; then echo "Use new --run or --resume: $run" >&2; exit 1; fi
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=setup
trap 'echo "Stopped: phase=$phase line=$LINENO; repeat same step/settings with --resume: $run" >&2' ERR
python - "$run" "$checkpoint" "$baseline" "$source" "$old_cache" "$config" "$step" <<'PY'
from pathlib import Path
import sys
from baselines.B5_best.common import read,write,sha
from baselines.B5_best_context.selected_study import SELECTED_B43_SHA256
run,base,baseline,source,old,config=map(Path,sys.argv[1:7])
step=sys.argv[7]
if sha(source)!=SELECTED_B43_SHA256:
    raise ValueError("Warm-start from selected B43-best15 only")
report=read(baseline/"evaluation.json")
if report.get("completed") is not True or report["checkpoint_sha256"]!=sha(base):
    raise ValueError("Use completed validation from the new complete FM_P_best45__P bundle")
value=dict(checkpoint=str(base.resolve()),checkpoint_sha256=sha(base),
    baseline_evaluation=str(baseline.resolve()),baseline_report_sha256=sha(baseline/"evaluation.json"),
    source_checkpoint=str(source.resolve()),source_sha256=sha(source),
    old_body_cache=str(old.resolve()),old_cache_sha256=sha(old/"cache.json"),
    config=str(config.resolve()),config_sha256=sha(config),seed=43,additional_epochs=10,
    learning_rate=.0001,batch_size=16,body_alpha=1.,optimizer_reset=True,epoch0_retained=True)
if (run/"run.json").exists():
    if read(run/"run.json")!=value: raise ValueError("Adaptation input/settings changed; use new run")
elif step in ("all","prepare"):
    write(run/"run.json",value)
else:
    raise ValueError("First run --step prepare")
PY
if [[ "$step" == prepare || "$step" == all ]]; then
  phase=newFM_training_predictions
  extra=(); if [[ "$resume" == 1 ]]; then extra+=(--resume); fi
  python -u -m baselines.B5_best_context.body_prepare \
    --checkpoint "$checkpoint" --config "$config" --baseline-evaluation "$baseline" \
    --output "$run/raw_cache" --device cuda "${extra[@]}" \
    2>&1 | tee -a "$run/prepare_predictions.log"
  phase=reuse_B_features
  python -u -m baselines.B5_best_context.body_adapt_prepare \
    --cache "$run/raw_cache" --old-body-cache "$old_cache" --source-checkpoint "$source" \
    --output "$run/B_cache" "${extra[@]}" \
    2>&1 | tee -a "$run/prepare_B.log"
  phase=warm_start_preflight
  preflight="$run/preflight"
  if [[ -f "$run/preflight_path.txt" ]]; then preflight=$(cat "$run/preflight_path.txt"); fi
  if [[ ! -f "$preflight/preflight.json" ]]; then
    if [[ -e "$preflight" ]]; then preflight="$run/preflight_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
    printf '%s\n' "$preflight" > "$run/preflight_path.txt"
    python -u -m baselines.B5_best_context.body_train \
      --cache "$run/B_cache" --output "$preflight" --group context --adapt-body-checkpoint "$source" \
      --seed 43 --epochs 10 --batch-size 16 --learning-rate .0001 --check-only \
      2>&1 | tee -a "$run/preflight.log"
  fi
  printf '%s\n' "$preflight" > "$run/preflight_path.txt"
fi
if [[ "$step" == train || "$step" == all ]]; then
  test -f "$(cat "$run/preflight_path.txt")/preflight.json"
  phase=adapt_body_only
  extra=(); if [[ "$resume" == 1 && -f "$run/adapt/config.json" ]]; then extra+=(--resume); fi
  python -u -m baselines.B5_best_context.body_train \
    --cache "$run/B_cache" --output "$run/adapt" --group context --adapt-body-checkpoint "$source" \
    --seed 43 --epochs 10 --batch-size 16 --learning-rate .0001 --device cuda "${extra[@]}" \
    2>&1 | tee -a "$run/train.log"
fi
if [[ "$step" == evaluate || "$step" == all ]]; then
  phase=full_record_evaluation
  study="$run/study"
  if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
  if [[ ! -f "$study/body_adapt_study.json" ]]; then
    if [[ -e "$study" ]]; then study="$run/study_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
    printf '%s\n' "$study" > "$run/study_path.txt"
    python -u -m baselines.B5_best_context.body_adapt_study \
      --config "$config" --cache "$run/B_cache" --adapt-run "$run/adapt" \
      --source-checkpoint "$source" --old-body-cache "$old_cache" --output "$study" --device cuda \
      2>&1 | tee -a "$run/evaluate.log"
  fi
  cat "$study/summary.csv"
fi
echo "Completed step=$step: $run"
