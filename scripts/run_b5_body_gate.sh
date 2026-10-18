#!/usr/bin/env bash
# body-only C/G continuation + read-only clipping audit. B5/d6 stay fixed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
source_run=
config=
run=
gpu=1
seed=42
epochs=10
resume=0
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --source-run) source_run="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ && "$epochs" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -n "$source_run" ]] || source_run="$repository/results/B5-best-context/body_v1"
[[ -n "$run" ]] || run="$repository/results/B5-best-context/body_gate_v1"
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
cache="$source_run/cache"
source_checkpoint="$source_run/context/best.pt"
test -f "$cache/cache.json"
test -f "$source_checkpoint"
test -f "$config"
if [[ "$resume" == 0 && -e "$run" ]]; then echo "Use a new --run or explicit --resume: $run" >&2; exit 1; fi
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=clipping_audit
trap 'echo "Stopped: phase=$phase line=$LINENO; repeat SAME --run/--seed/--epochs with --resume: $run" >&2' ERR
audit="$run/clipping_audit"
if [[ -f "$run/audit_path.txt" ]]; then audit=$(cat "$run/audit_path.txt"); fi
if [[ ! -f "$audit/audit.json" ]]; then
  if [[ -d "$audit" ]]; then audit="$run/clipping_audit_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_clip_audit \
    --config "$config" --cache "$cache" --output "$audit" \
    2>&1 | tee -a "$run/clipping_audit.log"
fi
printf '%s\n' "$audit" > "$run/audit_path.txt"
for mode in C G; do
  phase="preflight_$mode"
  preflight="$run/preflight_$mode"
  if [[ ! -f "$preflight/preflight.json" ]]; then
    if [[ -d "$preflight" ]]; then preflight="$run/preflight_${mode}_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
    python -u -m baselines.B5_best_context.body_train \
      --cache "$cache" --group context --record-gate "$mode" \
      --init-body-checkpoint "$source_checkpoint" --seed "$seed" --epochs "$epochs" \
      --output "$preflight" --check-only \
      2>&1 | tee -a "$run/preflight_${mode}.log"
    if [[ "$preflight" != "$run/preflight_$mode" ]]; then cp "$preflight/preflight.json" "$run/preflight_$mode/preflight.json"; fi
  fi
done
for mode in C G; do
  phase="train_$mode"
  extra=()
  if [[ "$resume" == 1 && -f "$run/$mode/config.json" ]]; then extra+=(--resume); fi
  python -u -m baselines.B5_best_context.body_train \
    --cache "$cache" --group context --record-gate "$mode" \
    --init-body-checkpoint "$source_checkpoint" --seed "$seed" --epochs "$epochs" \
    --output "$run/$mode" --device cuda "${extra[@]}" \
    2>&1 | tee -a "$run/${mode}.log"
done
phase=evaluation
study="$run/study"
if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
if [[ ! -f "$study/gate_checkpoints.json" ]]; then
  if [[ -d "$study" ]]; then study="$run/study_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_gate_study \
    --config "$config" --cache "$cache" --source-checkpoint "$source_checkpoint" \
    --control-checkpoint "$run/C/best.pt" --gate-checkpoint "$run/G/best.pt" \
    --output "$study" --device cuda \
    2>&1 | tee -a "$run/study.log"
fi
printf '%s\n' "$study" > "$run/study_path.txt"
cat "$study/summary.csv"
cat "$study/gate_effect.json"
echo "Completed: $run"
echo "Clipping audit: $audit"
echo "C/G study: $study"
