#!/usr/bin/env bash
# Frozen B5-best -> paired II adapter training -> four routes + context controls.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
run=
baseline=/home/qht/huawei/results/B5-best-context/fir25_20261017_225742/base_inference
gpu=1
epochs=10
seed=42
resume=0
config=
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --baseline-evaluation) baseline="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ && "$epochs" =~ ^[1-9][0-9]*$ && "$seed" =~ ^[0-9]+$ ]] || exit 2
[[ -n "$run" ]] || run="$repository/results/B5-best-context/body_v1_$(date +%Y%m%d_%H%M%S)_$$"
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
test -f "$config"
test -f "$baseline/evaluation.json"
if [[ "$resume" == 0 && -e "$run" ]]; then
  echo "Use a new --run or explicit --resume: $run" >&2
  exit 1
fi
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=prepare
trap 'echo "Stopped: phase=$phase line=$LINENO; use the SAME --run with --resume: $run" >&2' ERR
extra=()
[[ "$resume" == 0 ]] || extra+=(--resume)
python -u -m baselines.B5_best_context.body_prepare \
  --config "$config" --baseline-evaluation "$baseline" \
  --output "$run/cache" --device cuda "${extra[@]}" \
  2>&1 | tee -a "$run/prepare.log"
if [[ ! -f "$run/preflight/preflight.json" ]]; then
  phase=preflight
  preflight="$run/preflight"
  if [[ -d "$preflight" ]]; then preflight="$run/preflight_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_train \
    --cache "$run/cache" --group context --output "$preflight" \
    --epochs "$epochs" --seed "$seed" --check-only \
    2>&1 | tee -a "$run/preflight.log"
  if [[ "$preflight" != "$run/preflight" ]]; then
    cp "$preflight/preflight.json" "$run/preflight/preflight.json"
  fi
fi
for group in no_context context; do
  phase="train_$group"
  train_extra=()
  if [[ "$resume" == 1 && -f "$run/$group/config.json" ]]; then train_extra+=(--resume); fi
  python -u -m baselines.B5_best_context.body_train \
    --cache "$run/cache" --group "$group" --output "$run/$group" \
    --epochs "$epochs" --seed "$seed" --device cuda "${train_extra[@]}" \
    2>&1 | tee -a "$run/${group}.log"
done
phase=evaluation
study="$run/study"
if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
if [[ ! -f "$study/study.json" ]]; then
  # A partial evaluation is retained. Restart evaluation in a new directory.
  if [[ -d "$study" ]]; then study="$run/study_resume_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_study \
    --config "$config" --cache "$run/cache" \
    --context-checkpoint "$run/context/best.pt" \
    --no-context-checkpoint "$run/no_context/best.pt" \
    --output "$study" --device cuda \
    2>&1 | tee -a "$run/study.log"
fi
printf '%s\n' "$study" > "$run/study_path.txt"
cat "$study/summary.csv"
echo "Completed. Run: $run"
echo "Study: $study"
