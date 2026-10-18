#!/usr/bin/env bash
# Reuse body_v1 cache. Plot old weights, train seed43 pair, evaluate, compare and plot.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
source_run=
config=
run=
gpu=1
seed=43
resume=0
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --source-run) source_run="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]] || exit 2
[[ -n "$source_run" ]] || source_run="$repository/results/B5-best-context/body_v1"
[[ -n "$run" ]] || run="$repository/results/B5-best-context/body_seed${seed}_v1"
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
cache="$source_run/cache"
old_study="$source_run/study"
if [[ -f "$source_run/study_path.txt" ]]; then old_study=$(cat "$source_run/study_path.txt"); fi
test -f "$cache/cache.json"
test -f "$old_study/body_checkpoints.json"
test -f "$config"
if [[ "$resume" == 0 && -e "$run" ]]; then
  echo "Use a new --run or explicit --resume: $run" >&2; exit 1
fi
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=original_plots
trap 'echo "Stopped: phase=$phase line=$LINENO; repeat SAME --run/--seed with --resume: $run" >&2' ERR
old_plots="$run/seed42_plots"
if [[ -f "$run/original_plots_path.txt" ]]; then old_plots=$(cat "$run/original_plots_path.txt"); fi
if [[ ! -f "$old_plots/diagnostics.json" ]]; then
  if [[ -d "$old_plots" ]]; then old_plots="$run/seed42_plots_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_diagnose \
    --cache "$cache" --study "$old_study" --output "$old_plots" --device cuda \
    2>&1 | tee -a "$run/original_plots.log"
fi
printf '%s\n' "$old_plots" > "$run/original_plots_path.txt"
if [[ ! -f "$run/preflight/preflight.json" ]]; then
  phase=preflight
  preflight="$run/preflight"
  if [[ -d "$preflight" ]]; then preflight="$run/preflight_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_train \
    --cache "$cache" --group context --seed "$seed" --epochs 10 \
    --output "$preflight" --check-only \
    2>&1 | tee -a "$run/preflight.log"
  if [[ "$preflight" != "$run/preflight" ]]; then cp "$preflight/preflight.json" "$run/preflight/preflight.json"; fi
fi
for group in no_context context; do
  phase="train_$group"
  extra=()
  if [[ "$resume" == 1 && -f "$run/$group/config.json" ]]; then extra+=(--resume); fi
  python -u -m baselines.B5_best_context.body_train \
    --cache "$cache" --group "$group" --seed "$seed" --epochs 10 \
    --output "$run/$group" --device cuda "${extra[@]}" \
    2>&1 | tee -a "$run/${group}.log"
done
phase=evaluation
study="$run/study"
if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
if [[ ! -f "$study/study.json" ]]; then
  if [[ -d "$study" ]]; then study="$run/study_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_study \
    --config "$config" --cache "$cache" \
    --context-checkpoint "$run/context/best.pt" \
    --no-context-checkpoint "$run/no_context/best.pt" \
    --output "$study" --device cuda \
    2>&1 | tee -a "$run/study.log"
fi
printf '%s\n' "$study" > "$run/study_path.txt"
phase=comparison
comparison="$run/comparison"
if [[ -f "$run/comparison_path.txt" ]]; then comparison=$(cat "$run/comparison_path.txt"); fi
if [[ ! -f "$comparison/comparison.json" ]]; then
  if [[ -d "$comparison" ]]; then comparison="$run/comparison_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_compare \
    --before-study "$old_study" --after-study "$study" --output "$comparison" \
    2>&1 | tee -a "$run/comparison.log"
fi
printf '%s\n' "$comparison" > "$run/comparison_path.txt"
phase=new_plots
new_plots="$run/new_seed_plots"
if [[ -f "$run/new_plots_path.txt" ]]; then new_plots=$(cat "$run/new_plots_path.txt"); fi
if [[ ! -f "$new_plots/diagnostics.json" ]]; then
  if [[ -d "$new_plots" ]]; then new_plots="$run/new_seed_plots_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_diagnose \
    --cache "$cache" --study "$study" --output "$new_plots" --device cuda \
    2>&1 | tee -a "$run/new_plots.log"
fi
printf '%s\n' "$new_plots" > "$run/new_plots_path.txt"
cat "$comparison/seed_comparison.csv"
echo "Completed: $run"
echo "Original plots: $old_plots"
echo "New plots: $new_plots"
