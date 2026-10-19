#!/usr/bin/env bash
# Three independent checks: E24 alpha; O16 total30; seed43 O/E total30.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
root=
step=
e_run=
o30_run=
gpu=1
resume=0
while (($#)); do
  case "$1" in
    --step) step="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --repository-root) repository="$2"; shift 2 ;;
    --run-root) root="$2"; shift 2 ;;
    --e-run) e_run="$2"; shift 2 ;;
    --o30-run) o30_run="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$step" == alpha || "$step" == o16 || "$step" == seed43 ]] || { echo 'Use --step alpha|o16|seed43' >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
[[ -n "$root" ]] || root="$repository/results/B5-best-context/body_verification_v1"
run="$root/$step"
if [[ -e "$run" && "$resume" == 0 ]]; then echo "Use a new --run-root or --resume: $run" >&2; exit 1; fi
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=resolve_sources
trap 'echo "Stopped: step=$step phase=$phase line=$LINENO; repeat SAME command with --resume: $run" >&2' ERR
extra=()
if [[ -n "$e_run" ]]; then extra+=(--e-run "$e_run"); fi
if [[ -n "$o30_run" ]]; then extra+=(--o30-run "$o30_run"); fi
python -u -m baselines.B5_best_context.body_verify_prepare paths \
  --repository-root "$repository" --output "$run/paths.json" "${extra[@]}" \
  2>&1 | tee -a "$run/setup.log"
IFS=$'\t' read -r cache encoded_cache clip16_cache e_run o30_run o16_source config < "$run/paths.tsv"
case "$step" in
  alpha)
    marker=alpha_checkpoints.json
    ;;
  o16)
    phase=stage_extension
    extra=()
    if [[ "$resume" == 1 ]]; then extra+=(--resume); fi
    python -u -m baselines.B5_best_context.body_verify_prepare extend \
      --source-run "$o16_source" --cache "$clip16_cache" --epochs 30 \
      --output "$run/O16" "${extra[@]}" 2>&1 | tee -a "$run/extension.log"
    phase=check_matched_budget
    python - "$o30_run" "$run/O16" "$cache" "$clip16_cache" <<'PY'
from pathlib import Path
import sys
from baselines.B5_best.common import read, sha
from baselines.B5_best_context.body_features import BEAT16_ENCODING
from baselines.B5_best_context.body_encoding_study import check_training_pair
control, experiment, cache, encoded = map(Path, sys.argv[1:])
check_training_pair(read(control / "config.json"), read(experiment / "config.json"),
    sha(cache / "cache.json"), sha(encoded / "cache.json"), expected_encoding=BEAT16_ENCODING)
PY
    phase=extend_O16
    python -u -m baselines.B5_best_context.body_train \
      --cache "$clip16_cache" --group context --seed 42 --epochs 30 \
      --batch-size 16 --learning-rate 0.0002 --output "$run/O16" --device cuda --resume \
      2>&1 | tee -a "$run/O16.log"
    marker=encoding_checkpoints.json
    ;;
  seed43)
    for group in O E; do
      phase="preflight_$group"
      selected_cache="$cache"
      if [[ "$group" == E ]]; then selected_cache="$encoded_cache"; fi
      preflight="$run/preflight_$group"
      if [[ -f "$run/preflight_${group}_path.txt" ]]; then preflight=$(cat "$run/preflight_${group}_path.txt"); fi
      if [[ ! -f "$preflight/preflight.json" ]]; then
        if [[ -d "$preflight" ]]; then preflight="$run/preflight_${group}_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
        python -u -m baselines.B5_best_context.body_train \
          --cache "$selected_cache" --group context --seed 43 --epochs 30 \
          --batch-size 16 --learning-rate 0.0002 --output "$preflight" --check-only \
          2>&1 | tee -a "$run/preflight_${group}.log"
      fi
      printf '%s\n' "$preflight" > "$run/preflight_${group}_path.txt"
    done
    phase=pair_preflight
    python -u -m baselines.B5_best_context.body_encoding_study \
      --config "$config" --cache "$cache" --encoded-cache "$encoded_cache" \
      --control-run "$(cat "$run/preflight_O_path.txt")" --encoded-run "$(cat "$run/preflight_E_path.txt")" \
      --output "$run/preflight_pair.json" --check-only 2>&1 | tee -a "$run/preflight_pair.log"
    for group in O E; do
      phase="train_seed43_$group"
      selected_cache="$cache"
      if [[ "$group" == E ]]; then selected_cache="$encoded_cache"; fi
      extra=()
      if [[ "$resume" == 1 && -f "$run/$group/config.json" ]]; then extra+=(--resume); fi
      python -u -m baselines.B5_best_context.body_train \
        --cache "$selected_cache" --group context --seed 43 --epochs 30 \
        --batch-size 16 --learning-rate 0.0002 --output "$run/$group" --device cuda "${extra[@]}" \
        2>&1 | tee -a "$run/${group}.log"
    done
    marker=encoding_checkpoints.json
    ;;
esac
phase=evaluation
study="$run/study"
if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
if [[ ! -f "$study/$marker" ]]; then
  if [[ -d "$study" ]]; then study="$run/study_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  case "$step" in
    alpha)
      python -u -m baselines.B5_best_context.body_alpha_study \
        --config "$config" --cache "$encoded_cache" --checkpoint "$e_run/best.pt" \
        --checkpoint-kind E --expected-epoch 24 --output "$study" --device cuda \
        2>&1 | tee -a "$run/study.log"
      ;;
    o16)
      python -u -m baselines.B5_best_context.body_encoding_study \
        --config "$config" --cache "$cache" --encoded-cache "$clip16_cache" \
        --control-run "$o30_run" --encoded-run "$run/O16" --experimental-encoding legacy_beat16_stats8_v1 \
        --output "$study" --device cuda 2>&1 | tee -a "$run/study.log"
      ;;
    seed43)
      python -u -m baselines.B5_best_context.body_encoding_study \
        --config "$config" --cache "$cache" --encoded-cache "$encoded_cache" \
        --control-run "$run/O" --encoded-run "$run/E" --output "$study" --device cuda \
        2>&1 | tee -a "$run/study.log"
      ;;
  esac
fi
printf '%s\n' "$study" > "$run/study_path.txt"
cat "$study/summary.csv"
echo "Completed: step=$step; $study"
