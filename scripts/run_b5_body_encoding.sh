#!/usr/bin/env bash
# Paired history encoding experiment; no B5 generation, gate or d6 training.
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
[[ -n "$run" ]] || run="$repository/results/B5-best-context/body_encoding_v1"
[[ -n "$config" ]] || config="$repository/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml"
cache="$source_run/cache"
test -f "$cache/cache.json"
test -f "$config"
if [[ "$resume" == 0 && -e "$run" ]]; then echo "Use a new --run or explicit --resume: $run" >&2; exit 1; fi
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
phase=setup
trap 'echo "Stopped: phase=$phase line=$LINENO; repeat SAME settings with --resume: $run" >&2' ERR
python - "$run" "$cache" "$config" "$seed" "$epochs" <<'PY'
import json, pathlib, sys
from baselines.B5_best.common import sha
run, cache, config = map(pathlib.Path, sys.argv[1:4])
value = dict(cache=str(cache.resolve()), cache_sha256=sha(cache / "cache.json"),
             config=str(config.resolve()), config_sha256=sha(config), seed=int(sys.argv[4]),
             epochs=int(sys.argv[5]), initialization="same_random_weights", record_gate=None)
path = run / "run.json"
if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
    raise ValueError("Resume changed input/cache/config/seed/budget; use a new run")
path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
phase=encoding_prepare
extra=()
if [[ "$resume" == 1 ]]; then extra+=(--resume); fi
python -u -m baselines.B5_best_context.body_encoding_prepare \
  --config "$config" --cache "$cache" --output "$run/encoded_cache" "${extra[@]}" \
  2>&1 | tee -a "$run/prepare.log"
for mode in O E; do
  phase="preflight_$mode"
  selected_cache="$cache"
  if [[ "$mode" == E ]]; then selected_cache="$run/encoded_cache"; fi
  preflight="$run/preflight_$mode"
  if [[ -f "$run/preflight_${mode}_path.txt" ]]; then preflight=$(cat "$run/preflight_${mode}_path.txt"); fi
  if [[ ! -f "$preflight/preflight.json" ]]; then
    if [[ -d "$preflight" ]]; then preflight="$run/preflight_${mode}_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
    python -u -m baselines.B5_best_context.body_train \
      --cache "$selected_cache" --group context --seed "$seed" --epochs "$epochs" \
      --output "$preflight" --check-only \
      2>&1 | tee -a "$run/preflight_${mode}.log"
  fi
  printf '%s\n' "$preflight" > "$run/preflight_${mode}_path.txt"
done
phase=paired_preflight
python -u -m baselines.B5_best_context.body_encoding_study \
  --config "$config" --cache "$cache" --encoded-cache "$run/encoded_cache" \
  --control-run "$(cat "$run/preflight_O_path.txt")" \
  --encoded-run "$(cat "$run/preflight_E_path.txt")" \
  --output "$run/preflight_pair.json" --check-only \
  2>&1 | tee -a "$run/preflight_pair.log"
for mode in O E; do
  phase="train_$mode"
  selected_cache="$cache"
  if [[ "$mode" == E ]]; then selected_cache="$run/encoded_cache"; fi
  extra=()
  if [[ "$resume" == 1 && -f "$run/$mode/config.json" ]]; then extra+=(--resume); fi
  python -u -m baselines.B5_best_context.body_train \
    --cache "$selected_cache" --group context --seed "$seed" --epochs "$epochs" \
    --output "$run/$mode" --device cuda "${extra[@]}" \
    2>&1 | tee -a "$run/${mode}.log"
done
phase=evaluation
study="$run/study"
if [[ -f "$run/study_path.txt" ]]; then study=$(cat "$run/study_path.txt"); fi
if [[ ! -f "$study/encoding_checkpoints.json" ]]; then
  if [[ -d "$study" ]]; then study="$run/study_retry_$(date +%Y%m%d_%H%M%S)_$$"; fi
  python -u -m baselines.B5_best_context.body_encoding_study \
    --config "$config" --cache "$cache" --encoded-cache "$run/encoded_cache" \
    --control-run "$run/O" --encoded-run "$run/E" --output "$study" --device cuda \
    2>&1 | tee -a "$run/study.log"
fi
printf '%s\n' "$study" > "$run/study_path.txt"
cat "$study/summary.csv"
cat "$study/encoding_effect.json"
echo "Completed: $run"
echo "O/E study: $study"
