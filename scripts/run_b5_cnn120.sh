#!/usr/bin/env bash
# New experiment only; original data/results/weights are read-only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repository=/home/qht/huawei
gpu=0
run=
config=
baseline=
regenerate=0
prepare_only=0
asset_args=()
train_args=()
while (($#)); do
  case "$1" in
    --repository-root) repository="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --baseline-evaluation) baseline="$2"; shift 2 ;;
    --profile|--checkpoint|--body-checkpoint) asset_args+=("$1" "$2"); shift 2 ;;
    --training-seed|--epochs|--validate-every|--batch-records|--learning-rate|--fm-batch-size)
      train_args+=("$1" "$2"); shift 2 ;;
    --regenerate-base) regenerate=1; shift ;;
    --prepare-only) prepare_only=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a physical integer index' >&2; exit 2; }
[[ "$regenerate" == 0 || -z "$baseline" ]] || { echo 'Choose regenerate OR existing baseline' >&2; exit 2; }
[[ -n "$run" ]] || run="$repository/results/B5-final-exp-cnn120/run_$(date +%Y%m%d_%H%M%S)_$$"
[[ ! -e "$run" ]] || { echo "Use a NEW run directory: $run" >&2; exit 1; }
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
phase=preflight
trap 'echo "Stopped: phase=$phase line=$LINENO. Reports retained: $run. See resume instructions." >&2' ERR
python -u -m baselines.B5_final.profile --repository-root "$repository" "${asset_args[@]}" \
  --report "$run/assets.json" 2>&1 | tee "$run/preflight.log"
[[ -n "$config" ]] || config=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["validation_config"])' "$run/assets.json")
if [[ "$regenerate" == 1 ]]; then
  phase=generate_FM_P_validation
  checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_checkpoint"])' "$run/assets.json")
  baseline="$run/base_inference"
  python -u -m baselines.B5_best.validate --checkpoint "$checkpoint" --config "$config" \
    --output-dir "$baseline" --device cuda 2>&1 | tee "$run/base_inference.log"
fi
[[ -n "$baseline" ]] || baseline=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseline_evaluation"])' "$run/assets.json")
phase=reproduce_G0
python -u -m baselines.B5_final.validate --repository-root "$repository" "${asset_args[@]}" \
  --config "$config" --baseline-evaluation "$baseline" --output "$run/G0" \
  --device cuda --verify-reference 2>&1 | tee "$run/G0.log"
phase=prepare_frozen_cache
python -u -m baselines.B5_cnn120.prepare --repository-root "$repository" "${asset_args[@]}" \
  --config "$config" --baseline-evaluation "$baseline" --g0-report "$run/G0/final.json" \
  --output "$run/shared" --device cuda "${train_args[@]}" 2>&1 | tee "$run/prepare.log"
if [[ "$prepare_only" == 1 ]]; then
  echo "Prepared without optimizer updates: $run"
  exit 0
fi
for group in L120 C120; do
  phase="train_$group"
  python -u -m baselines.B5_cnn120.train --run "$run/shared" --group "$group" \
    --output "$run/$group" --device cuda --execute-training 2>&1 | tee "$run/$group.log"
done
for group in L120 C120; do
  for candidate in best last; do
    phase="verify_${group}_$candidate"
    python -u -m baselines.B5_cnn120.evaluate --run "$run/shared" \
      --checkpoint "$run/$group/$candidate.pt" --output "$run/verification/$group/$candidate" \
      --device cuda 2>&1 | tee "$run/verify_${group}_$candidate.log"
  done
done
phase=summarize
python -u -m baselines.B5_cnn120.summarize --run "$run" --output "$run/comparison"
echo "Completed. Results: $run/comparison/summary.md"
echo "Run directory: $run"
