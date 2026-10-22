#!/usr/bin/env bash
# Reuse immutable original slow120 FM/P caches; train new rate arms only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source_shared=
gpu=1
run=
prepare_only=0
training_args=()
while (($#)); do
  case "$1" in
    --source-shared) source_shared="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --training-seed|--epochs|--validate-every|--batch-records|--learning-rate)
      training_args+=("$1" "$2"); shift 2 ;;
    --prepare-only) prepare_only=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$source_shared" && -f "$source_shared/study.json" ]] || {
  echo '--source-shared must point to an existing original B5_slow120 shared cache' >&2; exit 2;
}
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a physical integer index' >&2; exit 2; }
[[ -n "$run" ]] || run="/home/qht/huawei/results/B5-final-exp-slowrate/run_gpu${gpu}_$(date +%Y%m%d_%H%M%S)_$$"
[[ ! -e "$run" ]] || { echo "Use a NEW run directory: $run" >&2; exit 1; }
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
phase=prepare_rate_inputs
trap 'echo "Stopped: phase=$phase line=$LINENO. Reports retained: $run" >&2' ERR
echo "Run directory: $run"
echo "Physical GPU: $gpu; process-local CUDA device: cuda:0"
python -u -m baselines.B5_slowrate.prepare --source-shared "$source_shared" \
  --output "$run/shared" "${training_args[@]}" 2>&1 | tee "$run/prepare.log"
if [[ "$prepare_only" == 1 ]]; then
  echo "Prepared without optimizer updates: $run"
  exit 0
fi
for group in R2p5 R5 R10; do
  phase="train_$group"
  python -u -m baselines.B5_slowrate.train --run "$run/shared" --group "$group" \
    --output "$run/$group" --device cuda --execute-training 2>&1 | tee "$run/$group.log"
done
for group in R2p5 R5 R10; do
  for candidate in best last; do
    phase="verify_${group}_$candidate"
    python -u -m baselines.B5_slowrate.evaluate --run "$run/shared" \
      --checkpoint "$run/$group/$candidate.pt" --output "$run/verification/$group/$candidate" \
      --device cuda 2>&1 | tee "$run/verify_${group}_$candidate.log"
  done
done
phase=summarize
python -u -m baselines.B5_slowrate.summarize --run "$run" --output "$run/comparison"
echo "Completed. Results: $run/comparison/summary.md"
echo "Run directory: $run"
