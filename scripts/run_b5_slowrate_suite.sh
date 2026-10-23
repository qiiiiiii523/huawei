#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu=1
args=()
while (($#)); do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --resume|--plan) args+=("$1"); shift ;;
    --preset|--source-shared|--run|--seeds|--stages|--groups|--reference-group|--rate-epochs|--rate-validate-every|--rate-batch-records|--followup-epochs|--followup-validate-every|--followup-batch-records|--learning-rate|--fusion-learning-rate|--tail-blocks|--p-preservation|--p-context-seconds|--p-extra-dilation)
      args+=("$1" "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a physical integer index' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
echo "Physical GPU: $gpu; process-local device: cuda:0"
exec python -u -m baselines.B5_slowrate_followup.suite "${args[@]}" --device cuda --execute-training
