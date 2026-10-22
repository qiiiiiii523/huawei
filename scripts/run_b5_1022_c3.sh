#!/usr/bin/env bash
# Three-arm B5-1022 experiment. Existing data, base weights and G120 stay read-only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

slow_run=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/shared
slow_checkpoint=/home/qht/huawei/results/B5-final-exp-slow120/run_20261021_140544_2522863/G120/best.pt
old_c3_checkpoint=
output=
gpu=1
seed=42
epochs=10
batch_size=16
learning_rate=2e-4

while (($#)); do
  case "$1" in
    --slow-run) slow_run="$2"; shift 2 ;;
    --slow-checkpoint) slow_checkpoint="$2"; shift 2 ;;
    --old-c3-checkpoint) old_c3_checkpoint="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --learning-rate) learning_rate="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "--gpu must be a physical integer index" >&2; exit 2; }
[[ -n "$old_c3_checkpoint" ]] || { echo "--old-c3-checkpoint is required" >&2; exit 2; }
[[ -f "$slow_run/study.json" ]] || { echo "Missing slow120 study: $slow_run/study.json" >&2; exit 2; }
[[ -f "$slow_checkpoint" ]] || { echo "Missing G120 checkpoint: $slow_checkpoint" >&2; exit 2; }
[[ -f "$old_c3_checkpoint" ]] || { echo "Missing old C3 checkpoint: $old_c3_checkpoint" >&2; exit 2; }
[[ -n "$output" ]] || output=/home/qht/huawei/results/B5-1022/g120_c3_$(date +%Y%m%d_%H%M%S)_$$
[[ ! -e "$output" ]] || { echo "Use a NEW output directory: $output" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

python -u -m baselines.B5_final.g120_c3 pipeline \
  --slow-run "$slow_run" \
  --slow-checkpoint "$slow_checkpoint" \
  --old-c3-checkpoint "$old_c3_checkpoint" \
  --output "$output" \
  --device cuda \
  --seed "$seed" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --learning-rate "$learning_rate"

echo "Completed comparison: $output/comparison/summary.md"
echo "Run directory: $output"
