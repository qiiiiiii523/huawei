#!/usr/bin/env bash
# No training, git operations or changes to source studies.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gpu=1
seed=42
output=
extra=()
while (($#)); do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --slow-run|--control-checkpoint|--control-evaluation) extra+=("$1" "$2"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$gpu" =~ ^[0-9]+$ || ! "$seed" =~ ^(42|43)$ ]]; then
  echo "Use --gpu N --seed 42|43 [--output ABS/new_directory]" >&2
  exit 2
fi
if [[ -z "$output" ]]; then
  output=/home/qht/huawei/results/B5-best-cross/seed${seed}_$(date +%Y%m%d_%H%M%S)_$$
fi
export CUDA_VISIBLE_DEVICES=$gpu
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd -- "$code_root"
mkdir -p -- "$(dirname -- "$output")"
if [[ -e "${output}.log" ]]; then
  echo "Log already exists; choose a new --output: ${output}.log" >&2
  exit 1
fi
echo "GPU=$gpu; seed=$seed; no training; output=$output"
python -u -m baselines.B5_cross.run \
  --seed "$seed" --device cuda --output "$output" "${extra[@]}" \
  2>&1 | tee "${output}.log"
cat -- "$output/summary.csv"
echo "Completed: $output/report.md and $output/result.json"
