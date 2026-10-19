#!/usr/bin/env bash
set -euo pipefail
gpu=1
output=
extra=()
while (($#)); do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --run|--seed43-review|--slow-bundle) extra+=("$1" "$2"); shift 2 ;;
    -h|--help) echo 'Usage: bash scripts/run_b5_public_fm_cross.sh [--gpu 1] [--output NEW_DIR] [--run DIR] [--seed43-review DIR] [--slow-bundle P/model.pt]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ $gpu =~ ^[0-9]+$ ]] || { echo 'GPU must be one physical index' >&2; exit 2; }
cd "$(dirname "$0")/.."
output=${output:-/home/qht/huawei/results/B5-best-exp-PublicFM/cross_P_$(date +%Y%m%d_%H%M%S)_$$}
mkdir -p "$(dirname "$output")"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
# Log beside the fresh output directory, never inside it before preflight.
python -u scripts/b5_public_fm_cross.py --output "$output" --device cuda "${extra[@]}" \
  2>&1 | tee "${output}.log"
cp "${output}.log" "$output/run.log"
cat "$output/summary.csv"
echo "Results: $output"
