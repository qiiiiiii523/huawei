#!/usr/bin/env bash
# One arm per GPU; no git/network operations. Logs stay OUTSIDE fresh output.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
group=
study=
output=
gpu=
resume=
while (($#)); do
  case "$1" in
    --group) group=$2; shift 2 ;;
    --study) study=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --resume) resume=$2; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$group" =~ ^(control|v13)$ || -z "$study" || -z "$output" || ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/run_b5_v13.sh --group control|v13 --study ABS/study.json --output ABS/run --gpu 0 [--resume ABS/last.pt]" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=$gpu
cd -- "$code_root"
mkdir -p -- "$(dirname -- "$output")"
extra=()
if [[ -n "$resume" ]]; then extra=(--resume "$resume"); fi
echo "Group=$group physical_GPU=$gpu code=$code_root output=$output"
python -u -m baselines.B5_v13.train \
  --study "$study" --group "$group" --output "$output" \
  --device cuda --execute-training "${extra[@]}" \
  2>&1 | tee -a "${output}.train.log"
stamp=$(date +%Y%m%d_%H%M%S)_$$
for checkpoint in best last; do
  evaluation="$output/evaluation_${checkpoint}_${stamp}"
  python -u -m baselines.B5_v13.validate \
    --study "$study" --checkpoint "$output/${checkpoint}.pt" \
    --output "$evaluation" --device cuda \
    2>&1 | tee "${evaluation}.log"
  printf '%s\n' "$evaluation" > "$output/latest_${checkpoint}_evaluation.txt"
done
echo "Completed training and best/last evaluation: $output"
