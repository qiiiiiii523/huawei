#!/usr/bin/env bash
# One paired arm; fixed cache, no git/network, no FM optimization.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
study=
cache=
group=
output=
gpu=
resume=
while (($#)); do
  case "$1" in
    --study) study=$2; shift 2 ;;
    --cache) cache=$2; shift 2 ;;
    --group) group=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --resume) resume=$2; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$study" || -z "$cache" || -z "$output" || ! "$group" =~ ^(control|pcc)$ || ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "Need --study ABS/study.json --cache ABS/cache --group control|pcc --output ABS/run --gpu N [--resume ABS/last.pt]" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=$gpu
cd -- "$code_root"
mkdir -p -- "$(dirname -- "$output")"
extra=()
if [[ -n "$resume" ]]; then extra=(--resume "$resume"); fi
python -u -m baselines.B5_slow_pcc.train \
  --study "$study" --cache "$cache" --group "$group" \
  --output "$output" --device cuda --execute-training "${extra[@]}" \
  2>&1 | tee -a "${output}.train.log"
stamp=$(date +%Y%m%d_%H%M%S)_$$
for checkpoint in best last; do
  evaluation="$output/evaluation_${checkpoint}_${stamp}"
  python -u -m baselines.B5_slow_pcc.evaluate \
    --study "$study" --cache "$cache" --checkpoint "$output/${checkpoint}.pt" \
    --output "$evaluation" --device cuda \
    2>&1 | tee "${evaluation}.log"
  printf '%s\n' "$evaluation" > "$output/latest_${checkpoint}_evaluation.txt"
done
python -u -m baselines.B5_slow_pcc.export \
  --study "$study" --checkpoint "$output/best.pt" \
  --output "$output/best_bundle_${stamp}.pt"
echo "Completed slow-CNN training, best/last evaluation and best portable export: $output"
