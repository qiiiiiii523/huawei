#!/usr/bin/env bash
# One arm: train -> evaluate best and last -> export the portable best model.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run=/home/qht/huawei/results/B5-best-exp-SlowMSResidual/ms_residual_v1
group=
gpu=
resume=
while (($#)); do
  case "$1" in
    --run|--group|--gpu|--resume)
      if (($# < 2)); then echo "Missing value: $1" >&2; exit 2; fi
      case "$1" in
        --run) run=$2 ;;
        --group) group=$2 ;;
        --gpu) gpu=$2 ;;
        --resume) resume=$2 ;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$group" =~ ^(control|residual)$ || ! "$gpu" =~ ^[0-9]+$ || "$run" != /* ]]; then
  echo "Need --group control|residual --gpu N [--run ABS/path] [--resume ABS/last.pt]" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=$gpu
cd -- "$code_root"
study="$run/shared/study.json"
cache="$run/shared/cache_view.json"
output="$run/$group"
test -f "$study"
test -f "$cache"
extra=()
if [[ -n "$resume" ]]; then extra=(--resume "$resume"); fi
python -u -m baselines.B5_slow_ms.train \
  --study "$study" --cache "$cache" --group "$group" \
  --output "$output" --device cuda --execute-training "${extra[@]}" \
  2>&1 | tee -a "${output}.train.log"
stamp=$(date +%Y%m%d_%H%M%S)_$$
for checkpoint in best last; do
  evaluation="$output/evaluation_${checkpoint}_${stamp}"
  python -u -m baselines.B5_slow_ms.evaluate \
    --study "$study" --cache "$cache" --checkpoint "$output/${checkpoint}.pt" \
    --output "$evaluation" --device cuda \
    2>&1 | tee "${evaluation}.log"
  printf '%s\n' "$evaluation" > "$output/latest_${checkpoint}_evaluation.txt"
done
bundle="$output/best_bundle_${stamp}.pt"
python -u -m baselines.B5_slow_ms.export \
  --study "$study" --checkpoint "$output/best.pt" --output "$bundle"
printf '%s\n' "$bundle" > "$output/latest_best_bundle.txt"
echo "Completed: $output"
