#!/usr/bin/env bash
# Run ONE arm. Existing source caches are read-only; no git/network/FM inference.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run=/home/qht/huawei/results/B5-best-exp-Control100PCC/partner_v1
group=
gpu=
resume=
while (($#)); do
  case "$1" in
    --run) run=$2; shift 2 ;;
    --group) group=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --resume) resume=$2; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$group" =~ ^(control_old_fm|adapt_control100)$ || ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "Need --group control_old_fm|adapt_control100 --gpu N [--run ABS/path] [--resume ABS/last.pt]" >&2
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
python -u -m baselines.B5_control_pcc.train \
  --study "$study" --cache "$cache" --group "$group" \
  --output "$output" --device cuda --execute-training "${extra[@]}" \
  2>&1 | tee -a "${output}.train.log"
stamp=$(date +%Y%m%d_%H%M%S)_$$
for checkpoint in best last; do
  evaluation="$output/evaluation_${checkpoint}_${stamp}"
  python -u -m baselines.B5_control_pcc.evaluate \
    --study "$study" --cache "$cache" --checkpoint "$output/${checkpoint}.pt" \
    --output "$evaluation" --device cuda \
    2>&1 | tee "${evaluation}.log"
  printf '%s\n' "$evaluation" > "$output/latest_${checkpoint}_evaluation.txt"
done
python -u -m baselines.B5_control_pcc.export \
  --study "$study" --checkpoint "$output/best.pt" \
  --output "$output/best_bundle_${stamp}.pt"
echo "Completed: $output. Compare BOTH best and last, all using control100 validation."
