#!/usr/bin/env bash
# One paired arm: Huawei-only continuation, best/last evaluation and portable export.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run=/home/qht/huawei/results/B5-P-ALL/from_Pbest_seed43_v1
group=
gpu=
resume=()
while (($#)); do
  case "$1" in
    --run) run=$2; shift 2 ;;
    --group) group=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --resume) resume=(--resume); shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$group" =~ ^(control|all)$ || ! "$gpu" =~ ^[0-9]+$ || "$run" != /* ]]; then
  echo 'Need --group control|all --gpu N [--run ABS/path] [--resume]' >&2
  exit 2
fi
test -f "$run/shared/study.json"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
exec 9>"$run/$group.lock"
flock -n 9 || { echo "Group $group is already running" >&2; exit 1; }
output="$run/$group"
echo "Checking source P best and cached records, then new paired training: $group"
python -u -m baselines.B5_p_all.train \
  --study "$run/shared/study.json" --group "$group" \
  --output "$output" --device cuda --execute-training "${resume[@]}" \
  2>&1 | tee -a "$run/$group.train.log"
stamp=$(date +%Y%m%d_%H%M%S)_$$
for name in best last; do
  evaluation="$output/evaluation_${name}_${stamp}"
  python -u -m baselines.B5_p_all.evaluate \
    --study "$run/shared/study.json" --checkpoint "$output/$name.pt" \
    --output "$evaluation" --device cuda --save-arrays \
    2>&1 | tee "${evaluation}.log"
  printf '%s\n' "$evaluation" > "$output/latest_${name}_evaluation.txt"
done
if [[ -e "$output/model.pt" ]]; then
  python - "$output" <<'PY'
import sys
from pathlib import Path
from baselines.B5_best.common import read,sha
root=Path(sys.argv[1]);meta=read(root/'model.json')
if meta['bundle_sha256']!=sha(root/'model.pt') or meta['checkpoint_sha256']!=sha(root/'best.pt'):
    raise SystemExit('Existing export differs; keep files and use an explicit new export path')
PY
else
  python -m baselines.B5_p_all.export \
    --study "$run/shared/study.json" --checkpoint "$output/best.pt" --output "$output/model.pt"
fi
echo "Completed: $output ; complete FM+slow bundle: $output/model.pt"
