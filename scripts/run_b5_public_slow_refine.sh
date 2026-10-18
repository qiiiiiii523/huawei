#!/usr/bin/env bash
set -euo pipefail
seed=
gpu=
root=/home/qht/huawei/results/B5-slow-public/P_refine_v1
resume=()
while (($#)); do
  case "$1" in
    --seed) seed="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --root) root="$2"; shift 2 ;;
    --resume) resume=(--resume); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$seed" =~ ^(42|43)$ && "$gpu" =~ ^[0-9]+$ ]] || {
  echo 'Usage: bash scripts/run_b5_public_slow_refine.sh --seed 42|43 --gpu N [--root DIR] [--resume]' >&2; exit 2;
}
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
test -f "$root/trials.json"
exec 9>"$root/seed$seed.lock"
flock -n 9 || { echo 'This training seed is already running' >&2; exit 1; }
for arm in lr1e4 lr3e5; do
  trial="$root/seed${seed}_$arm"
  arm_resume=()
  if ((${#resume[@]})) && [[ -f "$trial/P/last.pt" ]]; then arm_resume=(--resume); fi
  echo "Training seed $seed, $arm: verify frozen caches on CPU, then 780 stage2 updates."
  python -u -m baselines.B5_public_slow.train \
    --study "$trial/shared/study.json" --group P \
    --output "$trial/P" --device cuda --execute-training \
    "${arm_resume[@]}" 2>&1 | tee -a "$trial/train.log"
  if [[ "$arm" == lr1e4 ]]; then
    python -m baselines.B5_public_slow.refine verify-reproduction --trial "$trial"
  fi
  if [[ ! -e "$trial/P/model.pt" ]]; then
    python -m baselines.B5_public_slow.export \
      --study "$trial/shared/study.json" --checkpoint "$trial/P/best.pt" \
      --output "$trial/P/model.pt"
  else
    python - "$trial/P" <<'PY'
import sys
from pathlib import Path
from baselines.B5_best.common import read,sha
folder=Path(sys.argv[1]); meta=read(folder/'model.json')
if meta['checkpoint_sha256']!=sha(folder/'best.pt') or meta['bundle_sha256']!=sha(folder/'model.pt'):
    raise SystemExit('Existing export mismatch; choose a new export path')
PY
  fi
done
echo "Training seed $seed: both LR arms finished. Results: $root"
