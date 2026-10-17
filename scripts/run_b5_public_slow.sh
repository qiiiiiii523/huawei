#!/usr/bin/env bash
set -euo pipefail

group=
gpu=
run=/home/qht/huawei/results/B5-slow-public/CMP_v1
resume=()
while (($#)); do
  case "$1" in
    --group) group="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --resume) resume=(--resume); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$group" =~ ^[CMP]$ && "$gpu" =~ ^[0-9]+$ ]] || {
  echo 'Usage: bash scripts/run_b5_public_slow.sh --group C|M|P --gpu N [--run DIR] [--resume]' >&2
  exit 2
}
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
[[ -f "$run/shared/study.json" ]] || { echo 'Complete prepare/preflight first' >&2; exit 1; }
exec 9>"$run/$group.lock"
flock -n 9 || { echo "Group $group is already running" >&2; exit 1; }

# Logs beside each output: tee must not make a fresh group directory nonempty.
echo "$group: verifying immutable caches on CPU first; GPU activity starts after verification. Log: $run/$group.log"
python -u -m baselines.B5_public_slow.train \
  --study "$run/shared/study.json" \
  --group "$group" --output "$run/$group" --device cuda \
  --execute-training "${resume[@]}" 2>&1 | tee -a "$run/$group.log"

if [[ ! -e "$run/$group/model.pt" ]]; then
  python -m baselines.B5_public_slow.export \
    --study "$run/shared/study.json" --checkpoint "$run/$group/best.pt" \
    --output "$run/$group/model.pt"
else
  python - "$run/$group" <<'PY'
import sys
from pathlib import Path
from baselines.B5_best.common import read,sha
root=Path(sys.argv[1]); meta=read(root/'model.json')
if meta['checkpoint_sha256']!=sha(root/'best.pt') or meta['bundle_sha256']!=sha(root/'model.pt'):
    raise SystemExit('Existing export differs; keep it and choose a new export path explicitly')
print('Existing export verified')
PY
fi
echo "$group completed: $run/$group"
