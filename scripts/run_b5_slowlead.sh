#!/usr/bin/env bash
# Operates in this worktree; no git fetch, branch switches or source mutations.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu=0
group=
root=/home/qht/huawei/results/B5-1022-exp-slowlead/abc_v1
while (($#)); do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --group) group="$2"; shift 2 ;;
    --run) root="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ && "$group" =~ ^(A|B|C)$ ]] || { echo 'Use --group A|B|C --gpu N' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
[[ -f "$root/shared/preflight.json" ]] || { echo 'Run prepare successfully first' >&2; exit 1; }
python - "$root/shared/preflight.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1])).get('ready') is not True:
    raise ValueError('SlowLead preflight did not pass')
PY
mkdir -p "$root/logs"
trap 'echo "Stopped at line $LINENO. Keep $root/$group; rerun this command for epoch-boundary resume." >&2' ERR
if [[ ! -f "$root/$group/completed.json" ]]; then
  resume=()
  if [[ -f "$root/$group/last.pt" ]]; then
    resume=(--resume "$root/$group/last.pt")
  elif [[ -d "$root/$group" ]]; then
    echo "Incomplete directory without last.pt: $root/$group. Preserve/rename it before restarting." >&2
    exit 1
  fi
  python -u -m baselines.B5_slowlead.train --run "$root/shared" --group "$group" \
    --output "$root/$group" --device cuda --execute-training "${resume[@]}" \
    2>&1 | tee -a "$root/logs/$group.log"
fi
for name in best_r last; do
  dest="$root/$group/evaluations/$name"
  if [[ -f "$dest/evaluation.json" ]]; then
    python - "$dest/evaluation.json" "$root/$group/$name.pt" <<'PY'
import sys
from baselines.B5_best.common import read,sha
report=read(sys.argv[1])
if report.get('completed') is not True or report.get('checkpoint_sha256') != sha(sys.argv[2]):
    raise ValueError('Existing evaluation is incomplete or belongs to another checkpoint')
PY
  fi
  if [[ ! -f "$dest/evaluation.json" ]]; then
    [[ ! -e "$dest" ]] || { echo "Incomplete evaluation exists: $dest; preserve/rename it and rerun." >&2; exit 1; }
    python -u -m baselines.B5_slowlead.evaluate --run "$root/shared" \
      --checkpoint "$root/$group/$name.pt" --output "$dest" --device cuda \
      2>&1 | tee -a "$root/logs/${group}_${name}.log"
  fi
done
echo "Completed $group: $root/$group"
