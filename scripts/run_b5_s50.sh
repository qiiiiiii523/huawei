#!/usr/bin/env bash
# S50 original-loss continuation -> three evaluations -> comparison with existing C.
set -euo pipefail
gpu=2
root=/home/qht/huawei/results/B5-exp/slow_context_s50_v1
prepared=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
init=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/S30/best.pt
control=/home/qht/huawei/results/B5-exp/slow_loss_v1/C/checkpoints
resume=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --root) root=$2; shift 2 ;;
    --prepared) prepared=$2; shift 2 ;;
    --init-checkpoint) init=$2; shift 2 ;;
    --control-run) control=$2; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use one physical GPU index' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$gpu"
preflight="$root/shared/preflight.json"
if [[ ! -f "$preflight" ]]; then
  python -u -m baselines.B5_exp.check_s50 \
    --prepared "$prepared" \
    --init-checkpoint "$init" \
    --control-run "$control" \
    --report "$preflight" \
    --device cpu
fi
args=(--gpu "$gpu" --root "$root" --prepared "$prepared" --init-checkpoint "$init"
      --context-seconds 50 --run-name S50)
if [[ "$resume" == 1 ]]; then args+=(--resume); fi
bash scripts/run_b5_slow_loss.sh "${args[@]}"
if [[ "$resume" == 1 && -f "$root/comparison/audit.json" ]]; then
  python - "$root/comparison/audit.json" "$control" "$root/S50/checkpoints" <<'PY'
import json, sys
from pathlib import Path
from baselines.B5_exp.core import sha
from baselines.B5_exp.context_audit import verify_control_files
audit = json.load(open(sys.argv[1], encoding='utf-8'))
if not audit.get('completed'):
    raise SystemExit('Incomplete comparison')
verify_control_files(audit['control_reference'])
paths = {'S30': Path(sys.argv[2]), 'S50': Path(sys.argv[3])}
for item in audit['evaluations']:
    if sha(paths[item['context']] / ('evaluation_' + item['candidate']) / 'evaluation.json') != item['evaluation_sha256']:
        raise SystemExit('Stale comparison; preserve the old comparison and use a clean comparison directory')
PY
else
  if [[ "$resume" == 1 && -d "$root/comparison" ]]; then
    mv -- "$root/comparison" "$root/comparison_interrupted_$(date +%Y%m%d_%H%M%S)_$$"
  fi
  python -u -m baselines.B5_exp.compare_context \
    --s30 "$control" \
    --s50 "$root/S50/checkpoints" \
    --output "$root/comparison"
fi
echo "Completed S50 and S30 comparison: $root/comparison/report.md"
