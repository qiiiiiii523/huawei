#!/usr/bin/env bash
# One branch/group on one GPU. Existing prepared data and source best.pt are read-only.
set -euo pipefail
gpu=0
root=/home/qht/huawei/results/B5-exp/slow_loss_v1
prepared=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
init=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/S30/best.pt
resume=0
seconds=30
run_name=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --root) root=$2; shift 2 ;;
    --prepared) prepared=$2; shift 2 ;;
    --init-checkpoint) init=$2; shift 2 ;;
    --context-seconds) seconds=$2; shift 2 ;;
    --run-name) run_name=$2; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use one physical GPU index' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
group=$(python -c 'from baselines.B5_exp.train_loss import default_group; print(default_group())')
run_name=${run_name:-$group}
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use a simple run name' >&2; exit 2; }
run="$root/$run_name"
calibration="$root/shared/preflight.json"
test -f "$calibration"
if [[ "$resume" == 0 && -e "$run" ]]; then
  echo "Run directory already exists: $run. Use --resume or a new --root." >&2
  exit 1
fi
mkdir -p "$run"
# Logs live alongside artifacts; train_loss owns an initially empty run/checkpoints.
output="$run/checkpoints"
trap 'code=$?; echo "Stopped: group=$group line=$LINENO exit=$code output=$output" >&2; exit "$code"' ERR
args=(--prepared "$prepared" --init-checkpoint "$init" --calibration "$calibration"
      --context-seconds "$seconds" --output "$output" --device cuda --epochs 20 --batch-records 2 --validate-every 5 --execute-training)
if [[ "$resume" == 1 ]]; then
  test -f "$output/last.pt"
  if [[ ! -f "$output/completed.json" ]]; then
    python -u -m baselines.B5_exp.train_loss "${args[@]}" --resume "$output/last.pt" 2>&1 | tee -a "$run/train.log"
  fi
else
  python -u -m baselines.B5_exp.train_loss "${args[@]}" 2>&1 | tee "$run/train.log"
fi
for candidate in best_slow_r best_score last; do
  evaluation="$output/evaluation_$candidate"
  if [[ "$resume" == 1 && -f "$evaluation/evaluation.json" ]]; then
    python - "$evaluation/evaluation.json" "$output/$candidate.pt" <<'PY'
import json, sys
from baselines.B5_exp.core import sha
report = json.load(open(sys.argv[1], encoding='utf-8'))
if not report.get('completed') or report['probe_checkpoint_sha256'] != sha(sys.argv[2]):
    raise SystemExit('Stale evaluation; choose a new output or preserve and move the stale evaluation before resuming')
PY
    continue
  fi
  if [[ "$resume" == 1 && -d "$evaluation" ]]; then
    # Preserve incomplete reports, then restart evaluation in a clean directory.
    mv -- "$evaluation" "${evaluation}_interrupted_$(date +%Y%m%d_%H%M%S)_$$"
  fi
  python -u -m baselines.B5_exp.evaluate \
    --prepared "$prepared" \
    --checkpoint "$output/$candidate.pt" \
    --output "$evaluation" \
    --device cuda \
    2>&1 | tee "$run/evaluation_$candidate.log"
done
echo "Completed $group: $output"
