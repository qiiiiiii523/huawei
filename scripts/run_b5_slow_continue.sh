#!/usr/bin/env bash
# Separate slow-CNN stage; never changes legacy checkpoints or starts FM training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu=0
prepared=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
source_run=
root=
total=40
resume=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --prepared) prepared=$2; shift 2 ;;
    --source-run) source_run=$2; shift 2 ;;
    --root) root=$2; shift 2 ;;
    --total-epochs) total=$2; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use one physical GPU index' >&2; exit 2; }
[[ "$total" =~ ^[0-9]+$ && "$total" -gt 20 ]] || { echo 'Total epochs must exceed 20' >&2; exit 2; }
group=$(python -c 'from baselines.B5_exp.train_loss import default_group; print(default_group())')
case "$group" in
  C)
    source_run=${source_run:-/home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints}
    root=${root:-/home/qht/huawei/results/B5-exp/slow_continue_r_v1/S50}
    ;;
  F)
    source_run=${source_run:-/home/qht/huawei/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints}
    root=${root:-/home/qht/huawei/results/B5-exp/slow_continue_r_v1/S50_F}
    ;;
  *) echo 'Use B5-exp-S50 or B5-exp-S50-F' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
python - "$source_run" "$root" <<'PY'
import sys
from pathlib import Path
from baselines.B5_exp.continue_stage import eligible_output
eligible_output(Path(sys.argv[2]).resolve(), Path(sys.argv[1]).resolve())
PY
mkdir -p "$root"
args=(train --source-run "$source_run" --prepared "$prepared" --output "$root/run"
      --total-epochs "$total" --device cuda --execute-training)
if [[ "$resume" == 1 ]]; then args+=(--resume); fi
python -u -m baselines.B5_exp.continue_stage "${args[@]}" 2>&1 | tee -a "$root/train.log"
for candidate in best_r last; do
  python -u -m baselines.B5_exp.continue_stage evaluate \
    --stage "$root/run" --candidate "$candidate" --device cuda \
    2>&1 | tee -a "$root/evaluation_${candidate}.log"
done
echo "Slow-CNN stage and both evaluations completed: $root/run"
