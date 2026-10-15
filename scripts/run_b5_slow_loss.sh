#!/usr/bin/env bash
# One branch/group on one GPU. Existing prepared data and source best.pt are read-only.
set -euo pipefail
gpu=0
seed=42
group=
root=
prepared=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
init=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/S30/best.pt
resume=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --group) group=$2; shift 2 ;;
    --root) root=$2; shift 2 ;;
    --prepared) prepared=$2; shift 2 ;;
    --init-checkpoint) init=$2; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use one physical GPU index' >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ && ${#seed} -le 10 ]] || { echo 'seed must be in [0, 4294967295]' >&2; exit 2; }
(( 10#$seed <= 4294967295 )) || { echo 'seed must be in [0, 4294967295]' >&2; exit 2; }
seed=$((10#$seed))
if [[ -z "$root" ]]; then
  root=/home/qht/huawei/results/B5-exp/slow_loss_v1
  if [[ "$seed" != 42 ]]; then
    root="/home/qht/huawei/results/B5-exp/slow_loss_seed$seed"
  fi
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
if [[ -z "$group" ]]; then
  group=$(python -c 'from baselines.B5_exp.train_loss import default_group; print(default_group())')
fi
[[ "$group" =~ ^[CMF]$ ]] || { echo 'group must be C, M or F' >&2; exit 2; }
run="$root/$group"
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
      --group "$group" --seed "$seed"
      --output "$output" --device cuda --epochs 20 --batch-records 2 --validate-every 5 --execute-training)
if [[ "$resume" == 1 ]]; then
  test -f "$output/last.pt"
  python - "$output/resolved_config.json" "$group" "$seed" "$prepared" "$init" "$calibration" <<'PY'
import json, sys
from pathlib import Path
from baselines.B5_exp.core import sha
config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
expected = {'group': sys.argv[2], 'seed': int(sys.argv[3]),
            'prepared_sha256': sha(Path(sys.argv[4]) / 'prepared.json'),
            'init_checkpoint_sha256': sha(sys.argv[5]), 'calibration_sha256': sha(sys.argv[6])}
if any(config.get(key) != value for key, value in expected.items()):
    raise SystemExit('Resume seed/group/data/initializer/calibration disagrees with the original run')
PY
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
