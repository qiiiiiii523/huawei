#!/usr/bin/env bash
# Reuse S50 control; train ONLY S50-F, evaluate, then compare.
set -euo pipefail
gpu=2
root=/home/qht/huawei/results/B5-exp/slow_context_s50_f_v1
prepared=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
init=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/S30/best.pt
control=/home/qht/huawei/results/B5-exp/slow_context_s50_v1/S50/checkpoints
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
python -c "from baselines.B5_exp.train_loss import default_group; assert default_group() == 'F', 'Use B5-exp-S50-F branch'"
preflight="$root/shared/preflight.json"
if [[ ! -f "$preflight" ]]; then
  python -u -m baselines.B5_exp.check_s50_f \
    --prepared "$prepared" \
    --init-checkpoint "$init" \
    --control-run "$control" \
    --report "$preflight" \
    --device cpu
fi
python - "$preflight" "$control" <<'PY'
import json,sys
from pathlib import Path
from baselines.B5_exp.s50_f_audit import verify_control
proof=json.load(open(sys.argv[1],encoding='utf-8'))['s50_f_pair']
if Path(proof['control_reference']['path']).resolve()!=Path(sys.argv[2]).resolve():
    raise SystemExit('Preflight was paired with another control')
verify_control(proof['control_reference'])
PY
args=(--gpu "$gpu" --root "$root" --prepared "$prepared" --init-checkpoint "$init"
      --context-seconds 50 --run-name S50_F)
if [[ "$resume" == 1 ]]; then args+=(--resume); fi
bash scripts/run_b5_slow_loss.sh "${args[@]}"
if [[ "$resume" == 1 && -f "$root/comparison/audit.json" ]]; then
  python - "$root/comparison/audit.json" "$control" "$root/S50_F/checkpoints" <<'PY'
import json,sys
from pathlib import Path
from baselines.B5_exp.core import sha
from baselines.B5_exp.s50_f_audit import verify_control
audit=json.load(open(sys.argv[1],encoding='utf-8'))
if not audit.get('completed'): raise SystemExit('Incomplete comparison')
verify_control(audit['control_reference'])
paths={'S50':Path(sys.argv[2]),'S50-F':Path(sys.argv[3])}
for item in audit['evaluations']:
    if sha(paths[item['experiment']]/('evaluation_'+item['candidate'])/'evaluation.json')!=item['evaluation_sha256']:
        raise SystemExit('Stale comparison; preserve it and use a clean comparison directory')
PY
else
  if [[ "$resume" == 1 && -d "$root/comparison" ]]; then
    mv -- "$root/comparison" "$root/comparison_interrupted_$(date +%Y%m%d_%H%M%S)_$$"
  fi
  python -u -m baselines.B5_exp.compare_s50_f \
    --control "$control" \
    --experiment "$root/S50_F/checkpoints" \
    --output "$root/comparison"
fi
echo "Completed S50-F: $root/comparison/report.md"
