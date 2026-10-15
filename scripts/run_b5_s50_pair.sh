#!/usr/bin/env bash
# Run ONE matched arm; use separate terminals/devices for the three arms.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
group=
gpu=
root=/home/qht/huawei/results/B5-S50/fm_slow_qc_three_v2
resume=0
total_epochs=
while (($#)); do
  case "$1" in
    --group) group="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --root) root="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    --total-epochs) total_epochs="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$group" == control || "$group" == slow || "$group" == slow_f ]] || { echo 'Use --group control, slow or slow_f'; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use --gpu physical_GPU_number'; exit 2; }
[[ -z "$total_epochs" || "$total_epochs" =~ ^[1-9][0-9]*$ ]] || { echo 'Use a positive --total-epochs'; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
shared="$root/shared"
out="$root/$group"
test -f "$shared/shared.json"
test -f "$root/preflight.json"
python - "$root/preflight.json" "$shared/shared.json" <<'PY'
import hashlib,json,sys
with open(sys.argv[1],encoding='utf-8-sig') as stream:
    report=json.load(stream)
with open(sys.argv[2],'rb') as stream:
    digest=hashlib.sha256(stream.read()).hexdigest()
if not report.get('ready') or report.get('shared_sha256')!=digest:
    raise SystemExit('Run the new three-arm preflight for this shared cache first')
PY
mkdir -p "$out"
resume_args=()
budget_args=()
if [[ -n "$total_epochs" ]]; then
  budget_args=(--total-epochs "$total_epochs")
fi
if ((resume)); then
  test -f "$out/checkpoints/last.pt"
  resume_args=(--resume "$out/checkpoints/last.pt")
fi
trap 'echo "Stopped: group=$group line=$LINENO exit=$?; see $out/train.log" >&2' ERR
python -u -m baselines.B5_s50.train \
  --shared "$shared" \
  --group "$group" \
  --output "$out/checkpoints" \
  --device cuda \
  "${resume_args[@]}" \
  "${budget_args[@]}" \
  --execute-training \
  2>&1 | tee -a "$out/train.log"

evaluation_root=$(python - "$out" <<'PY'
import json,sys
from pathlib import Path
from baselines.B5_s50.continuation import evaluation_directory
out=Path(sys.argv[1])
done=json.loads((out/'checkpoints/completed.json').read_text(encoding='utf-8'))
if not done.get('completed'): raise SystemExit('Training budget has not completed')
print(evaluation_directory(out,done.get('total_epochs',done['additional_epochs'])))
PY
)
mkdir -p "$evaluation_root"
for name in best best_r best_bonus last; do
  if python - "$out/checkpoints/$name.pt" "$evaluation_root/$name/evaluation.json" <<'PY'
import sys
from baselines.B5_s50.continuation import completed_evaluation_matches
raise SystemExit(0 if completed_evaluation_matches(*sys.argv[1:]) else 1)
PY
  then
    echo "Keeping completed evaluation: $evaluation_root/$name (checkpoint SHA matches)"
    continue
  fi
  if [[ -d "$evaluation_root/$name" ]]; then
    mv -- "$evaluation_root/$name" "$evaluation_root/${name}_interrupted_$(date +%Y%m%d_%H%M%S)_$$"
  fi
  plot_args=()
  [[ "$name" != best ]] || plot_args=(--plots)
  python -u -m baselines.B5_s50.evaluate \
    --shared "$shared" \
    --checkpoint "$out/checkpoints/$name.pt" \
    --output "$evaluation_root/$name" \
    --reuse-root "$evaluation_root" \
    --device cuda \
    "${plot_args[@]}" \
    2>&1 | tee -a "$evaluation_root/evaluation_$name.log"
done
echo "Completed $group: $out"
