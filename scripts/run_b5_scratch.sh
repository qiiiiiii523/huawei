#!/usr/bin/env bash
# Independent early-public arm. Never writes B5-S50 or B5-best results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root=/home/qht/huawei/results/B5-scratch/early_public_v1
group=
gpu=
resume=0
while (($#)); do
  case "$1" in
    --root) root="$2"; shift 2 ;;
    --group) group="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$group" == control || "$group" == slow || "$group" == slow_f ]] || { echo 'Use --group control, slow or slow_f'; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use a physical GPU number'; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python - "$root/preflight.json" "$root/shared/shared.json" <<'PY'
import hashlib,json,sys
report=json.load(open(sys.argv[1],encoding='utf-8-sig'))
digest=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
if not report.get('ready') or report.get('initializer_stage')!='public' or report.get('shared_sha256')!=digest:
    raise SystemExit('Complete B5-scratch prepare/check first; B5-S50 preflight is not accepted')
PY
out="$root/$group"
mkdir -p "$out"
resume_args=()
if ((resume)); then
  test -f "$out/checkpoints/last.pt"
  resume_args=(--resume "$out/checkpoints/last.pt")
fi
python -u -m baselines.B5_scratch.train \
  --shared "$root/shared" --group "$group" \
  --output "$out/checkpoints" --device cuda \
  "${resume_args[@]}" --execute-training \
  2>&1 | tee -a "$out/train.log"
mkdir -p "$out/evaluations"
for name in best_r last; do
  if [[ -f "$out/evaluations/$name/evaluation.json" ]]; then
    python - "$out/checkpoints/$name.pt" "$out/evaluations/$name/evaluation.json" <<'PY'
import sys
from baselines.B5_scratch.continuation import completed_evaluation_matches
if not completed_evaluation_matches(sys.argv[1],sys.argv[2]):
    raise SystemExit('Evaluation is stale. Move it aside and repeat final evaluate; never silently reuse it')
PY
    continue
  fi
  plot_args=()
  [[ "$name" != best_r ]] || plot_args=(--plots)
  python -u -m baselines.B5_scratch.evaluate \
    --shared "$root/shared" --checkpoint "$out/checkpoints/$name.pt" \
    --output "$out/evaluations/$name" --reuse-root "$out/evaluations" \
    --device cuda "${plot_args[@]}" \
    2>&1 | tee -a "$out/evaluation_$name.log"
done
echo "Completed B5-scratch $group: $out"
