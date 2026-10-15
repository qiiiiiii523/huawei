#!/usr/bin/env bash
# Run ONE paired arm; use separate terminals/devices for the two arms.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
group=
gpu=
root=/home/qht/huawei/results/B5-S50/fm_condition_v1
resume=0
while (($#)); do
  case "$1" in
    --group) group="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --root) root="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$group" == control || "$group" == slow ]] || { echo 'Use --group control or slow'; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use --gpu physical_GPU_number'; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
shared="$root/shared"
out="$root/$group"
test -f "$shared/shared.json"
test -f "$root/preflight.json"
mkdir -p "$out"
resume_args=()
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
  --execute-training \
  2>&1 | tee -a "$out/train.log"

mkdir -p "$out/evaluations"
for name in best best_r best_bonus last; do
  if [[ -f "$out/evaluations/$name/evaluation.json" ]]; then
    echo "Keeping completed evaluation: $out/evaluations/$name (compare will verify its checkpoint hash)"
    continue
  fi
  # Keep partial evaluation directories; a new suffix is not accepted silently.
  # Move an interrupted directory aside before using --resume (see documentation).
  plot_args=()
  [[ "$name" != best ]] || plot_args=(--plots)
  python -u -m baselines.B5_s50.evaluate \
    --shared "$shared" \
    --checkpoint "$out/checkpoints/$name.pt" \
    --output "$out/evaluations/$name" \
    --reuse-root "$out/evaluations" \
    --device cuda \
    "${plot_args[@]}" \
    2>&1 | tee -a "$out/evaluation_$name.log"
done
echo "Completed $group: $out"
