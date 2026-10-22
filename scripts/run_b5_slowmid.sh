#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu=0
group=H
mode=mid
root=/home/qht/huawei/results/B5-final-exp-slowmid/mid_v1
while (($#)); do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --group) group="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --run) root="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ && "$group" =~ ^(H|P)$ && "$mode" =~ ^(add|mid)$ ]] || {
  echo 'Use --group H|P --mode add|mid --gpu N' >&2; exit 2;
}
export CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
arm="${group}_${mode}"
out="$root/$arm"
[[ -f "$root/shared/study.json" ]] || { echo 'Prepare this branch shared/ successfully first' >&2; exit 1; }
mkdir -p "$root/logs"
trap 'echo "Stopped at line $LINENO. Keep $out and rerun the SAME command to resume." >&2' ERR
if [[ ! -f "$out/completed.json" ]]; then
  resume=()
  if [[ -f "$out/last.pt" ]]; then
    resume=(--resume "$out/last.pt")
  elif [[ -d "$out" ]]; then
    echo "Existing arm has no last.pt: $out. Preserve/rename it before restarting." >&2
    exit 1
  fi
  python -u -m baselines.B5_slowjoint.train --run "$root/shared" --group "$group" --fusion-mode "$mode" \
    --output "$out" --device cuda --execute-training "${resume[@]}" \
    2>&1 | tee -a "$root/logs/$arm.log"
fi
for name in best_r last; do
  dest="$out/evaluations/$name"
  if [[ ! -f "$dest/evaluation.json" ]]; then
    [[ ! -e "$dest" ]] || { echo "Incomplete evaluation exists: $dest; preserve/rename it and rerun." >&2; exit 1; }
    python -u -m baselines.B5_slowjoint.evaluate --run "$root/shared" --checkpoint "$out/$name.pt" \
      --output "$dest" --device cuda 2>&1 | tee -a "$root/logs/${arm}_${name}.log"
  fi
done
echo "Completed $arm: $out"
