#!/usr/bin/env bash
# Export the verified pair and reproduce its full-record QC r. No training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu=1
root=/home/qht/huawei/results/B5-best/stable_v1
config=/home/qht/huawei/configs/experiments/b5_main_v2_public_finetune_meta.yaml
while (($#)); do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --root) root="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use --gpu physical_GPU_number'; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
test -f "$config"
if [[ -e "$root" ]]; then
  echo "Use a new --root; do not overwrite a stable bundle or interrupted validation" >&2
  exit 1
fi
mkdir -p "$root"
python -u -m baselines.B5_best.bundle \
  --output "$root/model.pt" \
  2>&1 | tee "$root/export.log"
python -u -m baselines.B5_best.validate \
  --checkpoint "$root/model.pt" \
  --config "$config" \
  --output-dir "$root/evaluation" \
  --device cuda \
  --verify-reference \
  2>&1 | tee "$root/validate.log"
echo "B5-best exported and reference r verified: $root"
