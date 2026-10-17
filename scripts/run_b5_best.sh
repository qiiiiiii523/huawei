#!/usr/bin/env bash
# Copy current weights and score reliable-I records. Historical mode is explicit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu=1
root=/home/qht/huawei/results/B5-best/stable_control100_pcc20_reliable_i_v1
config=/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239/run_config.yaml
legacy=0
reference=
while (($#)); do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --root) root="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --legacy-target-qc-only) legacy=1; shift ;;
    --reference-report) reference="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use --gpu physical_GPU_number'; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
test -f "$config"
[[ "$legacy" == 0 || -z "$reference" ]] || { echo 'Choose historical mode or new reference, not both' >&2; exit 2; }
extra=()
if [[ "$legacy" == 1 ]]; then
  extra=(--legacy-target-qc-only --verify-reference)
elif [[ -n "$reference" ]]; then
  test -f "$reference"
  extra=(--reference-report "$reference" --verify-reference)
fi
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
  "${extra[@]}" \
  2>&1 | tee "$root/validate.log"
echo "B5-best exported and scored; see evaluation.json for reference scope/result: $root"
