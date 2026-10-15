#!/usr/bin/env bash
# Existing weights and predictions only. This script never invokes training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpu=1
data_repo=/home/qht/huawei
run=/home/qht/huawei/results/B5/main_v2_score60_k16_public_finetune_meta_20261012_001711_2114239
prepared=/home/qht/huawei/results/B5-exp/slow_probe_parallel_v1/prepared
output=
cpu_only=0
resume=0
slow_args=()
with_s50=0
with_s50_f=0
while (($#)); do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --b5-run) run="$2"; shift 2 ;;
    --data-repo) data_repo="$2"; shift 2 ;;
    --prepared) prepared="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --cpu-only) cpu_only=1; shift ;;
    --resume) resume=1; shift ;;
    --with-s50) with_s50=1; shift ;;
    --with-s50-f) with_s50_f=1; shift ;;
    --slow-evaluation) slow_args+=(--slow-evaluation "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Use a physical GPU number'; exit 2; }
if ((with_s50)); then
  slow_args+=(--slow-evaluation "S50=$data_repo/results/B5-exp/slow_context_s50_v1/S50/checkpoints/evaluation_best_score")
fi
if ((with_s50_f)); then
  slow_args+=(--slow-evaluation "S50-F=$data_repo/results/B5-exp/slow_context_s50_f_v1/S50_F/checkpoints/evaluation_best_score")
fi
output="${output:-$run/qc_reassessment_v1}"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
options=(--device cuda --generate-missing)
if ((cpu_only)); then options=(--device cpu); fi
if ((resume)); then options+=(--resume); fi
trap 'echo "Stopped: line=$LINENO exit=$? output=$output" >&2' ERR
python -u -m baselines.B5.reassess_qc \
  --b5-run "$run" \
  --data-repo "$data_repo" \
  --prepared "$prepared" \
  --output-dir "$output" \
  "${options[@]}" \
  "${slow_args[@]}"
echo "QC reassessment complete: $output/report.md"
