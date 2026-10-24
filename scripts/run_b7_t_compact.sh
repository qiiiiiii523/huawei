#!/usr/bin/env bash
# Run INSIDE tmux with the intended BioFlow Python. Baseline files stay intact.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_b7_t_compact.sh {g6_19m|w144_g5_21m}" >&2
  exit 2
fi
preset="$1"
case "$preset" in
  g6_19m|w144_g5_21m) ;;
  *) echo "Unknown compact preset: $preset" >&2; exit 2 ;;
esac

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_python="${B7T_PYTHON:-python}"
task_cache="${B7T_CACHE_RUN:-/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42}"
task_output="${B7T_OUTPUT:-$task_root/results/B7-T-compact/$preset/seed42}"
task_gpu_p="${B7T_GPU_P:-${B7T_GPU:-0}}"
task_gpu_a="${B7T_GPU_A:-${B7T_GPU:-0}}"
task_gpu_b="${B7T_GPU_B:-${B7T_GPU:-0}}"

for cache in public10 ptb40 huawei; do
  if [[ ! -f "$task_cache/data/$cache/manifest.json" ]]; then
    echo "Missing completed cache: $task_cache/data/$cache/manifest.json" >&2
    exit 2
  fi
done
mkdir -p "$task_output"

# Probe only new runs. Existing last.pt is handled by the pipeline's strict
# identity checks. Every probe gets a new report; no previous report is erased.
if [[ "${B7T_PROBE_FIRST:-1}" == 1 && ! -f "$task_output/P/last.pt" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  for seconds in 10 40; do
    CUDA_VISIBLE_DEVICES="$task_gpu_p" "$task_python" -u -m baselines.B7.compact_probe \
      --preset "$preset" --window-seconds "$seconds" \
      --warmup "${B7T_PROBE_WARMUP:-3}" --iterations "${B7T_PROBE_ITERATIONS:-10}" \
      --memory-budget-gib "${B7T_MEMORY_BUDGET_GIB:-35}" \
      --output "$task_output/probes/${stamp}_${seconds}s.json" \
      2>&1 | tee -a "$task_output/bootstrap.log"
  done
fi

"$task_python" -u -m baselines.B7.compact \
  --preset "$preset" \
  --data-root "${B7T_DATA_ROOT:-/home/qht/huawei/data}" \
  --metadata "${B7T_METADATA:-/home/qht/huawei/metadata}" \
  --ptb-root "${B7T_PTB_ROOT:-/home/qht/huawei/data/PTB-Diagnostic}" \
  --reuse-data-from "$task_cache" --output "$task_output" \
  --gpu-p "$task_gpu_p" --gpu-a "$task_gpu_a" --gpu-b "$task_gpu_b" \
  --max-train "${B7T_MAX_TRAIN:-1}" \
  2>&1 | tee -a "$task_output/pipeline.log"
