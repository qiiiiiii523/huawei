#!/usr/bin/env bash
# Run from the activated BioFlow environment: bash scripts/run_b7t_direct.sh 1
set -euo pipefail

B7T_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
B7T_GPU="${1:-1}"
B7T_PYTHON="$(command -v python)"
B7T_SESSION="${B7T_SESSION:-b7t-direct}"
B7T_OUTPUT="${B7T_OUTPUT:-$B7T_ROOT/results/B7-T-direct-bs16}"
B7T_SOURCE="${B7T_SOURCE:-/home/qht/huawei_B7_T_compact/results/B7-T-compact-suite}"
B7T_CACHE="${B7T_CACHE:-/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42}"
B7T_HUAWEI="${B7T_HUAWEI:-$B7T_CACHE/data/huawei/manifest.json}"
if [[ -z "${B7T_PTB:-}" ]]; then
  if [[ -f "$B7T_CACHE/ptb_recovery/data/manifest.json" ]]; then
    B7T_PTB="$B7T_CACHE/ptb_recovery/data/manifest.json"
  else
    B7T_PTB="$B7T_CACHE/data/ptb40/manifest.json"
  fi
fi

cd "$B7T_ROOT"
command -v tmux >/dev/null
if tmux has-session -t "$B7T_SESSION" 2>/dev/null; then
  echo "tmux $B7T_SESSION already exists; inspect it before restarting. No jobs launched."
  exit 1
fi
[[ -f "$B7T_HUAWEI" && -f "$B7T_PTB" ]]
for preset in g6_19m w144_g5_21m; do
  for stage in P Q; do
    [[ -f "$B7T_SOURCE/$preset/seed42/$stage/completed.json" ]]
    [[ -f "$B7T_SOURCE/$preset/seed42/$stage/best_r.pt" ]]
  done
done
CUDA_VISIBLE_DEVICES="$B7T_GPU" "$B7T_PYTHON" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
mkdir -p "$B7T_OUTPUT"
# Environment values are exported into this named tmux session, never interpolated as shell code.
tmux new-session -d -s "$B7T_SESSION" -n monitor -c "$B7T_ROOT"
for key in B7T_ROOT B7T_GPU B7T_PYTHON B7T_OUTPUT B7T_SOURCE B7T_HUAWEI B7T_PTB; do
  tmux set-environment -t "$B7T_SESSION" "$key" "${!key}"
done
for preset in g6_19m w144_g5_21m; do
  directory="$B7T_OUTPUT/$preset/seed42"
  mkdir -p "$directory"
  # Pass the known preset via the tmux environment; each window runs one serial controller.
  if [[ "$preset" == g6_19m ]]; then
    task='g6_19m'
  else
    task='w144_g5_21m'
  fi
  tmux new-window -d -t "$B7T_SESSION" -n "$preset" -c "$B7T_ROOT" \
    "exec bash scripts/run_b7t_direct_worker.sh $task"
done
echo "Started $B7T_SESSION: two serial controllers, at most two jobs in this suite, physical GPU $B7T_GPU"
echo "Results: $B7T_OUTPUT"
echo "Attach: tmux attach -t $B7T_SESSION"
