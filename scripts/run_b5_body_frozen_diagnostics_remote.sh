#!/usr/bin/env bash
set -euo pipefail

python_bin="${1:?Pass the GPU Python executable as argument 1}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
output="results/context_body/A_frozen_adapter_v1"
test -f "$output/best.pt"

"$python_bin" -B -m scripts.evaluate_b5_body_diagnostics \
  --checkpoint-kind frozen --mode A_disabled --preflight --device cpu
for seed in 101 202; do
  "$python_bin" -B -m scripts.evaluate_b5_body_diagnostics \
    --checkpoint-kind frozen --mode A_shuffled --shuffle-seed "$seed" \
    --preflight --device cpu
done

"$python_bin" -u -B -m scripts.evaluate_b5_body_diagnostics \
  --checkpoint-kind frozen --mode A_disabled --device cuda \
  2>&1 | tee "$output/A_disabled.log"
for seed in 101 202; do
  "$python_bin" -u -B -m scripts.evaluate_b5_body_diagnostics \
    --checkpoint-kind frozen --mode A_shuffled --shuffle-seed "$seed" --device cuda \
    2>&1 | tee "$output/A_shuffled_seed_${seed}.log"
done
