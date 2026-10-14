#!/usr/bin/env bash
set -euo pipefail

python_bin="${1:?Pass the GPU Python executable as argument 1}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
output="results/context_body/A_frozen_meanstd_v1"
test -f "$output/best.pt"

for mode in A_disabled A_shuffled; do
  "$python_bin" -B -m scripts.evaluate_b5_body_diagnostics \
    --checkpoint-kind frozen_meanstd --mode "$mode" --preflight --device cpu
done
"$python_bin" -u -B -m scripts.evaluate_b5_body_diagnostics \
  --checkpoint-kind frozen_meanstd --mode A_disabled --device cuda \
  2>&1 | tee "$output/A_disabled.log"
for seed in 101 202; do
  "$python_bin" -u -B -m scripts.evaluate_b5_body_diagnostics \
    --checkpoint-kind frozen_meanstd --mode A_shuffled --shuffle-seed "$seed" --device cuda \
    2>&1 | tee "$output/A_shuffled_seed_${seed}.log"
done
"$python_bin" -u -B -m scripts.check_b5_body_adapter_usage \
  --checkpoint-kind frozen_meanstd --device cuda \
  2>&1 | tee "$output/adapter_usage.log"
