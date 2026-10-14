#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_b5_body_diagnostics_remote.sh /absolute/path/to/python
# Reuse existing control and A_true epoch-15 validation outputs. Only three
# additional inference passes run; no optimizer, training, or checkpoint writes.
python_bin="${1:?Pass the GPU Python executable as argument 1}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

for mode in base A_disabled A_shuffled; do
  "$python_bin" -B -m scripts.evaluate_b5_body_diagnostics \
    --mode "$mode" --preflight --device cpu
done

mkdir -p results/context_body/diagnostics
for mode in base A_disabled A_shuffled; do
  printf '%s starting %s\n' "$(date -Is)" "$mode" | tee -a results/context_body/diagnostics/run_sequence.log
  "$python_bin" -u -B -m scripts.evaluate_b5_body_diagnostics \
    --mode "$mode" --device cuda --batch-size 4 --steps 16 --samples 16 \
    2>&1 | tee "results/context_body/diagnostics/${mode}.log"
  printf '%s finished %s\n' "$(date -Is)" "$mode" | tee -a results/context_body/diagnostics/run_sequence.log
done
