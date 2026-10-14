#!/usr/bin/env bash
set -euo pipefail

# Independent cross-patient donor permutations; inference only.
python_bin="${1:?Pass the GPU Python executable as argument 1}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

for shuffle_seed in 101 202; do
  "$python_bin" -B -m scripts.evaluate_b5_body_diagnostics \
    --mode A_shuffled --shuffle-seed "$shuffle_seed" --preflight --device cpu
done

for shuffle_seed in 101 202; do
  name="A_shuffled_seed_${shuffle_seed}"
  printf '%s starting %s\n' "$(date -Is)" "$name" | tee -a results/context_body/diagnostics/run_sequence.log
  "$python_bin" -u -B -m scripts.evaluate_b5_body_diagnostics \
    --mode A_shuffled --shuffle-seed "$shuffle_seed" --device cuda \
    --batch-size 4 --steps 16 --samples 16 \
    2>&1 | tee "results/context_body/diagnostics/${name}.log"
  printf '%s finished %s\n' "$(date -Is)" "$name" | tee -a results/context_body/diagnostics/run_sequence.log
done
