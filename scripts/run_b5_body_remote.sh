#!/usr/bin/env bash
set -euo pipefail

# Run only on the checked AutoDL instance. Each group starts independently from
# results/B5/context_start/best.pt; no group resumes from another group.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_dir}/../.venv_gpu/bin/python"
cd "${repo_dir}"
mkdir -p results/context_body

for group in control A B; do
  printf '%s starting %s\n' "$(date -Is)" "${group}" | tee -a results/context_body/run_sequence.log
  "${python_bin}" -B -m scripts.train_b5_body_context \
    --group "${group}" --epochs 20 --batch-size 4 --accumulation 2 \
    --learning-rate 0.00005 --validate-every 5 \
    --steps 16 --samples 16 --device cuda \
    >> "results/context_body/${group}.log" 2>&1
  printf '%s finished %s\n' "$(date -Is)" "${group}" | tee -a results/context_body/run_sequence.log
done
