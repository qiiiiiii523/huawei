#!/usr/bin/env bash
set -euo pipefail

python_bin="${1:?Pass the GPU Python executable as argument 1}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

"$python_bin" -u -B -m scripts.train_b5_body_frozen_adapter \
  --device cuda --epochs 10 --validate-every 5 --batch-size 4 \
  --accumulation 2 --learning-rate 1e-4 \
  2>&1 | tee results/context_body/A_frozen_adapter_v1_train.log
